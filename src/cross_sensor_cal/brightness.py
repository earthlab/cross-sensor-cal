"""Utilities for brightness normalization of spectral cubes."""
from __future__ import annotations

from typing import Any

import numpy as np


def _bands_first(array: np.ndarray) -> tuple[np.ndarray, int]:
    """Return an array with bands as the leading dimension and the original axis."""
    if array.ndim != 3:
        raise ValueError("Expected a 3-D spectral cube")
    band_axis = 0
    if array.shape[-1] < array.shape[0] and array.shape[-1] < array.shape[1]:
        band_axis = array.ndim - 1
    return np.moveaxis(array, band_axis, 0), band_axis


def apply_brightness_correction(
    cube: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    method: str = "percentile_match",
    reference_cube: np.ndarray | None = None,
    reference_stats: tuple[np.ndarray, np.ndarray] | None = None,
    p_lo: float = 5.0,
    p_hi: float = 95.0,
    clip_min: float | None = 0.0,
    clip_max: float | None = 1.2,
    huber_delta: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply a per-band brightness correction to hyperspectral or multispectral data.

    This function rescales pixel values (reflectance or radiance) to normalize 
    scene brightness across varying illumination conditions (e.g., sun angle, 
    terrain shadow, or sensor gain drift). It can be run on corrected ENVI cubes 
    prior to BRDF correction.

    Mathematical form
    -----------------
    Each spectral band *b* is transformed as:

        y_corr[b] = gain[b] * y_raw[b] + offset[b]

    where `gain` and `offset` are estimated from robust statistics or regression 
    against a reference brightness distribution.

    Parameters
    ----------
    cube : np.ndarray
        Input data array of shape (B, Y, X) or (Y, X, B). 
    mask : np.ndarray, optional
        Boolean mask of valid pixels (True = use). Pixels outside mask are ignored.
    method : {'percentile_match', 'linear_regression', 'unit_gain_offset_only'}, default='percentile_match'
        Estimation strategy for gain/offset.
    reference_cube : np.ndarray, optional
        Reference data to match brightness against.
    reference_stats : tuple[np.ndarray, np.ndarray], optional
        Precomputed (p_lo, p_hi) percentiles per band for reference.
    p_lo, p_hi : float, optional
        Lower and upper percentiles for robust matching (default 5, 95).
    clip_min, clip_max : float, optional
        Optional clipping limits on corrected reflectance (default 0.0–1.2).
    huber_delta : float, optional
        Robust regression parameter (if using linear regression).

    Returns
    -------
    np.ndarray
        Brightness-corrected cube with same shape as input.
    dict
        Metadata dictionary including per-band gain, offset, and diagnostic stats.

    Examples
    --------
    >>> cube_corr, meta = apply_brightness_correction(cube, mask=mask)
    >>> print(meta['gain'][:5])
    >>> # For a known reference target
    >>> cube_corr, meta = apply_brightness_correction(cube, reference_cube=panel, method='linear_regression')

    Notes
    -----
    - All NaN pixels are preserved.
    - Gain/offset are computed per band, not globally.
    - Typical usage: after ENVI export but before BRDF correction.
    - See also: `cross_sensor_cal.topo_and_brdf_correction` for subsequent correction.
    """
    allowed_methods = {"percentile_match", "linear_regression", "unit_gain_offset_only"}
    if method not in allowed_methods:
        raise ValueError(f"Unknown brightness correction method: {method}")

    cube_np = np.asarray(cube, dtype=float)
    cube_bands_first, original_band_axis = _bands_first(cube_np)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != cube_np.shape:
            raise ValueError("Mask must have the same shape as cube")
        mask_bands_first, _ = _bands_first(mask_arr.astype(float))
        mask_bands_first = mask_bands_first.astype(bool)
    else:
        mask_bands_first = None

    valid_pixels = ~np.isnan(cube_bands_first)
    if mask_bands_first is not None:
        valid_pixels &= mask_bands_first
    valid_pixels = valid_pixels.astype(bool)

    flat_valid = np.where(valid_pixels, cube_bands_first, np.nan)

    # Compute robust per-band statistics (percentile match)
    obs_lo = np.nanpercentile(flat_valid, p_lo, axis=(1, 2))
    obs_hi = np.nanpercentile(flat_valid, p_hi, axis=(1, 2))

    ref_lo: np.ndarray
    ref_hi: np.ndarray
    if reference_stats is not None:
        ref_lo, ref_hi = reference_stats
        ref_lo = np.asarray(ref_lo, dtype=float)
        ref_hi = np.asarray(ref_hi, dtype=float)
    elif reference_cube is not None:
        ref_cube_np = np.asarray(reference_cube, dtype=float)
        ref_bands_first, _ = _bands_first(ref_cube_np)
        if ref_bands_first.shape != cube_bands_first.shape:
            raise ValueError("Reference cube must match cube shape for regression")
        ref_lo = np.nanpercentile(ref_bands_first, p_lo, axis=(1, 2))
        ref_hi = np.nanpercentile(ref_bands_first, p_hi, axis=(1, 2))
    else:
        ref_lo = obs_lo.copy()
        ref_hi = obs_hi.copy()

    bands = cube_bands_first.shape[0]
    gain = np.ones(bands, dtype=float)
    offset = np.zeros(bands, dtype=float)

    if method == "percentile_match":
        denom = obs_hi - obs_lo
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        gain = (ref_hi - ref_lo) / denom
        offset = ref_lo - gain * obs_lo
    elif method == "linear_regression" and reference_cube is not None:
        ref_cube_np = np.asarray(reference_cube, dtype=float)
        ref_bands_first, _ = _bands_first(ref_cube_np)
        for idx in range(bands):
            band_mask = valid_pixels[idx]
            x = cube_bands_first[idx][band_mask]
            y = ref_bands_first[idx][band_mask]
            if x.size == 0:
                continue
            A = np.vstack([x, np.ones_like(x)]).T
            sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            gain[idx] = sol[0]
            offset[idx] = sol[1]
    elif method == "unit_gain_offset_only":
        gain = np.ones(bands, dtype=float)
        offset = np.zeros(bands, dtype=float)

    # Apply affine transform per band (gain * pixel + offset)
    corrected_bands_first = gain[:, None, None] * cube_bands_first + offset[:, None, None]

    # Preserve NaN mask and clip values if requested
    corrected_bands_first = np.where(valid_pixels, corrected_bands_first, np.nan)
    if clip_min is not None or clip_max is not None:
        lower = -np.inf if clip_min is None else clip_min
        upper = np.inf if clip_max is None else clip_max
        corrected_bands_first = np.clip(corrected_bands_first, lower, upper)

    corrected_cube = np.moveaxis(corrected_bands_first, 0, original_band_axis)

    metadata: dict[str, Any] = {
        "gain": gain,
        "offset": offset,
        "method": method,
        "p_lo": p_lo,
        "p_hi": p_hi,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "huber_delta": huber_delta,
        "valid_fraction": float(np.mean(valid_pixels)),
        "reference_stats": (ref_lo, ref_hi),
    }

    # Return corrected cube and metadata for QA panel
    return corrected_cube, metadata
