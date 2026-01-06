from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np


def _prepare_cube(cube: np.ndarray) -> tuple[np.ndarray, bool]:
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError("cube must be 3-D with bands and spatial dimensions")
    # Prefer band-first arrays; move axis if necessary.
    if cube.shape[0] <= cube.shape[-1]:
        return cube.astype(np.float32, copy=True), False
    return np.moveaxis(cube, -1, 0).astype(np.float32, copy=True), True


def _prepare_mask(mask: Optional[np.ndarray], shape: tuple[int, int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape == shape:
        return mask_arr
    if mask_arr.shape == shape[1:]:
        return np.broadcast_to(mask_arr, shape)
    raise ValueError("mask shape must match cube or spatial dimensions")


def _band_percentiles(data: np.ndarray, mask: np.ndarray, p_lo: float, p_hi: float) -> tuple[np.ndarray, np.ndarray]:
    valid = data.reshape(data.shape[0], -1)
    mask_flat = mask.reshape(mask.shape[0], -1)
    p_lo_vals = np.empty(data.shape[0], dtype=np.float32)
    p_hi_vals = np.empty_like(p_lo_vals)
    for i in range(data.shape[0]):
        band = valid[i]
        m = mask_flat[i]
        if not np.any(m):
            p_lo_vals[i] = np.nan
            p_hi_vals[i] = np.nan
            continue
        band_valid = band[m]
        p_lo_vals[i] = np.nanpercentile(band_valid, p_lo)
        p_hi_vals[i] = np.nanpercentile(band_valid, p_hi)
    return p_lo_vals, p_hi_vals


def _solve_linear_regression(
    cube: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    huber_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    gains = np.ones(cube.shape[0], dtype=np.float32)
    offsets = np.zeros_like(gains)
    flat_cube = cube.reshape(cube.shape[0], -1)
    flat_ref = reference.reshape(reference.shape[0], -1)
    mask_flat = mask.reshape(mask.shape[0], -1)
    for i in range(cube.shape[0]):
        m = mask_flat[i]
        if not np.any(m):
            continue
        x = flat_cube[i, m]
        y = flat_ref[i, m]
        if x.size == 0:
            continue
        # Compute robust slope/offset with simple Huber weighting.
        x_mean = np.nanmean(x)
        y_mean = np.nanmean(y)
        cov = np.nanmean((x - x_mean) * (y - y_mean))
        var = np.nanmean((x - x_mean) ** 2)
        if not np.isfinite(cov) or not np.isfinite(var) or var == 0:
            continue
        slope = cov / max(var, 1e-6)
        residual = y - (slope * x + y_mean - slope * x_mean)
        weight = np.minimum(1.0, huber_delta / (np.abs(residual) + 1e-6))
        if np.any(weight > 0):
            slope = np.nanmean(weight * y * x) / max(np.nanmean(weight * x * x), 1e-6)
        intercept = y_mean - slope * x_mean
        gains[i] = np.float32(slope)
        offsets[i] = np.float32(intercept)
    return gains, offsets


def apply_brightness_correction(
    cube: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    method: Literal["percentile_match", "linear_regression", "unit_gain_offset_only"] = "percentile_match",
    reference_cube: Optional[np.ndarray] = None,
    reference_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    p_lo: float = 5.0,
    p_hi: float = 95.0,
    clip_min: float = 0.0,
    clip_max: float = 1.2,
    huber_delta: float = 0.1,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Apply a per-band brightness correction to hyperspectral/multispectral data.

    This normalizes scene brightness under varying illumination (sun angle, terrain shadow)
    or sensor gain drift. Typically used after ENVI export and before BRDF correction.

    Mathematical form
    -----------------
    For band b:
        y_corr[b] = gain[b] * y_raw[b] + offset[b]
    where gain/offset come from robust percentile matching or regression vs a reference.

    Parameters
    ----------
    cube : np.ndarray
        Array shaped (B, Y, X) or (Y, X, B).
    mask : np.ndarray, optional
        Boolean mask: True = valid pixel used for stats.
    method : {'percentile_match', 'linear_regression', 'unit_gain_offset_only'}, default='percentile_match'
        Strategy for estimating per-band gain/offset.
    reference_cube : np.ndarray, optional
        Reference to match brightness against (required for 'linear_regression').
    reference_stats : tuple[np.ndarray, np.ndarray], optional
        (p_lo, p_hi) per band for reference distribution (each shape (B,)).
    p_lo, p_hi : float
        Robust percentile bounds, default (5, 95).
    clip_min, clip_max : float
        Optional post-correction clipping, default (0.0, 1.2).
    huber_delta : float
        Huber delta for robust regression.

    Returns
    -------
    np.ndarray
        Corrected cube with same shape as input.
    dict
        Metadata with per-band gain/offset and diagnostics.

    Examples
    --------
    >>> corr, meta = apply_brightness_correction(cube, mask=mask)
    >>> corr2, meta = apply_brightness_correction(cube, reference_cube=panel, method='linear_regression')

    Notes
    -----
    - NaNs preserved. Bandwise affine transform.
    - See also: topo+BRDF correction stage and QA JSON (gain/offset are included there when run).
    """

    working, transposed = _prepare_cube(cube)
    mask_arr = _prepare_mask(mask, working.shape)
    metadata: dict[str, np.ndarray] = {}

    # Compute robust per-band stats.
    cube_lo, cube_hi = _band_percentiles(working, mask_arr, p_lo, p_hi)

    if method == "unit_gain_offset_only":
        gains = np.ones(working.shape[0], dtype=np.float32)
        offsets = np.zeros_like(gains)
    elif method == "percentile_match":
        if reference_stats is not None:
            ref_lo, ref_hi = reference_stats
        elif reference_cube is not None:
            ref_working, _ = _prepare_cube(reference_cube)
            if ref_working.shape[0] != working.shape[0]:
                raise ValueError("reference_cube must share band dimension with cube")
            ref_mask = _prepare_mask(mask, ref_working.shape)
            ref_lo, ref_hi = _band_percentiles(ref_working, ref_mask, p_lo, p_hi)
        else:
            ref_lo, ref_hi = cube_lo.copy(), cube_hi.copy()
        gains = (ref_hi - ref_lo) / np.maximum(cube_hi - cube_lo, 1e-6)
        offsets = ref_lo - gains * cube_lo
    elif method == "linear_regression":
        if reference_cube is None:
            raise ValueError("reference_cube is required for linear_regression method")
        ref_working, _ = _prepare_cube(reference_cube)
        if ref_working.shape != working.shape:
            raise ValueError("reference_cube must match cube shape for regression")
        gains, offsets = _solve_linear_regression(working, ref_working, mask_arr, huber_delta)
    else:
        raise ValueError(f"Unknown method: {method}")

    metadata["gain"] = gains
    metadata["offset"] = offsets
    metadata["p_lo"] = cube_lo
    metadata["p_hi"] = cube_hi

    # Apply affine per band.
    corrected = working * gains[:, None, None] + offsets[:, None, None]

    # Preserve NaNs & optional clip.
    corrected[~np.isfinite(working)] = np.nan
    if clip_min is not None or clip_max is not None:
        corrected = np.clip(corrected, clip_min, clip_max)

    # Return cube & metadata.
    if transposed:
        corrected = np.moveaxis(corrected, 0, -1)
    return corrected.astype(np.float32, copy=False), metadata


__all__ = ["apply_brightness_correction"]
