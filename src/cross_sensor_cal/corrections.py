"""
Portions of this module are adapted from HyTools: Hyperspectral image
processing library (GPLv3).
HyTools Authors: Adam Chlus, Zhiwei Ye, Philip Townsend.
This adapted version is simplified for NEON-only use in cross-sensor-cal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, TYPE_CHECKING

import numpy as np

__all__ = [
    "calc_cosine_i",
    "calc_volume_kernel",
    "calc_geom_kernel",
    "fit_and_save_brdf_model",
    "apply_topo_correct",
    "apply_brdf_correct",
    "apply_glint_correct",
]

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .neon_cube import NeonCube


_BRDF_COEFF_CACHE: dict[Path, dict[str, np.ndarray | str]] = {}


def _validate_angles(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Ensure all angle arrays are float32 with matching shapes."""

    cast_arrays: list[np.ndarray] = []
    reference_shape: Tuple[int, ...] | None = None
    for array in arrays:
        if array.ndim != 2:
            raise ValueError(
                "Angle ancillary rasters must be 2-D arrays with shape (Y, X)."
            )
        if reference_shape is None:
            reference_shape = array.shape
        elif array.shape != reference_shape:
            raise ValueError(
                "All ancillary rasters must share the same shape."
            )
        cast_arrays.append(array.astype(np.float32, copy=False))
    return tuple(cast_arrays)  # type: ignore[return-value]


def calc_cosine_i(
    solar_zn: np.ndarray,
    solar_az: np.ndarray,
    aspect: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    """Adapted from hytools.topo.calc_cosine_i (GPLv3; HyTools Authors:
    Adam Chlus, Zhiwei Ye, Philip Townsend). Simplified for NEON use.

    Compute the cosine of the solar incidence angle on a sloped surface.

    Parameters
    ----------
    solar_zn : np.ndarray
        Solar zenith angle [radians], shape (Y,X)
    solar_az : np.ndarray
        Solar azimuth angle [radians], shape (Y,X)
    aspect : np.ndarray
        Surface aspect [radians], shape (Y,X)
        Aspect is typically clockwise from North.
    slope : np.ndarray
        Surface slope [radians], shape (Y,X)

    Returns
    -------
    np.ndarray
        cos(i), shape (Y,X), float32

    Notes
    -----
    HyTools uses this quantity in topographic correction.
    This is basically:
        cos_i = cos(solar_zn)*cos(slope)
                + sin(solar_zn)*sin(slope)*cos(solar_az - aspect)
    """

    solar_zn, solar_az, aspect, slope = _validate_angles(
        solar_zn, solar_az, aspect, slope
    )

    cos_solar_zn = np.cos(solar_zn)
    sin_solar_zn = np.sin(solar_zn)
    cos_slope = np.cos(slope)
    sin_slope = np.sin(slope)
    cos_relative_az = np.cos(solar_az - aspect)

    cos_i = cos_solar_zn * cos_slope + sin_solar_zn * sin_slope * cos_relative_az
    return cos_i.astype(np.float32, copy=False)


def calc_volume_kernel(
    solar_az: np.ndarray,
    solar_zn: np.ndarray,
    sensor_az: np.ndarray,
    sensor_zn: np.ndarray,
    kernel_type: str,
) -> np.ndarray:
    """Adapted from hytools.brdf.kernels.calc_volume_kernel (GPLv3; HyTools Authors:
    Adam Chlus, Zhiwei Ye, Philip Townsend). Simplified for NEON use.

    Compute the BRDF volume scattering kernel for each pixel.
    """

    if kernel_type.lower() not in {"rossthick", "ross-thick", "ross_thick"}:
        raise NotImplementedError(
            f"Volume kernel '{kernel_type}' is not implemented in cross-sensor-cal."
        )

    solar_az, solar_zn, sensor_az, sensor_zn = _validate_angles(
        solar_az, solar_zn, sensor_az, sensor_zn
    )

    cos_solar = np.cos(solar_zn)
    sin_solar = np.sin(solar_zn)
    cos_sensor = np.cos(sensor_zn)
    sin_sensor = np.sin(sensor_zn)
    cos_rel_az = np.cos(solar_az - sensor_az)

    cos_phase = cos_solar * cos_sensor + sin_solar * sin_sensor * cos_rel_az
    cos_phase = np.clip(cos_phase, -1.0, 1.0)
    phase_angle = np.arccos(cos_phase)

    denom = cos_solar + cos_sensor
    denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)

    kernel = (
        ((np.pi / 2.0 - phase_angle) * cos_phase + np.sin(phase_angle)) / denom
        - np.pi / 4.0
    )

    kernel = np.where(np.isfinite(kernel), kernel, 0.0)
    return kernel.astype(np.float32)


def calc_geom_kernel(
    solar_az: np.ndarray,
    solar_zn: np.ndarray,
    sensor_az: np.ndarray,
    sensor_zn: np.ndarray,
    kernel_type: str,
    b_r: float = 1.0,
    h_b: float = 2.0,
) -> np.ndarray:
    """Adapted from hytools.brdf.kernels.calc_geom_kernel (GPLv3; HyTools Authors:
    Adam Chlus, Zhiwei Ye, Philip Townsend). Simplified for NEON use.

    Compute the BRDF geometric scattering kernel for each pixel.
    """

    if kernel_type.lower() not in {
        "lisparsereciprocal",
        "li-sparse-reciprocal",
        "li_sparse_reciprocal",
    }:
        raise NotImplementedError(
            f"Geometric kernel '{kernel_type}' is not implemented in cross-sensor-cal."
        )

    solar_az, solar_zn, sensor_az, sensor_zn = _validate_angles(
        solar_az, solar_zn, sensor_az, sensor_zn
    )

    cos_solar = np.cos(solar_zn)
    cos_sensor = np.cos(sensor_zn)
    tan_solar = np.tan(solar_zn)
    tan_sensor = np.tan(sensor_zn)
    sec_solar = 1.0 / np.clip(cos_solar, 1e-6, None)
    sec_sensor = 1.0 / np.clip(cos_sensor, 1e-6, None)

    relative_az = solar_az - sensor_az
    cos_rel = np.cos(relative_az)

    tan_term = np.maximum(
        tan_solar**2 + tan_sensor**2 - 2.0 * tan_solar * tan_sensor * cos_rel,
        0.0,
    )
    D = np.sqrt(tan_term)

    cos_t = h_b / np.sqrt(1.0 + D**2)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))

    t_angle = np.arctan(D)
    overlap = (
        (1.0 / np.pi)
        * (
            t_angle
            - sin_t * cos_t
        )
        * (sec_solar + sec_sensor)
    )

    kernel = (
        overlap
        - sec_solar
        - sec_sensor
        + 0.5 * (1.0 + cos_rel) * sec_solar * sec_sensor
    )

    kernel *= b_r
    kernel = np.where(np.isfinite(kernel), kernel, 0.0)
    return kernel.astype(np.float32)


def apply_topo_correct(
    cube,
    chunk_array: np.ndarray,
    ys: int,
    ye: int,
    xs: int,
    xe: int,
) -> np.ndarray:
    """Adapted from hytools.topo.apply_topo_correct (GPLv3; HyTools Authors:
    Adam Chlus, Zhiwei Ye, Philip Townsend). Simplified for NEON use.

    Perform topographic (illumination) correction on a hyperspectral chunk.
    """

    if chunk_array.dtype != np.float32:
        raise ValueError("Chunks passed to apply_topo_correct must be float32 arrays.")

    slope = cube.get_ancillary("slope", radians=True)[ys:ye, xs:xe]
    aspect = cube.get_ancillary("aspect", radians=True)[ys:ye, xs:xe]
    solar_zn = cube.get_ancillary("solar_zn", radians=True)[ys:ye, xs:xe]
    solar_az = cube.get_ancillary("solar_az", radians=True)[ys:ye, xs:xe]

    cos_i = calc_cosine_i(solar_zn, solar_az, aspect, slope)
    cos_solar = np.cos(solar_zn)

    normalization = np.ones_like(cos_i, dtype=np.float32)
    valid = cos_i > 0
    normalization[valid] = cos_solar[valid] / cos_i[valid]

    corrected = chunk_array * normalization[..., np.newaxis]

    if hasattr(cube, "mask_no_data"):
        mask = cube.mask_no_data[ys:ye, xs:xe]
        no_data_value = np.float32(getattr(cube, "no_data", np.nan))
        corrected = np.where(mask[..., np.newaxis], corrected, no_data_value)

    return corrected.astype(np.float32, copy=False)


def apply_brdf_correct(
    cube: "NeonCube",
    chunk_array: np.ndarray,
    ys: int,
    ye: int,
    xs: int,
    xe: int,
    coeff_path: Path | None = None,
) -> np.ndarray:
    """Perform BRDF normalization on a hyperspectral chunk.

    Parameters
    ----------
    cube : NeonCube
        The NEON cube providing ancillary angle rasters.
    chunk_array : np.ndarray
        Reflectance chunk (float32) to normalise.
    ys, ye, xs, xe : int
        Chunk boundaries within the full cube.
    coeff_path : Path | None, optional
        Path to a persisted BRDF coefficient JSON (as produced by
        :func:`fit_and_save_brdf_model`).  When provided, the file is loaded
        once and cached for subsequent calls.  If ``None`` or unavailable,
        fall back to any coefficients already attached to ``cube``; otherwise
        revert to neutral coefficients with a warning.
    """

    if chunk_array.dtype != np.float32:
        raise ValueError("Chunks passed to apply_brdf_correct must be float32 arrays.")

    coeffs_dict: dict[str, np.ndarray | str] | None = None
    coeff_path_resolved: Path | None = None
    if coeff_path is not None:
        coeff_path_resolved = Path(coeff_path).resolve()
        if coeff_path_resolved.exists():
            try:
                coeffs_dict = _BRDF_COEFF_CACHE.get(coeff_path_resolved)
                if coeffs_dict is None:
                    with coeff_path_resolved.open("r", encoding="utf-8") as coeff_file:
                        loaded = json.load(coeff_file)
                    iso_arr = np.asarray(loaded.get("iso"), dtype=np.float32)
                    vol_arr = np.asarray(loaded.get("vol"), dtype=np.float32)
                    geo_arr = np.asarray(loaded.get("geo"), dtype=np.float32)
                    coeffs_dict = {
                        "iso": iso_arr,
                        "vol": vol_arr,
                        "geo": geo_arr,
                        "volume_kernel": loaded.get("volume_kernel", "RossThick"),
                        "geom_kernel": loaded.get("geom_kernel", "LiSparseReciprocal"),
                    }
                    _BRDF_COEFF_CACHE[coeff_path_resolved] = coeffs_dict
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                logging.warning(
                    "⚠️  Failed to load BRDF coefficients from %s (%s); falling back to cube/neutrals.",
                    coeff_path_resolved,
                    exc,
                )
                coeffs_dict = None
        else:
            logging.warning(
                "⚠️  BRDF coefficient file %s not found; falling back to cube/neutrals.",
                coeff_path_resolved,
            )

    if coeffs_dict is None:
        coeffs_attr = getattr(cube, "brdf_coefficients", None)
        if coeffs_attr is not None:
            try:
                iso_arr = np.asarray(coeffs_attr["iso"], dtype=np.float32)
                vol_arr = np.asarray(coeffs_attr["vol"], dtype=np.float32)
                geo_arr = np.asarray(coeffs_attr["geo"], dtype=np.float32)
            except KeyError as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    "BRDF coefficients dictionary must include 'iso', 'vol', 'geo'."
                ) from exc
            coeffs_dict = {
                "iso": iso_arr,
                "vol": vol_arr,
                "geo": geo_arr,
                "volume_kernel": coeffs_attr.get("volume_kernel", "RossThick"),
                "geom_kernel": coeffs_attr.get("geom_kernel", "LiSparseReciprocal"),
            }

    expected_bands = chunk_array.shape[-1]
    if coeffs_dict is None:
        logging.warning(
            "⚠️  No BRDF coefficients available for %s; using neutral coefficients.",
            getattr(cube, "base_key", "unknown"),
        )
        coeffs_dict = {
            "iso": np.ones(expected_bands, dtype=np.float32),
            "vol": np.zeros(expected_bands, dtype=np.float32),
            "geo": np.zeros(expected_bands, dtype=np.float32),
            "volume_kernel": "RossThick",
            "geom_kernel": "LiSparseReciprocal",
        }

    iso = np.asarray(coeffs_dict["iso"], dtype=np.float32)
    vol = np.asarray(coeffs_dict["vol"], dtype=np.float32)
    geo = np.asarray(coeffs_dict["geo"], dtype=np.float32)

    if iso.ndim != 1 or vol.shape != iso.shape or geo.shape != iso.shape:
        raise ValueError("BRDF coefficient arrays must be one-dimensional with matching shapes.")

    if iso.size != expected_bands:
        logging.warning(
            "⚠️  BRDF coefficient size mismatch (%d vs %d); falling back to neutral coefficients.",
            iso.size,
            expected_bands,
        )
        iso = np.ones(expected_bands, dtype=np.float32)
        vol = np.zeros(expected_bands, dtype=np.float32)
        geo = np.zeros(expected_bands, dtype=np.float32)

    kernel_type_vol = str(coeffs_dict.get("volume_kernel", "RossThick"))
    kernel_type_geo = str(coeffs_dict.get("geom_kernel", "LiSparseReciprocal"))

    solar_zn = cube.get_ancillary("solar_zn", radians=True)[ys:ye, xs:xe]
    solar_az = cube.get_ancillary("solar_az", radians=True)[ys:ye, xs:xe]
    sensor_zn = cube.get_ancillary("sensor_zn", radians=True)[ys:ye, xs:xe]
    sensor_az = cube.get_ancillary("sensor_az", radians=True)[ys:ye, xs:xe]

    volume_kernel = calc_volume_kernel(
        solar_az,
        solar_zn,
        sensor_az,
        sensor_zn,
        kernel_type=kernel_type_vol,
    )
    geom_kernel = calc_geom_kernel(
        solar_az,
        solar_zn,
        sensor_az,
        sensor_zn,
        kernel_type=kernel_type_geo,
    )

    denominator = (
        iso[np.newaxis, np.newaxis, :]
        + vol[np.newaxis, np.newaxis, :] * volume_kernel[..., np.newaxis]
        + geo[np.newaxis, np.newaxis, :] * geom_kernel[..., np.newaxis]
    )

    denominator = np.where(np.abs(denominator) < 1e-6, np.nan, denominator)

    corrected = chunk_array / denominator
    corrected = np.where(np.isfinite(corrected), corrected, chunk_array)

    if hasattr(cube, "mask_no_data"):
        mask = cube.mask_no_data[ys:ye, xs:xe]
        no_data_value = np.float32(getattr(cube, "no_data", np.nan))
        corrected = np.where(mask[..., np.newaxis], corrected, no_data_value)

    return corrected.astype(np.float32, copy=False)


def apply_glint_correct(*args, **kwargs):
    """Placeholder for sunglint correction.

    In hytools.glint.apply_glint_correct (GPLv3), sunglint correction is used
    for aquatic environments. We are not supporting that right now.
    """

    raise NotImplementedError("Glint correction is not implemented in cross-sensor-cal.")


def fit_and_save_brdf_model(cube: "NeonCube", out_dir: Path) -> Path:
    """Estimate and persist BRDF coefficients for a NEON flightline.

    Parameters
    ----------
    cube : NeonCube
        Loaded NEON hyperspectral cube providing reflectance and ancillary angles.
    out_dir : Path
        Directory where the fitted BRDF coefficient JSON should be stored.

    Returns
    -------
    Path
        Path to the JSON file containing the BRDF model coefficients.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coeff_path = out_dir / f"{cube.base_key}_brdf_model.json"
    if coeff_path.exists():
        return coeff_path

    solar_zn = cube.get_ancillary("solar_zn", radians=True)
    solar_az = cube.get_ancillary("solar_az", radians=True)
    sensor_zn = cube.get_ancillary("sensor_zn", radians=True)
    sensor_az = cube.get_ancillary("sensor_az", radians=True)

    volume_kernel = calc_volume_kernel(
        solar_az, solar_zn, sensor_az, sensor_zn, kernel_type="RossThick"
    )
    geom_kernel = calc_geom_kernel(
        solar_az,
        solar_zn,
        sensor_az,
        sensor_zn,
        kernel_type="LiSparseReciprocal",
    )

    mask = getattr(cube, "mask_no_data", None)
    if mask is None:
        mask = np.ones_like(volume_kernel, dtype=bool)
    valid = mask.astype(bool)
    valid &= np.isfinite(volume_kernel) & np.isfinite(geom_kernel)

    flat_valid = valid.reshape(-1)
    if np.count_nonzero(flat_valid) < 3:
        logging.warning(
            "⚠️  Not enough valid pixels to fit BRDF model for %s; using neutral coefficients.",
            getattr(cube, "base_key", "unknown"),
        )
        iso = np.ones(cube.bands, dtype=np.float32)
        vol = np.zeros(cube.bands, dtype=np.float32)
        geo = np.zeros(cube.bands, dtype=np.float32)
    else:
        design_stack = np.stack(
            [
                np.ones_like(volume_kernel, dtype=np.float32),
                volume_kernel,
                geom_kernel,
            ],
            axis=-1,
        )
        design_flat = design_stack.reshape(-1, 3)
        design_valid = design_flat[flat_valid]

        reflectance = np.asarray(cube.data, dtype=np.float32).reshape(-1, cube.bands)

        iso = np.ones(cube.bands, dtype=np.float32)
        vol = np.zeros(cube.bands, dtype=np.float32)
        geo = np.zeros(cube.bands, dtype=np.float32)

        for band_idx in range(cube.bands):
            y = reflectance[flat_valid, band_idx].astype(np.float64, copy=False)
            finite_mask = np.isfinite(y)
            if np.count_nonzero(finite_mask) < 3:
                continue
            X = design_valid[finite_mask].astype(np.float64, copy=False)
            y_valid = y[finite_mask]
            try:
                solution, *_ = np.linalg.lstsq(X, y_valid, rcond=None)
            except np.linalg.LinAlgError:
                continue
            iso[band_idx] = np.float32(solution[0])
            vol[band_idx] = np.float32(solution[1])
            geo[band_idx] = np.float32(solution[2])

    coeff_payload = {
        "iso": iso.astype(float).tolist(),
        "vol": vol.astype(float).tolist(),
        "geo": geo.astype(float).tolist(),
        "volume_kernel": "RossThick",
        "geom_kernel": "LiSparseReciprocal",
    }

    with coeff_path.open("w", encoding="utf-8") as coeff_file:
        json.dump(coeff_payload, coeff_file, indent=2)

    return coeff_path
