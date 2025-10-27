"""
Portions of this module are adapted from HyTools: Hyperspectral image
processing library (GPLv3).
HyTools Authors: Adam Chlus, Zhiwei Ye, Philip Townsend.
This adapted version is simplified for NEON-only use in cross-sensor-cal.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "calc_cosine_i",
    "calc_volume_kernel",
    "calc_geom_kernel",
    "apply_topo_correct",
    "apply_brdf_correct",
    "apply_glint_correct",
]


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
    cube,
    chunk_array: np.ndarray,
    ys: int,
    ye: int,
    xs: int,
    xe: int,
) -> np.ndarray:
    """Adapted from hytools.brdf.apply_brdf_correct (GPLv3; HyTools Authors:
    Adam Chlus, Zhiwei Ye, Philip Townsend). Simplified for NEON use.

    Perform BRDF normalization on a hyperspectral chunk.
    """

    if chunk_array.dtype != np.float32:
        raise ValueError("Chunks passed to apply_brdf_correct must be float32 arrays.")

    coeffs = getattr(cube, "brdf_coefficients", None)
    if coeffs is None:
        raise ValueError(
            "NeonCube does not provide 'brdf_coefficients'. BRDF correction cannot proceed."
        )

    try:
        iso = np.asarray(coeffs["iso"], dtype=np.float32)
        vol = np.asarray(coeffs["vol"], dtype=np.float32)
        geo = np.asarray(coeffs["geo"], dtype=np.float32)
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError("BRDF coefficients dictionary must include 'iso', 'vol', 'geo'.") from exc

    if iso.ndim != 1 or vol.shape != iso.shape or geo.shape != iso.shape:
        raise ValueError("BRDF coefficient arrays must be one-dimensional with matching shapes.")

    solar_zn = cube.get_ancillary("solar_zn", radians=True)[ys:ye, xs:xe]
    solar_az = cube.get_ancillary("solar_az", radians=True)[ys:ye, xs:xe]
    sensor_zn = cube.get_ancillary("sensor_zn", radians=True)[ys:ye, xs:xe]
    sensor_az = cube.get_ancillary("sensor_az", radians=True)[ys:ye, xs:xe]

    kernel_type_vol = coeffs.get("volume_kernel", "RossThick")
    kernel_type_geo = coeffs.get("geom_kernel", "LiSparseReciprocal")

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
