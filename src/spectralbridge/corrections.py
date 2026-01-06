"""
Portions of this module are adapted from HyTools: Hyperspectral image
processing library (GPLv3).
HyTools Authors: Adam Chlus, Zhiwei Ye, Philip Townsend.
This adapted version is simplified for NEON-only use in cross-sensor-cal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, TYPE_CHECKING

import numpy as np

from spectralbridge.paths import normalize_brdf_model_path, scene_prefix_from_dir

__all__ = [
    "calc_cosine_i",
    "calc_volume_kernel",
    "calc_geom_kernel",
    "load_brdf_model",
    "fit_and_save_brdf_model",
    "apply_topo_correct",
    "apply_brdf_correct",
    "apply_glint_correct",
]

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .neon_cube import NeonCube


_BRDF_COEFF_CACHE: dict[Path, dict[str, np.ndarray | str]] = {}
MAX_UNITLESS_REFLECTANCE = 1.2
MIN_SCS_C_DENOM = 1e-3


@dataclass
class NDVIBinningConfig:
    """Configuration for NDVI binning used during BRDF fitting and application."""

    ndvi_min: float = 0.05
    ndvi_max: float = 1.0
    n_bins: int = 25
    perc_min: float | None = 10.0
    perc_max: float | None = 95.0


@dataclass
class ReferenceGeometry:
    """Reference geometry (degrees) for BRDF normalization."""

    solar_zenith_deg: float = 45.0
    view_zenith_deg: float = 0.0
    relative_azimuth_deg: float = 0.0


def log_stats(name: str, arr: np.ndarray, mask: np.ndarray | None = None) -> None:
    """Log min/max and validity fractions for quick debugging."""

    if mask is None:
        mask = np.ones_like(arr, dtype=bool)
    finite = np.isfinite(arr) & mask
    if finite.size == 0:
        logging.debug("%s: empty array", name)
        return
    fraction_valid = np.count_nonzero(finite) / finite.size
    if np.count_nonzero(finite) == 0:
        logging.debug("%s: all non-finite (mask fraction=%.5f)", name, fraction_valid)
        return
    subset = arr[finite]
    logging.debug(
        "%s: min=%.6f max=%.6f valid_fraction=%.5f",
        name,
        float(np.nanmin(subset)),
        float(np.nanmax(subset)),
        fraction_valid,
    )


def _scs_c_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    min_denom: float = MIN_SCS_C_DENOM,
) -> np.ndarray:
    """Internal helper applying the SCS+C denominator guard.

    Denominators below ``min_denom`` are forced to a neutral ratio of 1.0 to
    avoid exploding corrections. Caller is responsible for any logging.
    """

    ratio = np.ones_like(numerator, dtype=np.float32)
    valid = denominator > min_denom
    ratio[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return ratio


def load_brdf_model(flightline_dir: Path) -> dict:
    """Load the BRDF model JSON for ``flightline_dir`` after normalizing its name."""

    flightline_dir = Path(flightline_dir)
    prefix = scene_prefix_from_dir(flightline_dir)
    preferred = flightline_dir / f"{prefix}_brdf_model.json"
    actual = normalize_brdf_model_path(flightline_dir) or preferred
    if not actual.exists():
        raise FileNotFoundError(
            f"BRDF model not found for {prefix}: expected {preferred.name}"
        )
    return json.loads(actual.read_text(encoding="utf-8"))


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


def _select_band_for_wavelength(cube: "NeonCube", target_nm: float) -> int:
    """Return the band index closest to ``target_nm``."""

    diffs = np.abs(np.asarray(cube.wavelengths, dtype=np.float32) - np.float32(target_nm))
    return int(np.argmin(diffs))


def compute_ndvi(
    cube: "NeonCube",
    data_unitless: np.ndarray,
    red_target_nm: float = 665.0,
    nir_target_nm: float = 865.0,
) -> np.ndarray:
    """Compute NDVI from unitless reflectance data."""

    red_idx = _select_band_for_wavelength(cube, red_target_nm)
    nir_idx = _select_band_for_wavelength(cube, nir_target_nm)
    red = data_unitless[..., red_idx]
    nir = data_unitless[..., nir_idx]
    denom = nir + red
    ndvi = (nir - red) / np.where(np.abs(denom) > 1e-6, denom, np.nan)
    return ndvi.astype(np.float32)


def compute_ndvi_bins(ndvi: np.ndarray, config: NDVIBinningConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return NDVI bin edges and bin indices."""

    ndvi_clean = ndvi[np.isfinite(ndvi)]
    ndvi_min = config.ndvi_min
    ndvi_max = config.ndvi_max
    if ndvi_clean.size > 0:
        if config.perc_min is not None:
            ndvi_min = max(ndvi_min, float(np.nanpercentile(ndvi_clean, config.perc_min)))
        if config.perc_max is not None:
            ndvi_max = min(ndvi_max, float(np.nanpercentile(ndvi_clean, config.perc_max)))
    edges = np.linspace(ndvi_min, ndvi_max, config.n_bins + 1, dtype=np.float32)
    bins = np.digitize(ndvi, edges, right=False) - 1
    bins = np.clip(bins, 0, config.n_bins - 1)
    bins[(ndvi < ndvi_min) | (ndvi > ndvi_max)] = -1
    return edges, bins.astype(np.int32)


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
    use_scs_c: bool = True,
) -> np.ndarray:
    """Topographic (illumination) correction on a hyperspectral chunk."""

    if chunk_array.dtype != np.float32:
        raise ValueError("Chunks passed to apply_topo_correct must be float32 arrays.")

    slope = cube.get_ancillary("slope", radians=True)[ys:ye, xs:xe]
    aspect = cube.get_ancillary("aspect", radians=True)[ys:ye, xs:xe]
    solar_zn = cube.get_ancillary("solar_zn", radians=True)[ys:ye, xs:xe]
    solar_az = cube.get_ancillary("solar_az", radians=True)[ys:ye, xs:xe]

    cos_i = calc_cosine_i(solar_zn, solar_az, aspect, slope)
    cos_solar = np.cos(solar_zn)
    cos_beta = np.cos(slope)

    scale_factor = float(getattr(cube, "scale_factor", 1.0)) or 1.0
    data_unitless = chunk_array.astype(np.float32, copy=False) * np.float32(scale_factor)

    if hasattr(cube, "mask_no_data"):
        valid_mask = cube.mask_no_data[ys:ye, xs:xe].astype(bool)
    else:
        valid_mask = np.ones_like(cos_i, dtype=bool)

    valid_mask &= np.isfinite(data_unitless).all(axis=-1)
    valid_mask &= np.isfinite(cos_i)

    corrected_unitless = data_unitless.copy()

    if use_scs_c:
        cos_i_valid = cos_i[valid_mask]
        cos_solar_valid = cos_solar[valid_mask]
        cos_beta_valid = cos_beta[valid_mask]
        for band in range(data_unitless.shape[-1]):
            rho_band = data_unitless[..., band]
            y = rho_band[valid_mask].astype(np.float64, copy=False)
            x = cos_i_valid.astype(np.float64, copy=False)
            finite = np.isfinite(y) & np.isfinite(x)
            if np.count_nonzero(finite) < 2:
                logging.debug("Band %d: insufficient samples for SCS+C; using neutral", band)
                continue
            X = np.stack([x[finite], np.ones_like(x[finite])], axis=1)
            try:
                coeffs, *_ = np.linalg.lstsq(X, y[finite], rcond=None)
                a, b = coeffs
            except np.linalg.LinAlgError:
                logging.debug("Band %d: regression failed; using neutral", band)
                continue
            if np.isclose(a, 0.0):
                C_val = 0.0
                logging.debug("Band %d: regression slope near zero; C set to 0", band)
            else:
                C_val = float(b / a)
            num = cos_solar_valid * cos_beta_valid + C_val
            den = cos_i_valid + C_val
            min_denom = MIN_SCS_C_DENOM
            tiny = den > 0
            ratio_valid = _scs_c_ratio(num, den, min_denom=min_denom)
            if np.any(tiny & (den <= min_denom)):
                logging.debug(
                    "Band %d: %.5f fraction of pixels hit denom<=%.1e guard",
                    band,
                    np.count_nonzero(tiny & (den <= min_denom)) / tiny.size,
                    min_denom,
                )
            ratios_band = np.ones_like(cos_i, dtype=np.float32)
            ratios_band[valid_mask] = ratio_valid.astype(np.float32)
            log_stats(
                f"topo_ratio_band{band}",
                ratios_band,
                mask=valid_mask,
            )
            corrected_unitless[..., band] = rho_band * ratios_band
    else:
        normalization = np.ones_like(cos_i, dtype=np.float32)
        valid_cos = cos_i > 0
        normalization[valid_cos] = cos_solar[valid_cos] / cos_i[valid_cos]
        corrected_unitless = data_unitless * normalization[..., np.newaxis]

    no_data_value = np.float32(getattr(cube, "no_data", np.nan))
    corrected_scaled = corrected_unitless / np.float32(scale_factor)
    corrected_scaled = np.where(valid_mask[..., np.newaxis], corrected_scaled, no_data_value)

    return corrected_scaled.astype(np.float32, copy=False)


def apply_brdf_correct(
    cube: "NeonCube",
    chunk_array: np.ndarray,
    ys: int,
    ye: int,
    xs: int,
    xe: int,
    coeff_path: Path | None = None,
    ndvi_config: NDVIBinningConfig | None = None,
    reference_geometry: ReferenceGeometry | None = None,
) -> np.ndarray:
    """Perform BRDF normalization on a hyperspectral chunk using FlexBRDF ratio."""

    if chunk_array.dtype != np.float32:
        raise ValueError("Chunks passed to apply_brdf_correct must be float32 arrays.")

    ndvi_config = ndvi_config or NDVIBinningConfig()
    reference_geometry = reference_geometry or ReferenceGeometry()

    scale_factor = float(getattr(cube, "scale_factor", 1.0)) or 1.0

    coeffs_dict: dict[str, np.ndarray | str] | None = None
    coeff_path_resolved: Path | None = None
    if coeff_path is not None:
        coeff_path_resolved = Path(coeff_path).resolve()
        flightline_dir = coeff_path_resolved.parent
        prefix = scene_prefix_from_dir(flightline_dir)
        preferred = flightline_dir / f"{prefix}_brdf_model.json"
        normalized_path = normalize_brdf_model_path(flightline_dir)
        if normalized_path is not None:
            coeff_path_resolved = normalized_path.resolve()
        elif preferred.exists():
            coeff_path_resolved = preferred.resolve()

        if coeff_path_resolved.exists():
            try:
                coeffs_dict = _BRDF_COEFF_CACHE.get(coeff_path_resolved)
                if coeffs_dict is None:
                    loaded = load_brdf_model(flightline_dir)
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
            except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
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

    chunk_unitless = chunk_array * np.float32(scale_factor)
    ndvi = compute_ndvi(cube, chunk_unitless)
    ndvi_edges_raw = coeffs_dict.get("ndvi_edges")
    edges = (
        np.asarray(ndvi_edges_raw, dtype=np.float32)
        if ndvi_edges_raw is not None
        else np.array([], dtype=np.float32)
    )
    if edges.size < 2:
        edges, bin_idx = compute_ndvi_bins(ndvi, ndvi_config)
    else:
        bin_idx = np.digitize(ndvi, edges, right=False) - 1
        bin_idx[(ndvi < edges[0]) | (ndvi > edges[-1])] = -1

    n_bins = edges.size - 1 if edges.size >= 2 else ndvi_config.n_bins
    bin_idx_safe = bin_idx.copy()
    if np.any(bin_idx_safe < 0):
        logging.debug(
            "NDVI binning: %d pixels outside [%0.3f,%0.3f]; assigning to bin 0",
            np.count_nonzero(bin_idx_safe < 0),
            edges[0] if edges.size else ndvi_config.ndvi_min,
            edges[-1] if edges.size else ndvi_config.ndvi_max,
        )
        bin_idx_safe[bin_idx_safe < 0] = 0

    iso = np.asarray(coeffs_dict["iso"], dtype=np.float32)
    vol = np.asarray(coeffs_dict["vol"], dtype=np.float32)
    geo = np.asarray(coeffs_dict["geo"], dtype=np.float32)

    if iso.ndim == 1:
        iso = iso[np.newaxis, :]
        vol = vol[np.newaxis, :]
        geo = geo[np.newaxis, :]

    if iso.shape[1] != expected_bands:
        logging.warning(
            "⚠️  BRDF coefficient size mismatch (%d vs %d); falling back to neutral coefficients.",
            iso.shape[1],
            expected_bands,
        )
        iso = np.ones((1, expected_bands), dtype=np.float32)
        vol = np.zeros_like(iso)
        geo = np.zeros_like(iso)

    if iso.shape[0] != n_bins:
        logging.warning(
            "⚠️  BRDF coefficient NDVI bin mismatch (%d vs %d); broadcasting neutral coefficients across bins.",
            iso.shape[0],
            n_bins,
        )
        iso = np.ones((n_bins, expected_bands), dtype=np.float32)
        vol = np.zeros_like(iso)
        geo = np.zeros_like(iso)

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

    ref_solar = np.deg2rad(reference_geometry.solar_zenith_deg)
    ref_view = np.deg2rad(reference_geometry.view_zenith_deg)
    ref_rel = np.deg2rad(reference_geometry.relative_azimuth_deg)
    ref_solar_az = np.zeros_like(solar_az) + ref_rel
    ref_sensor_az = np.zeros_like(solar_az)
    ref_volume = calc_volume_kernel(
        ref_solar_az,
        np.full_like(solar_zn, ref_solar),
        ref_sensor_az,
        np.full_like(sensor_zn, ref_view),
        kernel_type=kernel_type_vol,
    )
    ref_geom = calc_geom_kernel(
        ref_solar_az,
        np.full_like(solar_zn, ref_solar),
        ref_sensor_az,
        np.full_like(sensor_zn, ref_view),
        kernel_type=kernel_type_geo,
    )

    valid_mask = np.isfinite(chunk_unitless)
    if hasattr(cube, "mask_no_data"):
        valid_mask &= cube.mask_no_data[ys:ye, xs:xe][..., np.newaxis]

    corrected_unitless = np.full_like(chunk_unitless, np.nan, dtype=np.float32)
    for b in range(expected_bands):
        iso_band = iso[:, b]
        vol_band = vol[:, b]
        geo_band = geo[:, b]
        for bin_id in range(iso.shape[0]):
            mask_bin = bin_idx_safe == bin_id
            if not np.any(mask_bin):
                continue
            R_pix = (
                iso_band[bin_id]
                + vol_band[bin_id] * volume_kernel
                + geo_band[bin_id] * geom_kernel
            )
            R_ref = (
                iso_band[bin_id]
                + vol_band[bin_id] * ref_volume
                + geo_band[bin_id] * ref_geom
            )
            cf = np.where((R_pix > 0) & (R_ref > 0), R_ref / R_pix, np.nan)
            if bin_id == 0 and b == 0:
                log_stats("cf_bin0_band0", cf, mask_bin)
            target_mask = mask_bin & valid_mask[..., b]
            corrected_unitless[..., b] = np.where(
                target_mask,
                chunk_unitless[..., b] * cf,
                corrected_unitless[..., b],
            )

    no_data_value = np.float32(getattr(cube, "no_data", np.nan))
    corrected_scaled = corrected_unitless / np.float32(scale_factor)
    corrected_scaled = np.where(np.isfinite(corrected_scaled), corrected_scaled, no_data_value)

    return corrected_scaled.astype(np.float32, copy=False)


def apply_glint_correct(*args, **kwargs):
    """Placeholder for sunglint correction.

    In hytools.glint.apply_glint_correct (GPLv3), sunglint correction is used
    for aquatic environments. We are not supporting that right now.
    """

    raise NotImplementedError("Glint correction is not implemented in cross-sensor-cal.")


def fit_and_save_brdf_model(
    cube: "NeonCube",
    out_dir: Path,
    ndvi_config: NDVIBinningConfig | None = None,
    rho_min: float = 0.0,
    rho_max: float | None = 2.0,
) -> Path:
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

    prefix = scene_prefix_from_dir(out_dir)
    preferred = out_dir / f"{prefix}_brdf_model.json"
    normalized = normalize_brdf_model_path(out_dir)
    coeff_path = normalized if normalized is not None else preferred
    if coeff_path.exists():
        return coeff_path

    ndvi_config = ndvi_config or NDVIBinningConfig()

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

    reflectance_unitless = (
        np.asarray(cube.data, dtype=np.float32) * np.float32(getattr(cube, "scale_factor", 1.0) or 1.0)
    )

    ndvi = compute_ndvi(cube, reflectance_unitless)
    edges, ndvi_bins = compute_ndvi_bins(ndvi, ndvi_config)

    flat_valid = valid.reshape(-1)
    design_stack = np.stack(
        [
            np.ones_like(volume_kernel, dtype=np.float32),
            volume_kernel,
            geom_kernel,
        ],
        axis=-1,
    )
    design_flat = design_stack.reshape(-1, 3)
    reflectance_flat = reflectance_unitless.reshape(-1, cube.bands)
    ndvi_flat = ndvi_bins.reshape(-1)

    n_bins = ndvi_config.n_bins
    iso = np.ones((n_bins, cube.bands), dtype=np.float32)
    vol = np.zeros_like(iso)
    geo = np.zeros_like(iso)

    for band_idx in range(cube.bands):
        y_all = reflectance_flat[:, band_idx].astype(np.float64, copy=False)
        for bin_id in range(n_bins):
            bin_mask = flat_valid & (ndvi_flat == bin_id)
            if np.count_nonzero(bin_mask) < 3:
                continue
            y = y_all[bin_mask]
            X = design_flat[bin_mask].astype(np.float64, copy=False)
            finite_mask = np.isfinite(y)
            if rho_max is not None:
                finite_mask &= y <= rho_max
            finite_mask &= y >= rho_min
            if np.count_nonzero(finite_mask) < 3:
                continue
            try:
                solution, *_ = np.linalg.lstsq(X[finite_mask], y[finite_mask], rcond=None)
            except np.linalg.LinAlgError:
                continue
            iso[bin_id, band_idx] = np.float32(solution[0])
            vol[bin_id, band_idx] = np.float32(solution[1])
            geo[bin_id, band_idx] = np.float32(solution[2])

    coeff_payload = {
        "iso": iso.astype(float).tolist(),
        "vol": vol.astype(float).tolist(),
        "geo": geo.astype(float).tolist(),
        "volume_kernel": "RossThick",
        "geom_kernel": "LiSparseReciprocal",
        "ndvi_edges": edges.astype(float).tolist(),
    }

    with coeff_path.open("w", encoding="utf-8") as coeff_file:
        json.dump(coeff_payload, coeff_file, indent=2)

    return coeff_path
