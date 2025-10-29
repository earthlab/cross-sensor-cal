# DEPRECATED: Replaced by cross_sensor_cal.resample / standard_resample.
# The active pipeline ONLY resamples from the BRDF+topo corrected ENVI cube,
# and ONLY writes ENVI (.img/.hdr).
# This file has been staged for removal.
"""Spectral resampling utilities for NEON hyperspectral products.

References:
    Chander, G., Markham, B., & Helder, D. (2009). Remote Sensing of Environment.
    Teillet, P. et al. (2007). Remote Sensing of Environment.
    Claverie, M. et al. (2018). Harmonized Landsat and Sentinel-2 (HLS) framework.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.stats import norm

from ._optional import require_ray, require_spectral
from .file_types import (
    NEONReflectanceResampledENVIFile,
    NEONReflectanceResampledHDRFile,
    NEONReflectanceBRDFCorrectedENVIHDRFile
)
from .utils import get_package_data_path

_RAY_CONVOLUTION_WORKER = None



@lru_cache(maxsize=1)
def _require_spectral():
    """Return spectral helpers, raising a clear error when unavailable."""

    spectral = require_spectral()
    return spectral.open_image, spectral.io.envi


def _files_exist_and_nonempty(*paths: Path) -> bool:
    """Return ``True`` when every provided path exists and has a payload."""

    for path in paths:
        if not path.exists():
            return False
        try:
            if path.stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def _parse_wavelengths(raw_wavelengths):
    """Normalise raw wavelength values from an ENVI header into floats."""

    if raw_wavelengths is None:
        return []

    if isinstance(raw_wavelengths, (int, float)):
        return [float(raw_wavelengths)]

    if isinstance(raw_wavelengths, str):
        raw_wavelengths = [raw_wavelengths]

    cleaned = []

    for value in raw_wavelengths:
        if isinstance(value, (int, float)):
            cleaned.append(float(value))
            continue

        if not isinstance(value, str):
            continue

        # Remove ENVI braces and split on common delimiters
        token = value.strip().strip('{}')
        if not token:
            continue

        token = token.replace(',', ' ')
        for part in token.split():
            try:
                cleaned.append(float(part))
            except ValueError:
                continue

    return cleaned


def _ensure_nm_and_sort(wavelengths: Iterable[float], cube: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Convert wavelength units to nm and enforce ascending order."""

    wl = np.asarray(wavelengths, dtype=float).copy()
    if wl.size == 0:
        return wl, cube, np.array([], dtype=int)

    if np.nanmax(wl) < 20.0:
        wl *= 1000.0

    order = np.argsort(wl)
    wl_sorted = wl[order]

    if cube is None:
        return wl_sorted, None, order

    return wl_sorted, cube[..., order], order


def _nearest_or_linear_sample(
    cube: np.ndarray,
    wl_nm: np.ndarray,
    target_centers_nm: Iterable[float],
    mode: str = "nearest",
) -> np.ndarray:
    """Sample nearest or linearly interpolated spectra for each target wavelength."""

    if mode not in {"nearest", "linear"}:
        raise ValueError(f"Unsupported sampling mode: {mode}")

    rows, cols, in_bands = cube.shape
    centers = np.asarray(target_centers_nm, dtype=float)
    if np.nanmax(centers) < 20.0:
        centers = centers * 1000.0

    out = np.empty((rows, cols, centers.size), dtype=cube.dtype)

    if mode == "nearest":
        # [B_out]
        idx = np.abs(wl_nm[None, :] - centers[:, None]).argmin(axis=1)
        out = cube[:, :, idx]
        return out.astype(np.float32, copy=False)

    # Linear interpolation along the spectral axis
    for i, center in enumerate(centers):
        j = int(np.searchsorted(wl_nm, center, side="left"))
        if j <= 0:
            out[:, :, i] = cube[:, :, 0]
        elif j >= in_bands:
            out[:, :, i] = cube[:, :, -1]
        else:
            left = j - 1
            wl_left = wl_nm[left]
            wl_right = wl_nm[j]
            weight = (center - wl_left) / (wl_right - wl_left)
            out[:, :, i] = (1 - weight) * cube[:, :, left] + weight * cube[:, :, j]

    return out.astype(np.float32, copy=False)


def _gaussian_rsr_delta_norm(wl_nm: np.ndarray, center_nm: float, fwhm_nm: float) -> np.ndarray:
    """Create a Δλ-normalised Gaussian spectral response."""

    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    srf = norm.pdf(wl_nm, loc=center_nm, scale=sigma)
    delta_lambda = np.gradient(wl_nm)
    denom = np.sum(srf * delta_lambda)
    if denom > 0:
        return srf / denom
    return srf


def _build_W_from_gaussians(
    wl_nm: np.ndarray,
    centers_nm: Iterable[float],
    fwhms_nm: Iterable[float],
) -> np.ndarray:
    """Construct weighting matrix from Gaussian SRFs (Δλ-normalised)."""

    centers = np.asarray(centers_nm, dtype=float)
    fwhms = np.asarray(fwhms_nm, dtype=float)

    if centers.size == 0:
        return np.zeros((0, wl_nm.size), dtype=float)

    if np.nanmax(centers) < 20.0:
        centers = centers * 1000.0
    if np.nanmax(fwhms) < 20.0:
        fwhms = fwhms * 1000.0

    rows = [_gaussian_rsr_delta_norm(wl_nm, c, f) for c, f in zip(centers, fwhms)]
    return np.vstack(rows) if rows else np.zeros((0, wl_nm.size), dtype=float)


def _build_W_from_tabulated_srfs(wl_nm: np.ndarray, srfs_dict: Dict[str, Dict[str, Iterable[float]]]) -> np.ndarray:
    """Construct weighting matrix using tabulated SRFs and Δλ normalisation."""

    rows = []
    for spec in srfs_dict.values():
        lam = np.asarray(spec["wavelengths"], dtype=float)
        rsp = np.asarray(spec["response"], dtype=float)

        if lam.size == 0:
            rows.append(np.zeros_like(wl_nm, dtype=float))
            continue

        if np.nanmax(lam) < 20.0:
            lam = lam * 1000.0

        srf = np.interp(wl_nm, lam, rsp, left=0.0, right=0.0)
        delta_lambda = np.gradient(wl_nm)
        denom = np.sum(srf * delta_lambda)
        rows.append(srf / denom if denom > 0 else srf)

    return np.vstack(rows) if rows else np.zeros((0, wl_nm.size), dtype=float)


def _apply_convolution_with_renorm(
    cube: np.ndarray,
    weights: np.ndarray,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """Apply spectral convolution with per-pixel renormalisation."""

    rows, cols, bands = cube.shape

    flat = cube.reshape(-1, bands).astype(np.float32, copy=False)
    weights = np.asarray(weights, dtype=np.float32)

    if nodata is not None and not np.isnan(nodata):
        nodata_mask = np.isclose(flat, nodata)
    else:
        nodata_mask = np.zeros_like(flat, dtype=bool)

    nan_mask = np.isnan(flat)
    invalid_mask = nodata_mask | nan_mask

    if invalid_mask.any():
        data = np.where(invalid_mask, 0.0, flat)
        valid = (~invalid_mask).astype(np.float32, copy=False)
    else:
        data = flat
        valid = np.ones_like(flat, dtype=np.float32)

    numer = data @ weights.T
    denom = valid @ weights.T

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)

    return result.reshape(rows, cols, -1).astype(np.float32, copy=False)


def _ray_convolution_chunk_fn(
    chunk: np.ndarray,
    weights: np.ndarray,
    nodata: Optional[float],
) -> np.ndarray:
    """Ray worker wrapper that reuses the serial convolution implementation."""

    return _apply_convolution_with_renorm(chunk, weights, nodata)


def _apply_convolution_with_renorm_ray(
    cube: np.ndarray,
    weights: np.ndarray,
    nodata: Optional[float] = None,
    *,
    chunk_lines: Optional[int] = None,
    num_cpus: Optional[int] = None,
) -> np.ndarray:
    """Parallel spectral convolution using Ray workers."""

    ray = require_ray()

    global _RAY_CONVOLUTION_WORKER

    if _RAY_CONVOLUTION_WORKER is None:
        _RAY_CONVOLUTION_WORKER = ray.remote(_ray_convolution_chunk_fn)

    rows, cols, _ = cube.shape

    if chunk_lines is None or chunk_lines < 1:
        available_cpus = num_cpus
        if available_cpus is None:
            available_cpus = int(ray.available_resources().get("CPU", 1)) or 1
        target_chunks = max(available_cpus * 4, 1)
        chunk_lines = max(1, math.ceil(rows / target_chunks))

    chunk_lines = min(chunk_lines, rows)

    weights_ref = ray.put(weights)
    tasks = []
    for start in range(0, rows, chunk_lines):
        stop = min(start + chunk_lines, rows)
        chunk = cube[start:stop, :, :]
        tasks.append(_RAY_CONVOLUTION_WORKER.remote(chunk, weights_ref, nodata))

    chunks = ray.get(tasks)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _legacy_gaussian_weights(wl_nm: np.ndarray, centers_nm: np.ndarray, fwhms_nm: np.ndarray) -> np.ndarray:
    """Reproduce legacy Gaussian weights normalised by sum of weights."""

    weights = []
    for center, fwhm in zip(centers_nm, fwhms_nm):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        srf = norm.pdf(wl_nm, loc=center, scale=sigma)
        total = np.sum(srf)
        weights.append(srf / total if total > 0 else srf)
    return np.vstack(weights) if weights else np.zeros((0, wl_nm.size), dtype=float)


def resample(
    input_dir: Path,
    method: str = "convolution",
    straight_mode: str = "nearest",
    num_cpus: Optional[int] = None,
    use_ray: Optional[bool] = None,
    ray_chunk_lines: Optional[int] = None,
):
    """Perform spectral resampling using configurable response models.

    Args:
        input_dir: Directory containing BRDF+TOPO corrected hyperspectral imagery.
        method: Resampling method. "convolution" performs Δλ-normalised SRF integration
            with per-pixel renormalisation (preferred). "gaussian" preserves legacy
            Gaussian-weighted sums. "straight" samples the nearest (or linearly
            interpolated) band centres without SRFs.
        straight_mode: Sampling mode for the "straight" method ("nearest" or "linear").
        num_cpus: Optional explicit CPU count when running the convolution step with Ray.
        use_ray: Force enabling/disabling Ray for the convolution step. ``None``
            (default) enables Ray automatically when ``num_cpus`` requests more than
            one core.
        ray_chunk_lines: Optional override for the number of image lines processed per
            Ray task. ``None`` auto-derives a chunk size from the CPU count.
    """

    method = method.lower()
    if method not in {"straight", "gaussian", "convolution"}:
        raise ValueError(f"Unsupported resampling method: {method}")

    straight_mode = straight_mode.lower()
    if straight_mode not in {"nearest", "linear"}:
        raise ValueError(f"Unsupported straight sampling mode: {straight_mode}")

    use_ray = (
        False
        if method != "convolution"
        else (bool(use_ray) if use_ray is not None else (num_cpus not in (None, 1)))
    )

    open_image, envi = _require_spectral()

    ray_module = None
    ray_cpus = None
    ray_started_here = False

    if use_ray:
        from .ray_utils import init_ray
        ray_module = require_ray()
        ray_cpus = init_ray(num_cpus)
        ray_started_here = True
        print(f"🧠 Ray initialised with {ray_cpus} CPUs for convolution resampling.")

    print(f"🚀 Starting {method} resample for {input_dir}")
    brdf_corrected_hdr_files = NEONReflectanceBRDFCorrectedENVIHDRFile.find_in_directory(input_dir)

    try:
        for hdr_file in brdf_corrected_hdr_files:
            if "mask" in (hdr_file.suffix or "").lower():
                print(f"ℹ️ Skipping mask file without spectral bands: {hdr_file.file_path}")
                continue

            try:
                print(f"📂 Opening: {hdr_file.file_path}")
                img = open_image(str(hdr_file.file_path))
                hyperspectral_data = np.asarray(img.load()).astype(np.float32, copy=False)
            except Exception as e:
                print(f"❌ ERROR: Could not load {hdr_file.file_path}: {e}")
                continue

            header = envi.read_envi_header(str(hdr_file.file_path))
            wavelengths = _parse_wavelengths(header.get('wavelength'))

            if not wavelengths:
                try:
                    bands_file = get_package_data_path('hyperspectral_bands.json')
                except FileNotFoundError:
                    wavelengths = []
                else:
                    with bands_file.open('r', encoding='utf-8') as f:
                        wavelengths = _parse_wavelengths(json.load(f).get('bands'))

            if not wavelengths:
                print("❌ ERROR: No wavelengths found.")
                continue

            wl_nm, hyperspectral_data, _ = _ensure_nm_and_sort(wavelengths, cube=hyperspectral_data)

            rows, cols, bands = hyperspectral_data.shape
            if len(wl_nm) != bands:
                print(f"❌ ERROR: Band mismatch ({len(wl_nm)} wavelengths vs {bands} bands).")
                continue

            nodata_value = header.get('data ignore value')
            nodata_float: Optional[float] = None
            if nodata_value is not None:
                if isinstance(nodata_value, (list, tuple)) and nodata_value:
                    try:
                        nodata_float = float(nodata_value[0])
                    except (TypeError, ValueError):
                        nodata_float = None
                else:
                    try:
                        nodata_float = float(nodata_value)
                    except (TypeError, ValueError):
                        nodata_float = None

            if nodata_float is not None:
                hyperspectral_data = np.where(
                    np.isclose(hyperspectral_data, nodata_float),
                    np.nan,
                    hyperspectral_data,
                )

            try:
                sensor_params_file = get_package_data_path('landsat_band_parameters.json')
            except FileNotFoundError:
                print('❌ ERROR: Sensor response parameters are not bundled with the package.')
                continue
            with sensor_params_file.open('r', encoding='utf-8') as f:
                all_sensor_params = json.load(f)

            for sensor_name, sensor_params in all_sensor_params.items():
                dir_prefix = f"{method.capitalize()}_Reflectance_Resample"
                resampled_dir = hdr_file.directory / f"{dir_prefix}_{sensor_name.replace(' ', '_')}"
                os.makedirs(resampled_dir, exist_ok=True)

                # Build output file paths using corrected naming conventions
                resampled_hdr_file = NEONReflectanceResampledHDRFile.from_components(
                    domain=hdr_file.domain,
                    site=hdr_file.site,
                    date=hdr_file.date,
                    sensor=sensor_name,
                    suffix=hdr_file.suffix,
                    folder=resampled_dir,
                    time=hdr_file.time,
                    tile=hdr_file.tile,
                    directional=hdr_file.directional
                )

                resampled_img_file = NEONReflectanceResampledENVIFile.from_components(
                    domain=hdr_file.domain,
                    site=hdr_file.site,
                    date=hdr_file.date,
                    sensor=sensor_name,
                    suffix=hdr_file.suffix,
                    folder=resampled_dir,
                    time=hdr_file.time,
                    tile=hdr_file.tile,
                    directional=hdr_file.directional
                )

                if _files_exist_and_nonempty(
                    resampled_hdr_file.path, resampled_img_file.path
                ):
                    print(
                        f"⚠️ Skipping resampling for {sensor_name}: files already exist."
                    )
                    continue

                if (
                    resampled_hdr_file.path.exists()
                    or resampled_img_file.path.exists()
                ):
                    print(
                        f"♻️ Incomplete resample detected for {sensor_name}; "
                        "recomputing outputs."
                    )

                centers_nm = np.asarray(sensor_params["wavelengths"], dtype=float)
                fwhms_nm = np.asarray(sensor_params.get("fwhms", []), dtype=float)

                if centers_nm.size and np.nanmax(centers_nm) < 20.0:
                    centers_nm = centers_nm * 1000.0
                if fwhms_nm.size and np.nanmax(fwhms_nm) < 20.0:
                    fwhms_nm = fwhms_nm * 1000.0

                if method == "straight":
                    resampled = _nearest_or_linear_sample(
                        hyperspectral_data,
                        wl_nm,
                        centers_nm,
                        mode=straight_mode,
                    )

                elif method == "gaussian":
                    legacy_weights = _legacy_gaussian_weights(wl_nm, centers_nm, fwhms_nm)
                    flat = np.nan_to_num(hyperspectral_data, nan=0.0).reshape(-1, bands)
                    resampled = flat @ legacy_weights.T
                    resampled = resampled.reshape(rows, cols, -1).astype(np.float32)

                else:  # convolution
                    srfs_filename = sensor_name.replace(' ', '_').lower() + '_srfs.json'
                    try:
                        srfs_path = get_package_data_path(srfs_filename)
                    except FileNotFoundError:
                        srfs_path = None
                    if srfs_path is not None:
                        with srfs_path.open('r', encoding='utf-8') as srfs_file:
                            srfs_dict: Dict[str, Dict[str, Iterable[float]]] = json.load(srfs_file)
                        weights = _build_W_from_tabulated_srfs(wl_nm, srfs_dict)
                    else:
                        weights = _build_W_from_gaussians(wl_nm, centers_nm, fwhms_nm)

                    nodata_arg = None if nodata_float is None else nodata_float
                    if use_ray:
                        resampled = _apply_convolution_with_renorm_ray(
                            hyperspectral_data,
                            weights,
                            nodata=nodata_arg,
                            chunk_lines=ray_chunk_lines,
                            num_cpus=ray_cpus,
                        )
                    else:
                        resampled = _apply_convolution_with_renorm(
                            hyperspectral_data,
                            weights,
                            nodata=nodata_arg,
                        )

                if nodata_float is not None:
                    resampled_to_save = np.where(np.isnan(resampled), nodata_float, resampled)
                else:
                    resampled_to_save = resampled

                resampled_to_save = resampled_to_save.astype(np.float32, copy=False)

                n_out_bands = resampled.shape[2]
                new_metadata = {
                    'description': f'Resampled hyperspectral image using {sensor_name} ({method} method)',
                    'samples': cols,
                    'lines': rows,
                    'bands': n_out_bands,
                    'data type': 4,
                    'interleave': 'bsq',
                    'byte order': 0,
                    'sensor type': sensor_name,
                    'wavelength units': 'nanometers',
                    'wavelength': [str(w) for w in centers_nm],
                    'map info': header.get('map info'),
                    'coordinate system string': header.get('coordinate system string'),
                    'data ignore value': nodata_value,
                }

                envi.save_image(
                    str(resampled_hdr_file.file_path),
                    resampled_to_save,
                    metadata=new_metadata,
                    force=True,
                )
                print(f"✅ Resampled file saved: {resampled_hdr_file.file_path}")
    finally:
        if ray_started_here and ray_module is not None:
            ray_module.shutdown()
