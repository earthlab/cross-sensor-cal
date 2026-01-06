# DEPRECATED: Replaced by spectralbridge.resample / standard_resample.
# The active pipeline ONLY resamples from the BRDF+topo corrected ENVI cube,
# and ONLY writes ENVI (.img/.hdr).
# This file has been staged for removal.
"""Spectral convolution helpers used by the streamlined pipeline."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .file_types import (
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceBRDFCorrectedENVIHDRFile,
    NEONReflectanceResampledENVIFile,
    NEONReflectanceResampledHDRFile,
)
from .pipelines.pipeline import convolve_resample_product, _parse_envi_header
from .utils import get_package_data_path
from .utils_checks import is_valid_envi_pair


logger = logging.getLogger(__name__)

DEFAULT_SENSOR_LIBRARY = "landsat_band_parameters.json"


def _load_sensor_library() -> Mapping[str, dict[str, Iterable[float]]]:
    try:
        library_path = get_package_data_path(DEFAULT_SENSOR_LIBRARY)
    except FileNotFoundError as exc:  # pragma: no cover - package data missing
        raise RuntimeError("Sensor response library not bundled with package") from exc

    with library_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise RuntimeError("Sensor response library is not a mapping")

    return payload


def _build_sensor_srfs(
    wavelengths: np.ndarray,
    centres: Iterable[float],
    fwhm: Iterable[float],
) -> dict[str, np.ndarray]:
    wl = np.asarray(wavelengths, dtype=np.float32)
    if wl.ndim != 1 or wl.size == 0:
        return {}

    centres_arr = np.asarray(list(centres), dtype=np.float32)
    fwhm_arr = np.asarray(list(fwhm), dtype=np.float32) if fwhm else np.array([], dtype=np.float32)

    srfs: dict[str, np.ndarray] = {}
    for idx, centre in enumerate(centres_arr):
        key = f"band_{idx + 1:02d}"
        width = float(fwhm_arr[idx]) if idx < fwhm_arr.size else float(fwhm_arr[-1]) if fwhm_arr.size else 0.0
        if math.isclose(width, 0.0):
            weights = np.zeros_like(wl)
            nearest = int(np.abs(wl - centre).argmin())
            weights[nearest] = 1.0
        else:
            sigma = width / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            sigma = max(sigma, np.finfo(np.float32).eps)
            weights = np.exp(-0.5 * ((wl - centre) / sigma) ** 2)
        weight_sum = float(np.sum(weights, dtype=np.float64))
        if weight_sum > 0:
            weights = weights / weight_sum
        srfs[key] = weights.astype(np.float32, copy=False)

    return srfs


def convolve_all_sensors(
    *,
    corrected_img_path: Path,
    corrected_hdr_path: Path,
    out_dir: Path,
    sensor_list: list[str] | None = None,
) -> None:
    """Convolve a corrected ENVI cube to the requested sensor bandpasses."""

    corrected_img_path = Path(corrected_img_path)
    corrected_hdr_path = Path(corrected_hdr_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_valid_envi_pair(corrected_img_path, corrected_hdr_path):
        raise FileNotFoundError(
            "Corrected ENVI missing or invalid: "
            f"{corrected_img_path}, {corrected_hdr_path}"
        )

    try:
        hdr = NEONReflectanceBRDFCorrectedENVIHDRFile.from_filename(corrected_hdr_path)
    except ValueError as exc:  # pragma: no cover - malformed name
        raise RuntimeError(f"Unable to parse corrected HDR filename: {corrected_hdr_path}") from exc

    try:
        img = NEONReflectanceBRDFCorrectedENVIFile.from_filename(corrected_img_path)
    except ValueError:
        # Fallback to metadata from the HDR file for naming.
        img = None

    library = _load_sensor_library()

    header = _parse_envi_header(corrected_hdr_path)
    wavelengths = header.get("wavelength")
    if not wavelengths:
        raise RuntimeError("Corrected HDR missing wavelength metadata required for convolution")

    wavelengths_arr = np.asarray(wavelengths, dtype=np.float32)

    selected_sensors = sensor_list or list(library.keys())

    for sensor_name in selected_sensors:
        params = library.get(sensor_name)
        if not params:
            logger.warning("⚠️  Sensor '%s' not present in library; skipping.", sensor_name)
            continue

        srfs = _build_sensor_srfs(
            wavelengths_arr,
            params.get("wavelengths", []),
            params.get("fwhms", []),
        )
        if not srfs:
            logger.warning(
                "⚠️  No spectral response functions computed for %s; skipping.",
                sensor_name,
            )
            continue

        dir_prefix = "Convolution_Reflectance_Resample"
        sensor_dir = out_dir / f"{dir_prefix}_{sensor_name.replace(' ', '_')}"
        sensor_dir.mkdir(parents=True, exist_ok=True)

        base_product = img.product if img is not None else hdr.product or "30006.001"
        suffix = img.suffix if img is not None else hdr.suffix or "envi"
        resampled_img = NEONReflectanceResampledENVIFile.from_components(
            domain=hdr.domain,
            site=hdr.site,
            date=hdr.date,
            sensor=sensor_name,
            suffix=suffix,
            folder=sensor_dir,
            time=hdr.time,
            tile=hdr.tile,
            directional=hdr.directional,
            product=base_product,
        )
        resampled_hdr = NEONReflectanceResampledHDRFile.from_components(
            domain=hdr.domain,
            site=hdr.site,
            date=hdr.date,
            sensor=sensor_name,
            suffix=suffix,
            folder=sensor_dir,
            time=hdr.time,
            tile=hdr.tile,
            directional=hdr.directional,
            product=base_product,
        )

        out_stem = resampled_img.path.with_suffix("")

        if is_valid_envi_pair(resampled_img.path, resampled_hdr.path):
            logger.info(
                "✅ Convolution for %s already complete, skipping",
                sensor_name,
            )
            continue

        try:
            convolve_resample_product(
                corrected_hdr_path=corrected_hdr_path,
                sensor_srf=srfs,
                out_stem_resampled=out_stem,
            )
        except Exception:  # pragma: no cover - error logging path
            logger.error(
                "⚠️  Resample failed for %s (%s)",
                corrected_img_path.name,
                sensor_name,
                exc_info=True,
            )
            raise

        if not is_valid_envi_pair(resampled_img.path, resampled_hdr.path):
            raise RuntimeError(
                "Resampled outputs missing or invalid for "
                f"{sensor_name}: {resampled_img.path}"
            )


__all__ = ["convolve_all_sensors"]
