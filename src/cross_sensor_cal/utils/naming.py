"""Canonical path helpers for pipeline outputs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

from ..file_types import (
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceENVIFile,
    NEONReflectanceFile,
    NEONReflectanceResampledENVIFile,
)
from ..utils.paths import get_package_data_path


def _normalize_product_code(product_code: str | None, fallback: str | None) -> str:
    """Return a fully-qualified DP1 product code."""

    candidate = product_code or fallback or "DP1.30006.001"
    candidate = candidate.strip()
    if not candidate:
        return "DP1.30006.001"

    upper = candidate.upper()
    if upper.startswith("DP1."):
        return f"DP1.{candidate[4:]}" if len(candidate) > 4 else "DP1.30006.001"
    if upper.startswith("DP1"):
        trimmed = candidate[3:]
        trimmed = trimmed.lstrip(".")
        return f"DP1.{trimmed}" if trimmed else "DP1.30006.001"
    if upper.startswith("DP"):
        return candidate
    return f"DP1.{candidate.strip('.')}" if candidate.strip(".") else "DP1.30006.001"


def _product_numeric(product_code: str) -> str:
    trimmed = product_code.strip()
    trimmed = trimmed.replace("DP1.", "") if trimmed.upper().startswith("DP1.") else trimmed
    trimmed = trimmed.replace("DP1", "") if trimmed.upper().startswith("DP1") else trimmed
    trimmed = trimmed.lstrip(".")
    return trimmed or "30006.001"


@lru_cache(maxsize=1)
def _sensor_library_keys() -> list[str]:
    try:
        library_path = get_package_data_path("landsat_band_parameters.json")
    except FileNotFoundError:
        return []

    try:
        with library_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict):
        return [str(key) for key in payload.keys()]
    return []


def get_flightline_products(
    base_folder: Path,
    product_code: str,
    flight_stem: str,
) -> Dict[str, object]:
    """Return canonical paths for all artefacts tied to *flight_stem*.

    The returned mapping contains ``Path`` objects keyed by artefact type. It
    relies solely on the repository's file type helpers so every stage operates
    on consistent filenames.
    """

    base_path = Path(base_folder)
    base_path.mkdir(parents=True, exist_ok=True)

    stem_only = Path(flight_stem).stem
    h5_path = base_path / f"{stem_only}.h5"

    refl = NEONReflectanceFile.from_filename(h5_path)
    if not refl.tile:
        raise RuntimeError(
            "Unable to determine NEON tile identifier from flightline stem; "
            "expected standard NEON naming convention."
        )

    product_full = _normalize_product_code(product_code, refl.product)
    product_numeric = _product_numeric(product_full)

    raw_envi = NEONReflectanceENVIFile.from_components(
        domain=refl.domain or "D00",
        site=refl.site or "SITE",
        product=product_full,
        tile=refl.tile,
        date=refl.date or "00000000",
        time=refl.time,
        directional=refl.directional,
        folder=base_path,
    )
    raw_img_path = raw_envi.path
    raw_hdr_path = raw_img_path.with_suffix(".hdr")

    corrected = NEONReflectanceBRDFCorrectedENVIFile.from_components(
        domain=refl.domain or "D00",
        site=refl.site or "SITE",
        date=refl.date or "00000000",
        suffix="envi",
        folder=base_path,
        time=refl.time,
        tile=refl.tile,
        directional=refl.directional,
        product=product_numeric,
    )
    corrected_img_path = corrected.path
    corrected_hdr_path = corrected_img_path.with_suffix(".hdr")
    correction_json_path = corrected_img_path.with_suffix(".json")

    sensor_products: Dict[str, Path] = {}
    suffix = corrected.suffix or "envi"
    for sensor_name in _sensor_library_keys():
        sensor_dir = base_path / f"Convolution_Reflectance_Resample_{sensor_name.replace(' ', '_')}"
        resampled = NEONReflectanceResampledENVIFile.from_components(
            domain=refl.domain or "D00",
            site=refl.site or "SITE",
            date=refl.date or "00000000",
            sensor=sensor_name,
            suffix=suffix,
            folder=sensor_dir,
            time=refl.time,
            tile=refl.tile,
            directional=refl.directional,
            product=product_numeric,
        )
        sensor_products[sensor_name] = resampled.path

    return {
        "h5": h5_path,
        "raw_envi_img": raw_img_path,
        "raw_envi_hdr": raw_hdr_path,
        "correction_json": correction_json_path,
        "corrected_img": corrected_img_path,
        "corrected_hdr": corrected_hdr_path,
        "sensor_products": sensor_products,
    }


__all__ = ["get_flightline_products"]

