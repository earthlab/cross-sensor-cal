"""Canonical path helpers for pipeline outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..paths import FlightlinePaths


def get_flight_paths(base_folder: Path, flight_stem: str) -> Dict[str, Path]:
    """Return canonical high-level paths for a flight line."""

    flight_paths = FlightlinePaths(base_folder=Path(base_folder), flight_id=flight_stem)

    return {
        "base": flight_paths.base_folder,
        "work_dir": flight_paths.flight_dir,
        "h5_path": flight_paths.h5,
    }


def _pick_uncorrected_envi_pair(
    primary_dir: Path,
    flight_stem: str,
    *,
    legacy_dir: Optional[Path] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Try to identify the *uncorrected* ENVI pair for ``flight_stem``."""

    search_roots: tuple[Path, ...]
    if legacy_dir is not None and Path(legacy_dir) != Path(primary_dir):
        search_roots = (Path(primary_dir), Path(legacy_dir))
    else:
        search_roots = (Path(primary_dir),)

    for root in search_roots:
        candidates = sorted(root.glob(f"{flight_stem}*.img"))

        for img_path in candidates:
            name_lower = img_path.name.lower()
            if "_brdfandtopo_corrected_envi" in name_lower:
                continue

            hdr_path = img_path.with_suffix(".hdr")
            if (
                img_path.exists()
                and img_path.is_file()
                and img_path.stat().st_size > 0
                and hdr_path.exists()
                and hdr_path.is_file()
                and hdr_path.stat().st_size > 0
            ):
                return img_path, hdr_path

    return None, None


def get_flightline_products(
    base_folder: Path,
    product_code: str,
    flight_stem: str,
) -> Dict[str, Any]:
    """Return canonical paths for artefacts tied to ``flight_stem``."""

    flightline = FlightlinePaths(base_folder=Path(base_folder), flight_id=flight_stem)
    work_dir = flightline.flight_dir
    h5_path = flightline.h5

    _ = product_code  # retained for API compatibility

    raw_img_guess = flightline.envi_img
    raw_hdr_guess = flightline.envi_hdr

    def _good(path: Path) -> bool:
        return path.exists() and path.is_file() and path.stat().st_size > 0

    if _good(raw_img_guess) and _good(raw_hdr_guess):
        raw_envi_img = raw_img_guess
        raw_envi_hdr = raw_hdr_guess
    else:
        discovered_img, discovered_hdr = _pick_uncorrected_envi_pair(
            primary_dir=work_dir,
            flight_stem=flight_stem,
            legacy_dir=flightline.base_folder,
        )
        raw_envi_img = discovered_img if discovered_img is not None else raw_img_guess
        raw_envi_hdr = discovered_hdr if discovered_hdr is not None else raw_hdr_guess

    corrected_img = flightline.corrected_img
    corrected_hdr = flightline.corrected_hdr
    correction_json = flightline.corrected_json

    def _sensor_pair(sensor_name: str) -> Dict[str, Path]:
        product_paths = flightline.sensor_product(sensor_name)
        return {
            "img": product_paths.img,
            "hdr": product_paths.hdr,
            "parquet": product_paths.parquet,
            "qa_png": product_paths.qa_png,
            "qa_pdf": product_paths.qa_pdf,
            "qa_json": product_paths.qa_json,
        }

    sensor_products: Dict[str, Dict[str, Path]] = {
        "landsat_tm": _sensor_pair("landsat_tm"),
        "landsat_etm+": _sensor_pair("landsat_etm+"),
        "landsat_oli": _sensor_pair("landsat_oli"),
        "landsat_oli2": _sensor_pair("landsat_oli2"),
        "micasense": _sensor_pair("micasense"),
        "micasense_to_match_tm_etm+": _sensor_pair("micasense_to_match_tm_etm+"),
        "micasense_to_match_oli_oli2": _sensor_pair("micasense_to_match_oli_oli2"),
    }

    return {
        "base": flightline.base_folder,
        "work_dir": work_dir,
        "h5_path": h5_path,
        "h5": h5_path,
        "raw_envi_img": raw_envi_img,
        "raw_envi_hdr": raw_envi_hdr,
        "raw_envi_parquet": flightline.envi_parquet,
        "correction_json": correction_json,
        "corrected_img": corrected_img,
        "corrected_hdr": corrected_hdr,
        "corrected_parquet": flightline.corrected_parquet,
        "sensor_products": sensor_products,
    }


__all__ = ["get_flight_paths", "get_flightline_products"]
