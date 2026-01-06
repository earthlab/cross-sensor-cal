"""Canonical path helpers for pipeline outputs."""

from __future__ import annotations

import re
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

    # First, try exact match with flight_stem prefix
    for root in search_roots:
        if not root.exists():
            continue
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

    # If exact match fails, try flexible matching for legacy datetime format files
    # Pattern: NEON_D13_NIWO_DP1_20200720_163210_reflectance
    # We want to match files like: NEON_D13_NIWO_DP1.30006.001_20200720T163210_20200720_163210_reflectance_envi.img
    flight_parts = flight_stem.split("_")
    if len(flight_parts) >= 6 and flight_parts[0] == "NEON":
        domain = flight_parts[1]  # e.g., "D13"
        site = flight_parts[2]    # e.g., "NIWO"
        # Find date pattern (8 digits) and time pattern (6 digits)
        date_match = re.search(r"(\d{8})", flight_stem)
        time_match = re.search(r"(\d{6})", flight_stem)
        
        if date_match and time_match:
            date = date_match.group(1)
            time = time_match.group(1)
            
            # Search for all *_envi.img files to catch all variations
            for root in search_roots:
                if not root.exists():
                    continue
                # First try *_reflectance_envi.img, then fall back to *_envi.img
                for pattern in ["*_reflectance_envi.img", "*_envi.img"]:
                    try:
                        all_candidates = sorted(root.glob(pattern))
                    except (OSError, PermissionError):
                        # Skip if we can't read the directory
                        continue
                    
                    for img_path in all_candidates:
                        name_lower = img_path.name.lower()
                        if "_brdfandtopo_corrected_envi" in name_lower:
                            continue
                        
                        # Check if this file matches our flight_stem by looking for key components
                        # Must contain: domain, site, date, and time (with either T or _ separator)
                        filename = img_path.name
                        expected_prefix = f"NEON_{domain}_{site}_"
                        has_domain_site = filename.startswith(expected_prefix)
                        has_date = date in filename
                        has_time = time in filename
                        
                        # If all key components match, this is likely our file
                        if has_domain_site and has_date and has_time:
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
        # Ensure work_dir exists before searching (it might not exist on first run)
        # But we can still search even if it doesn't exist - glob will just return empty
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
