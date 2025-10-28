"""Canonical path helpers for pipeline outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _pick_uncorrected_envi_pair(
    base_folder: Path,
    flight_stem: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Try to identify the *uncorrected* ENVI pair for ``flight_stem``."""

    base_folder = Path(base_folder)
    candidates = sorted(base_folder.glob(f"{flight_stem}*.img"))

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

    base_folder = Path(base_folder)
    _ = product_code  # retained for API compatibility

    h5_path = base_folder / f"{flight_stem}.h5"

    raw_img_guess = base_folder / f"{flight_stem}_envi.img"
    raw_hdr_guess = base_folder / f"{flight_stem}_envi.hdr"

    def _good(path: Path) -> bool:
        return path.exists() and path.is_file() and path.stat().st_size > 0

    if _good(raw_img_guess) and _good(raw_hdr_guess):
        raw_envi_img = raw_img_guess
        raw_envi_hdr = raw_hdr_guess
    else:
        discovered_img, discovered_hdr = _pick_uncorrected_envi_pair(
            base_folder=base_folder,
            flight_stem=flight_stem,
        )
        raw_envi_img = discovered_img if discovered_img is not None else raw_img_guess
        raw_envi_hdr = discovered_hdr if discovered_hdr is not None else raw_hdr_guess

    corrected_img = base_folder / f"{flight_stem}_brdfandtopo_corrected_envi.img"
    corrected_hdr = base_folder / f"{flight_stem}_brdfandtopo_corrected_envi.hdr"
    correction_json = base_folder / f"{flight_stem}_brdfandtopo_corrected_envi.json"

    sensor_products: Dict[str, Path] = {
        "landsat_tm": base_folder / f"{flight_stem}_landsat_tm.tif",
        "landsat_etm+": base_folder / f"{flight_stem}_landsat_etm+.tif",
        "landsat_oli": base_folder / f"{flight_stem}_landsat_oli.tif",
        "landsat_oli2": base_folder / f"{flight_stem}_landsat_oli2.tif",
        "micasense": base_folder / f"{flight_stem}_micasense.tif",
    }

    return {
        "h5": h5_path,
        "raw_envi_img": raw_envi_img,
        "raw_envi_hdr": raw_envi_hdr,
        "correction_json": correction_json,
        "corrected_img": corrected_img,
        "corrected_hdr": corrected_hdr,
        "sensor_products": sensor_products,
    }


__all__ = ["get_flightline_products"]
