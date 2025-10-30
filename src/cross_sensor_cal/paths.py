from __future__ import annotations

from pathlib import Path
import re
import shutil


def scene_prefix_from_dir(flightline_dir: Path) -> str:
    flightline_dir = Path(flightline_dir)
    # Prefer *_envi.img; else *_directional_reflectance.h5; else folder name
    imgs = sorted(flightline_dir.glob("*_envi.img"))
    if imgs:
        stem = imgs[0].stem
        return stem[:-5] if stem.endswith("_envi") else stem
    h5s = sorted(flightline_dir.glob("*_directional_reflectance.h5"))
    if h5s:
        return h5s[0].stem
    return flightline_dir.name


_site_re = re.compile(r"NEON_[A-Z0-9]+_([A-Z]{4})_DP1_")


def site_from_prefix(prefix: str) -> str | None:
    m = _site_re.search(prefix)
    return m.group(1) if m else None


def normalize_brdf_model_path(flightline_dir: Path) -> Path | None:
    """
    Ensure the BRDF model JSON in ``flightline_dir`` matches the scene prefix:
        <prefix>_brdf_model.json

    If a legacy file like 'NIWO_brdf_model.json' exists, rename it.
    Returns the normalized Path (or None if nothing found).
    """
    flightline_dir = Path(flightline_dir)
    prefix = scene_prefix_from_dir(flightline_dir)
    target = flightline_dir / f"{prefix}_brdf_model.json"
    if target.exists():
        return target

    # legacy site-level name e.g., NIWO_brdf_model.json
    site = site_from_prefix(prefix)
    if site:
        legacy = flightline_dir / f"{site}_brdf_model.json"
        if legacy.exists():
            shutil.move(str(legacy), str(target))
            return target
    # also accept any stray *_brdf_model.json and normalize to target
    for p in flightline_dir.glob("*_brdf_model.json"):
        if p != target:
            shutil.move(str(p), str(target))
            return target
    return None


__all__ = [
    "scene_prefix_from_dir",
    "site_from_prefix",
    "normalize_brdf_model_path",
]
