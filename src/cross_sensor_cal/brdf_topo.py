"""BRDF and topographic correction helpers for the streamlined pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from .corrections import (
    apply_brdf_correct,
    apply_topo_correct,
    fit_and_save_brdf_model,
)
from .envi_writer import EnviWriter
from .neon_cube import NeonCube
from .progress_utils import TileProgressReporter
from .utils_checks import is_valid_envi_pair, is_valid_json


logger = logging.getLogger(__name__)

_REQUIRED_SUFFIX = "_reflectance_envi"
_CORRECTED_SUFFIX = "_reflectance_brdfandtopo_corrected_envi"


def _derive_corrected_stem(raw_img_path: Path) -> str:
    stem = raw_img_path.stem
    if stem.endswith(_REQUIRED_SUFFIX):
        return stem[: -len(_REQUIRED_SUFFIX)] + _CORRECTED_SUFFIX
    if stem.endswith(_REQUIRED_SUFFIX.replace("_reflectance", "")):
        return stem + "_brdfandtopo_corrected_envi"
    if _CORRECTED_SUFFIX in stem:
        return stem
    return f"{stem}_brdfandtopo_corrected_envi"


def build_correction_parameters_dict(
    *,
    h5_path: Path,
    raw_img_path: Path,
    raw_hdr_path: Path,
    base_folder: Path,
    flight_stem: str | None = None,
    product_code: str | None = None,
) -> dict:
    """Compute the correction parameter payload without writing it to disk."""

    del flight_stem, product_code  # maintained for future extensibility

    h5_path = Path(h5_path)
    raw_img_path = Path(raw_img_path)
    raw_hdr_path = Path(raw_hdr_path)
    base_folder = Path(base_folder)
    base_folder.mkdir(parents=True, exist_ok=True)

    corrected_stem = _derive_corrected_stem(raw_img_path)

    cube = NeonCube(h5_path=h5_path)
    coeff_path = fit_and_save_brdf_model(cube, base_folder)

    geometry_stats: dict[str, dict[str, float]] = {}
    ancillary_keys = (
        "solar_zn",
        "solar_az",
        "sensor_zn",
        "sensor_az",
        "slope",
        "aspect",
    )
    for key in ancillary_keys:
        try:
            array = cube.get_ancillary(key, radians=True)
        except Exception as exc:  # pragma: no cover - optional ancillary failures
            logger.warning("⚠️  Failed to extract ancillary '%s': %s", key, exc)
            continue
        try:
            min_val = float(np.nanmin(array))
        except ValueError:  # pragma: no cover - all values NaN
            min_val = float("nan")
        try:
            max_val = float(np.nanmax(array))
        except ValueError:  # pragma: no cover - all values NaN
            max_val = float("nan")

        geometry_stats[key] = {
            "mean": float(np.nanmean(array, dtype=np.float64)),
            "std": float(np.nanstd(array, dtype=np.float64)),
            "min": min_val,
            "max": max_val,
        }

    return {
        "base_key": cube.base_key,
        "stem": corrected_stem,
        "lines": cube.lines,
        "samples": cube.columns,
        "bands": cube.bands,
        "h5_path": str(h5_path.resolve()),
        "raw_img_path": str(raw_img_path.resolve()),
        "raw_hdr_path": str(raw_hdr_path.resolve()),
        "coefficients_path": str(coeff_path.resolve()),
        "geometry": geometry_stats,
        "notes": "generated before BRDF/topo correction",
    }


def build_and_write_correction_json(
    h5_path: Path,
    raw_img_path: Path,
    raw_hdr_path: Path,
    out_dir: Path,
) -> Path:
    """Persist BRDF/topographic correction parameters for a flightline."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = build_correction_parameters_dict(
        h5_path=h5_path,
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        base_folder=out_dir,
    )

    corrected_stem = params.get("stem")
    if not corrected_stem:
        corrected_stem = _derive_corrected_stem(Path(raw_img_path))

    json_path = out_dir / f"{corrected_stem}.json"

    if is_valid_json(json_path):
        logger.info("✅ Correction JSON already complete for %s, skipping", corrected_stem)
        return json_path

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    if not is_valid_json(json_path):
        raise RuntimeError(f"Failed to write correction JSON: {json_path}")

    logger.info("📝 Correction parameters saved: %s", json_path)
    return json_path


def apply_brdf_topo_core(
    *,
    raw_img_path: Path,
    raw_hdr_path: Path,
    params: dict,
    out_img_path: Path,
    out_hdr_path: Path,
    interactive_mode: bool = True,
    log_every: int = 25,
) -> None:
    """Run the BRDF+topographic correction using ``params`` into ``out_*`` paths."""

    raw_img_path = Path(raw_img_path)
    raw_hdr_path = Path(raw_hdr_path)
    out_img_path = Path(out_img_path)
    out_hdr_path = Path(out_hdr_path)
    out_img_path.parent.mkdir(parents=True, exist_ok=True)

    source_h5 = Path(params.get("h5_path", "")) if isinstance(params, dict) else None
    if source_h5 is None or not source_h5.exists():
        raise RuntimeError(
            "Correction parameters missing valid 'h5_path'; cannot apply BRDF/topo correction."
        )

    coeff_path = Path(params.get("coefficients_path", "")) if isinstance(params, dict) else None
    if coeff_path and not coeff_path.exists():
        logger.warning(
            "⚠️  BRDF coefficient file referenced by JSON is missing: %s", coeff_path
        )
        coeff_path = None

    cube = NeonCube(h5_path=source_h5)

    header = cube.build_envi_header()
    header["description"] = (
        "BRDF + topographic corrected reflectance (float32); generated by cross-sensor-cal pipeline"
    )
    header.setdefault("data type", 4)
    header.setdefault("byte order", 0)
    if hasattr(cube, "no_data"):
        header.setdefault("data ignore value", float(getattr(cube, "no_data")))

    writer = EnviWriter(out_img_path.with_suffix(""), header)

    chunk_y = 100
    chunk_x = 100
    total_chunks = cube.chunk_count(chunk_y=chunk_y, chunk_x=chunk_x)
    reporter = TileProgressReporter(
        stage_name="BRDF+topo correction",
        total_tiles=total_chunks,
        interactive_mode=interactive_mode,
        log_every=log_every,
    )

    try:
        for ys, ye, xs, xe, raw_chunk in cube.iter_chunks(
            chunk_y=chunk_y, chunk_x=chunk_x
        ):
            chunk = np.asarray(raw_chunk, dtype=np.float32)
            corrected_chunk = apply_topo_correct(cube, chunk, ys, ye, xs, xe)
            corrected_chunk = apply_brdf_correct(
                cube,
                corrected_chunk,
                ys,
                ye,
                xs,
                xe,
                coeff_path=coeff_path,
            )
            corrected_chunk = corrected_chunk.astype(np.float32, copy=False)
            writer.write_chunk(corrected_chunk, ys, xs)
            reporter.update(1)
    finally:
        writer.close()
        reporter.close()

    if not is_valid_envi_pair(out_img_path, out_hdr_path):
        raise RuntimeError(f"BRDF/topo correction failed for {out_img_path}")


def apply_brdf_topo_correction(
    *,
    raw_img_path: Path,
    raw_hdr_path: Path,
    correction_json_path: Path,
    out_dir: Path,
    interactive_mode: bool = True,
    log_every: int = 25,
) -> Tuple[Path, Path]:
    """Apply BRDF + topographic correction using precomputed parameters."""

    raw_img_path = Path(raw_img_path)
    raw_hdr_path = Path(raw_hdr_path)
    correction_json_path = Path(correction_json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corrected_stem = _derive_corrected_stem(raw_img_path)
    corrected_img_path = out_dir / f"{corrected_stem}.img"
    corrected_hdr_path = out_dir / f"{corrected_stem}.hdr"

    if is_valid_envi_pair(corrected_img_path, corrected_hdr_path):
        logger.info("✅ BRDF+topo correction already complete for %s, skipping", corrected_stem)
        return corrected_img_path, corrected_hdr_path

    if not is_valid_json(correction_json_path):
        raise RuntimeError(
            "Missing or invalid correction JSON before correction for "
            f"{corrected_stem}: {correction_json_path}"
        )

    with correction_json_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    json_stem = params.get("stem") if isinstance(params, dict) else None
    if json_stem:
        corrected_stem = json_stem
        corrected_img_path = out_dir / f"{corrected_stem}.img"
        corrected_hdr_path = out_dir / f"{corrected_stem}.hdr"

    if is_valid_envi_pair(corrected_img_path, corrected_hdr_path):
        logger.info("✅ BRDF+topo correction already complete for %s, skipping", corrected_stem)
        return corrected_img_path, corrected_hdr_path

    apply_brdf_topo_core(
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        params=params,
        out_img_path=corrected_img_path,
        out_hdr_path=corrected_hdr_path,
        interactive_mode=interactive_mode,
        log_every=log_every,
    )

    logger.info("✅ Corrected ENVI saved: %s", corrected_img_path)
    return corrected_img_path, corrected_hdr_path


__all__ = [
    "build_correction_parameters_dict",
    "build_and_write_correction_json",
    "apply_brdf_topo_core",
    "apply_brdf_topo_correction",
]
