# DEPRECATED: Superseded by cross_sensor_cal.pipelines.pipeline.go_forth_and_multiply().
# The active pipeline enforces ENVI-only, idempotent stages defined in src/cross_sensor_cal/pipelines/pipeline.py.
# This file has been staged for removal.
"""High-level orchestration utilities for the cross-sensor-cal workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .file_types import NEONReflectanceENVIFile, NEONReflectanceFile
from .utils_checks import is_valid_envi_pair, is_valid_json

if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from .brdf_topo import build_and_write_correction_json as _build_json_fn
    from .brdf_topo import apply_brdf_topo_correction as _apply_correction_fn
    from .convolution import convolve_all_sensors as _convolve_fn
    from .neon_to_envi import neon_to_envi_no_hytools as _export_fn


logger = logging.getLogger(__name__)


def _derive_raw_envi_paths(h5_path: Path, out_dir: Path) -> tuple[Path, Path, str]:
    """Compute the expected raw ENVI paths for a given NEON HDF5 file."""

    refl = NEONReflectanceFile.from_filename(h5_path)
    if not refl.tile:
        raise RuntimeError(
            "Unable to determine NEON tile identifier from HDF5 filename; "
            "expected standard NEON naming convention."
        )

    envi = NEONReflectanceENVIFile.from_components(
        domain=refl.domain,
        site=refl.site,
        product=refl.product or "DP1.30006.001",
        tile=refl.tile,
        date=refl.date,
        time=refl.time,
        directional=refl.directional,
        folder=out_dir,
    )

    img_path = envi.path
    hdr_path = img_path.with_suffix(".hdr")
    stem = img_path.stem
    return img_path, hdr_path, stem


def export_envi_from_h5(h5_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Export a NEON directional reflectance HDF5 file to ENVI format."""

    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_img_path, raw_hdr_path, stem = _derive_raw_envi_paths(h5_path, out_dir)

    if is_valid_envi_pair(raw_img_path, raw_hdr_path):
        logger.info("✅ ENVI export already complete for %s, skipping", stem)
        return raw_img_path, raw_hdr_path

    from .neon_to_envi import neon_to_envi_no_hytools

    export_metadata = neon_to_envi_no_hytools([str(h5_path)], str(out_dir))
    if not export_metadata:
        raise RuntimeError(f"Failed to export ENVI from {h5_path}")

    first_entry = export_metadata[0]
    raw_img_path = Path(first_entry["img"]).resolve()
    raw_hdr_path = Path(first_entry["hdr"]).resolve()

    if not is_valid_envi_pair(raw_img_path, raw_hdr_path):
        raise RuntimeError(f"ENVI export failed for {h5_path}")

    logger.info("📦 Exported ENVI reflectance: %s", raw_hdr_path)
    return raw_img_path, raw_hdr_path


def build_and_write_correction_json(*args, **kwargs):
    from .brdf_topo import build_and_write_correction_json as _impl

    return _impl(*args, **kwargs)


def apply_brdf_topo_correction(*args, **kwargs):
    from .brdf_topo import apply_brdf_topo_correction as _impl

    return _impl(*args, **kwargs)


def convolve_all_sensors(*args, **kwargs):
    from .convolution import convolve_all_sensors as _impl

    return _impl(*args, **kwargs)


def process_flightline(
    h5_path: Path,
    out_dir: Path,
    *,
    sensor_list: list[str] | None = None,
) -> None:
    """Full per-flightline pipeline with idempotent skipping."""

    logger = logging.getLogger(__name__)

    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("📦 Exporting ENVI (no HyTools) [%s]...", h5_path.name)
    raw_img_path, raw_hdr_path = export_envi_from_h5(h5_path=h5_path, out_dir=out_dir)

    correction_json_path = build_and_write_correction_json(
        h5_path=h5_path,
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        out_dir=out_dir,
    )

    if not is_valid_json(correction_json_path):
        raise RuntimeError(
            f"Correction JSON invalid for {h5_path.name}: {correction_json_path}"
        )

    corrected_img_path, corrected_hdr_path = apply_brdf_topo_correction(
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        correction_json_path=correction_json_path,
        out_dir=out_dir,
    )

    if not is_valid_envi_pair(corrected_img_path, corrected_hdr_path):
        raise RuntimeError(
            f"Corrected ENVI invalid for {h5_path.name}: {corrected_img_path}"
        )

    logger.info("🎯 Convolving corrected reflectance: %s", corrected_hdr_path)
    convolve_all_sensors(
        corrected_img_path=corrected_img_path,
        corrected_hdr_path=corrected_hdr_path,
        out_dir=out_dir,
        sensor_list=sensor_list,
    )

    logger.info("🎉 Pipeline complete!")


__all__ = [
    "export_envi_from_h5",
    "process_flightline",
]
