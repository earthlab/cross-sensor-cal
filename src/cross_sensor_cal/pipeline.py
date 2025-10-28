"""High-level orchestration utilities for the cross-sensor-cal workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from .brdf_topo import build_and_write_correction_json as _build_json_fn
    from .brdf_topo import apply_brdf_topo_correction as _apply_correction_fn
    from .convolution import convolve_all_sensors as _convolve_fn
    from .neon_to_envi import neon_to_envi_no_hytools as _export_fn


logger = logging.getLogger(__name__)


def export_envi_from_h5(h5_path: Path, out_dir: Path) -> Tuple[Path, Path]:
    """Export a NEON directional reflectance HDF5 file to ENVI format.

    Parameters
    ----------
    h5_path
        Path to the NEON ``*_directional_reflectance.h5`` file.
    out_dir
        Destination directory for the exported ENVI files.

    Returns
    -------
    tuple[Path, Path]
        Paths to the exported ``.img`` and ``.hdr`` files respectively.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from .neon_to_envi import neon_to_envi_no_hytools

    export_metadata = neon_to_envi_no_hytools([str(h5_path)], str(out_dir))
    if not export_metadata:
        raise RuntimeError(f"Failed to export ENVI from {h5_path}")

    first_entry = export_metadata[0]
    raw_img = Path(first_entry["img"]).resolve()
    raw_hdr = Path(first_entry["hdr"]).resolve()

    if not raw_img.exists() or not raw_hdr.exists():
        raise RuntimeError(
            f"ENVI export for {h5_path} did not produce expected files in {out_dir}"
        )

    logger.info("📦 Exported ENVI reflectance: %s", raw_hdr)
    return raw_img, raw_hdr


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
    """Run the full processing workflow for a single NEON flightline.

    Steps
    -----
    1. Export raw ENVI from the HDF5 directional reflectance cube.
    2. Compute BRDF/topographic correction metadata and persist as JSON.
    3. Apply the correction using the persisted parameters.
    4. Spectrally convolve the corrected ENVI to the requested sensor bandpasses.
    """

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

    if correction_json_path is None or not correction_json_path.exists():
        raise RuntimeError(
            f"Failed to create correction JSON for {h5_path.name} in {out_dir}"
        )

    corrected_img_path, corrected_hdr_path = apply_brdf_topo_correction(
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        correction_json_path=correction_json_path,
        out_dir=out_dir,
    )

    if not corrected_img_path.exists() or not corrected_hdr_path.exists():
        raise RuntimeError(
            f"BRDF/topo correction did not produce corrected ENVI for {h5_path.name}"
        )
    if "_brdfandtopo_corrected_envi" not in corrected_img_path.name:
        raise RuntimeError(
            "Corrected ENVI filename missing required suffix "
            f"(got {corrected_img_path.name})"
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
