"""HyTools-free NEON to ENVI exporter used by the production pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from .envi_writer import EnviWriter
from .file_types import NEONReflectanceENVIFile, NEONReflectanceFile
from .neon_cube import NeonCube
from .progress_utils import TileProgressReporter


def _resolve_output(envi_file: NEONReflectanceENVIFile) -> Path:
    """Return the ENVI stem that should be written for ``envi_file``."""

    return envi_file.path.with_suffix("")


def neon_to_envi_no_hytools(
    images: Iterable[str],
    output_dir: str,
    brightness_offset: float | None = None,
    *,
    interactive_mode: bool = True,
    log_every: int = 25,
) -> list[dict]:
    """Convert NEON `.h5` reflectance cubes into ENVI BSQ rasters.

    The implementation streams chunks out of :class:`NeonCube` and writes them via
    :class:`EnviWriter`. It is restart-safe: valid outputs are reused automatically
    when the function is called again.
    """

    image_list = list(images)
    if len(image_list) != 1:
        raise NotImplementedError(
            "neon_to_envi_no_hytools currently supports exactly one NEON HDF5 "
            f"input; received {len(image_list)} files."
        )

    h5_path = Path(image_list[0])
    if not h5_path.exists():
        raise FileNotFoundError(f"NEON HDF5 file not found: {h5_path}")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    cube = NeonCube(h5_path=h5_path)

    header = cube.build_envi_header()
    header["description"] = (
        "Uncorrected NEON reflectance (float32 BSQ) exported via NeonCube"
    )

    try:
        refl_file = NEONReflectanceFile.from_filename(h5_path)
    except ValueError as exc:
        raise RuntimeError(
            "Unable to parse NEON reflectance filename. Expected one of the standard "
            "NEON patterns (tile+directional, legacy date+time, coordinate-based, "
            "or bidirectional reflectance)."
        ) from exc

    envi_file = NEONReflectanceENVIFile.from_components(
        domain=refl_file.domain,
        site=refl_file.site,
        product=refl_file.product,
        tile=refl_file.tile,
        date=refl_file.date,
        time=refl_file.time,
        directional=getattr(refl_file, "directional", False),
        folder=output_dir_path,
    )

    out_stem = _resolve_output(envi_file)
    writer = EnviWriter(out_stem, header)

    offset_value: np.float32 | None = None
    if brightness_offset is not None:
        offset_value = np.float32(brightness_offset)

    chunk_y = 100
    chunk_x = 100
    total_chunks = cube.chunk_count(chunk_y=chunk_y, chunk_x=chunk_x)
    reporter = TileProgressReporter(
        stage_name="ENVI export",
        total_tiles=total_chunks,
        interactive_mode=interactive_mode,
        log_every=log_every,
    )

    try:
        for ys, ye, xs, xe, raw_chunk in cube.iter_chunks(
            chunk_y=chunk_y, chunk_x=chunk_x
        ):
            chunk = raw_chunk.astype("float32", copy=False)
            if offset_value is not None:
                chunk = chunk + offset_value
            writer.write_chunk(chunk, ys, xs)
            reporter.update(1)
    finally:
        writer.close()
        reporter.close()

    metadata = {
        "hdr": str(out_stem.with_suffix(".hdr")),
        "img": str(out_stem.with_suffix(".img")),
        "lines": cube.lines,
        "samples": cube.columns,
        "bands": cube.bands,
    }

    return [metadata]


def _main(argv: Iterable[str] | None = None) -> None:
    """Simple CLI wrapper around :func:`neon_to_envi_no_hytools`."""

    parser = argparse.ArgumentParser(
        description="Convert a NEON AOP HDF5 flight line into an ENVI (.img/.hdr) pair."
    )
    parser.add_argument("image", help="Path to the NEON HDF5 reflectance product")
    parser.add_argument("output_dir", help="Directory where ENVI outputs will be written")
    parser.add_argument(
        "--brightness-offset",
        type=float,
        default=None,
        help="Optional scalar added to every reflectance value before writing.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    neon_to_envi_no_hytools([
        args.image,
    ], args.output_dir, brightness_offset=args.brightness_offset)


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    _main()
