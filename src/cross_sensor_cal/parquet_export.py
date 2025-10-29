"""Parquet export utilities for ENVI reflectance products.

The helpers in this module are intentionally lightweight at import time so that
unit tests can import :mod:`cross_sensor_cal.parquet_export` without requiring
heavy geospatial dependencies. Expensive libraries (``rasterio``, ``pandas``,
``pyarrow``) are imported lazily inside :func:`build_parquet_from_envi`.
"""

from __future__ import annotations

from pathlib import Path
import re

__all__ = [
    "parquet_exists_and_valid",
    "build_parquet_from_envi",
    "ensure_parquet_for_envi",
]


def parquet_exists_and_valid(parquet_path: Path) -> bool:
    """Return ``True`` when ``parquet_path`` exists and is non-empty."""

    try:
        path = Path(parquet_path)
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _extract_wavelengths(envi_hdr: Path, band_count: int) -> list[float] | None:
    """Parse ``wavelength`` metadata from an ENVI header, if present."""

    try:
        text = Path(envi_hdr).read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    raw_block = match.group(1).replace("\n", " ")
    tokens = [token.strip() for token in raw_block.split(",") if token.strip()]
    wavelengths: list[float] = []
    for token in tokens:
        try:
            wavelengths.append(float(token))
        except ValueError:
            return None

    if len(wavelengths) != band_count:
        return None

    return wavelengths


def _chunk_to_dataframe(
    *,
    chunk,
    row_offset: int,
    col_offset: int,
    wavelengths: list[float] | None,
):
    """Convert a chunked raster block into a pandas DataFrame."""

    import numpy as np
    import pandas as pd

    if chunk.size == 0:
        return pd.DataFrame(columns=["y", "x", "band", "wavelength_nm", "reflectance"])

    chunk = np.moveaxis(chunk, 0, -1)  # (rows, cols, bands)
    rows, cols, bands = chunk.shape
    if rows == 0 or cols == 0 or bands == 0:
        return pd.DataFrame(columns=["y", "x", "band", "wavelength_nm", "reflectance"])

    yy = np.arange(row_offset, row_offset + rows, dtype=np.int32)
    xx = np.arange(col_offset, col_offset + cols, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")

    pixel_count = rows * cols
    band_indices = np.arange(bands, dtype=np.int32)

    y_vals = np.repeat(grid_y.reshape(-1), bands)
    x_vals = np.repeat(grid_x.reshape(-1), bands)
    band_vals = np.tile(band_indices, pixel_count)
    reflectance_vals = chunk.reshape(-1, bands).reshape(-1).astype("float32", copy=False)

    if wavelengths is not None:
        wl_array = np.asarray(wavelengths, dtype="float32")
        wavelength_vals = wl_array[band_vals]
    else:
        wavelength_vals = np.full(reflectance_vals.shape, np.nan, dtype="float32")

    data = {
        "y": y_vals.astype("int32", copy=False),
        "x": x_vals.astype("int32", copy=False),
        "band": band_vals.astype("int32", copy=False),
        "wavelength_nm": wavelength_vals,
        "reflectance": reflectance_vals,
    }
    return pd.DataFrame(data)


def build_parquet_from_envi(
    envi_img: Path,
    envi_hdr: Path,
    parquet_path: Path,
    chunk_size: int = 2048,
) -> None:
    """Read an ENVI cube and emit a long-form Parquet table."""

    import rasterio
    from rasterio.windows import Window
    import pyarrow as pa
    import pyarrow.parquet as pq

    envi_img = Path(envi_img)
    envi_hdr = Path(envi_hdr)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        parquet_path.unlink()

    with rasterio.open(envi_img) as src:
        band_count = src.count
        height = src.height
        width = src.width

        wavelengths = _extract_wavelengths(envi_hdr, band_count)

        if src.block_shapes:
            block_y, block_x = src.block_shapes[0]
            if block_y <= 0:
                block_y = chunk_size
            if block_x <= 0:
                block_x = chunk_size
        else:
            block_y = chunk_size
            block_x = chunk_size

        block_y = max(1, min(chunk_size, block_y))
        block_x = max(1, min(chunk_size, block_x))

        writer: pq.ParquetWriter | None = None
        try:
            for row_start in range(0, height, block_y):
                row_end = min(row_start + block_y, height)
                for col_start in range(0, width, block_x):
                    col_end = min(col_start + block_x, width)
                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    chunk = src.read(window=window)
                    if chunk.size == 0:
                        continue
                    df = _chunk_to_dataframe(
                        chunk=chunk,
                        row_offset=row_start,
                        col_offset=col_start,
                        wavelengths=wavelengths,
                    )
                    if df.empty:
                        continue
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, table.schema)
                    writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    if not parquet_path.exists():
        schema = pa.schema(
            [
                ("y", pa.int32()),
                ("x", pa.int32()),
                ("band", pa.int32()),
                ("wavelength_nm", pa.float32()),
                ("reflectance", pa.float32()),
            ]
        )
        empty_table = pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in schema],
            names=[field.name for field in schema],
        )
        pq.write_table(empty_table, parquet_path)


def ensure_parquet_for_envi(envi_img: Path, logger) -> Path | None:
    """Ensure a ``.parquet`` sidecar exists for ``envi_img``."""

    envi_img = Path(envi_img)
    if not envi_img.exists() or envi_img.stat().st_size == 0:
        logger.warning(
            "⚠️ Cannot export Parquet for %s because .img is missing or empty",
            envi_img.name,
        )
        return None

    parquet_path = envi_img.with_suffix(".parquet")
    if parquet_exists_and_valid(parquet_path):
        logger.info(
            "⏭️ Parquet already present for %s -> %s (skipping)",
            envi_img.name,
            parquet_path.name,
        )
        return parquet_path

    envi_hdr = envi_img.with_suffix(".hdr")
    if not envi_hdr.exists() or envi_hdr.stat().st_size == 0:
        logger.warning(
            "⚠️ Cannot export Parquet for %s because .hdr is missing or empty",
            envi_img.name,
        )
        return None

    try:
        build_parquet_from_envi(envi_img, envi_hdr, parquet_path)
        logger.info(
            "✅ Wrote Parquet for %s -> %s",
            envi_img.name,
            parquet_path.name,
        )
        return parquet_path
    except Exception as exc:  # pragma: no cover - surfaced via warning log
        logger.warning("⚠️ Failed Parquet export for %s: %s", envi_img.name, exc)
        return None
