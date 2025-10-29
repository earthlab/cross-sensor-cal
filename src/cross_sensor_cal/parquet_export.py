"""Helpers to convert ENVI reflectance cubes into tabular Parquet summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ._optional import require_rasterio
from .envi_header import _parse_envi_header


def _require_pyarrow():
    try:
        import pyarrow as pa
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise ModuleNotFoundError(
            "The 'pyarrow' package is required for Parquet export. Install it with `pip install pyarrow`."
        ) from exc
    return pa


def _require_pyarrow_parquet():
    return _require_pyarrow().parquet


def parquet_exists_and_valid(parquet_path: Path) -> bool:
    parquet_path = Path(parquet_path)
    return parquet_path.exists() and parquet_path.is_file() and parquet_path.stat().st_size > 0


def _iter_windows(width: int, height: int, chunk_size: int):
    """Yield rasterio windows that roughly contain ``chunk_size`` pixels."""

    chunk_pixels = max(1, int(chunk_size))
    rows_per_chunk = chunk_pixels // max(1, width)
    if rows_per_chunk <= 0:
        rows_per_chunk = 1
    rows_per_chunk = min(rows_per_chunk, height)

    if rows_per_chunk * width <= chunk_pixels:
        cols_per_chunk = width
    else:
        cols_per_chunk = max(1, chunk_pixels // rows_per_chunk)
    cols_per_chunk = min(cols_per_chunk, width)

    for row_off in range(0, height, rows_per_chunk):
        window_height = min(rows_per_chunk, height - row_off)
        for col_off in range(0, width, cols_per_chunk):
            window_width = min(cols_per_chunk, width - col_off)
            yield row_off, col_off, window_height, window_width


def envi_to_parquet(
    envi_img: Path,
    envi_hdr: Path,
    parquet_path: Path,
    *,
    chunk_size: int = 2048,
) -> None:
    """Stream an ENVI cube into a Parquet table with pixel spectra."""

    rasterio = require_rasterio()
    pq_module = _require_pyarrow_parquet()
    pa_module = _require_pyarrow()

    img_path = Path(envi_img)
    hdr_path = Path(envi_hdr)
    parquet_path = Path(parquet_path)

    header = _parse_envi_header(hdr_path)
    wavelengths = header.get("wavelength")
    wavelengths_arr: Optional[np.ndarray] = None
    if isinstance(wavelengths, list) and wavelengths:
        try:
            wavelengths_arr = np.asarray(wavelengths, dtype=np.float32)
        except ValueError:
            wavelengths_arr = None

    img_path = Path(img_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(img_path) as src:
        band_count = src.count
        height, width = src.height, src.width
        transform = src.transform

        writer = None
        try:
            for row_off, col_off, window_height, window_width in _iter_windows(
                width, height, chunk_size
            ):
                window = rasterio.windows.Window(
                    col_off=col_off, row_off=row_off, width=window_width, height=window_height
                )
                data = src.read(window=window)
                if data.size == 0:
                    continue

                rows = np.arange(row_off, row_off + window_height, dtype=np.int64)
                cols = np.arange(col_off, col_off + window_width, dtype=np.int64)
                row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")

                if transform is not None:
                    xs, ys = rasterio.transform.xy(
                        transform,
                        row_grid,
                        col_grid,
                        offset="center",
                    )
                    xs = np.asarray(xs, dtype=np.float64)
                    ys = np.asarray(ys, dtype=np.float64)
                else:
                    xs = col_grid.astype(np.float64)
                    ys = row_grid.astype(np.float64)

                pixel_count = xs.size
                xs_flat = xs.reshape(-1)
                ys_flat = ys.reshape(-1)

                band_indices = np.tile(np.arange(band_count, dtype=np.int32), pixel_count)
                x_out = np.repeat(xs_flat, band_count)
                y_out = np.repeat(ys_flat, band_count)

                reflectance = data.reshape(band_count, pixel_count).T.reshape(-1).astype(np.float32)

                if wavelengths_arr is not None and len(wavelengths_arr) == band_count:
                    wavelength_values = np.tile(wavelengths_arr, pixel_count)
                else:
                    wavelength_values = np.full(reflectance.shape, np.nan, dtype=np.float32)

                table = pa_module.table(
                    {
                        "x": pa_module.array(x_out, type=pa_module.float64()),
                        "y": pa_module.array(y_out, type=pa_module.float64()),
                        "band": pa_module.array(band_indices, type=pa_module.int32()),
                        "wavelength_nm": pa_module.array(wavelength_values, type=pa_module.float32()),
                        "reflectance": pa_module.array(reflectance, type=pa_module.float32()),
                    }
                )

                if writer is None:
                    writer = pq_module.ParquetWriter(str(parquet_path), table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()


def ensure_parquet_for_envi(
    envi_img: Path,
    envi_hdr: Path,
    logger,
    *,
    chunk_size: int = 2048,
) -> Optional[Path]:
    """Create a Parquet sidecar for an ENVI cube if missing."""

    img_path = Path(envi_img)
    hdr_path = Path(envi_hdr)
    parquet_path = img_path.with_suffix(".parquet")

    if parquet_exists_and_valid(parquet_path):
        logger.info("⏭️  Skipped Parquet export for %s: already present", img_path.stem)
        return parquet_path

    try:
        envi_to_parquet(img_path, hdr_path, parquet_path, chunk_size=chunk_size)
    except Exception as exc:  # pragma: no cover - logged for pipeline resilience
        logger.warning("⚠️  Failed Parquet export for %s: %s", img_path.stem, exc)
        return None

    if parquet_exists_and_valid(parquet_path):
        logger.info("✅ Wrote Parquet for %s → %s", img_path.stem, parquet_path.name)
        return parquet_path

    logger.warning(
        "⚠️  Parquet export for %s produced an invalid file at %s", img_path.stem, parquet_path
    )
    return None


__all__ = ["parquet_exists_and_valid", "envi_to_parquet", "ensure_parquet_for_envi"]
