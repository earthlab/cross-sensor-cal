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


def _crs_from_header_text(header_text: str) -> int | str | None:
    match = re.search(
        r"coordinate system string\s*=\s*(.+)",
        header_text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    match = re.search(r"map info\s*=\s*\{([^}]*)\}", header_text, re.IGNORECASE)
    if not match:
        return None

    parts = [part.strip() for part in match.group(1).split(",")]
    try:
        utm_zone = int(parts[7])
        hemisphere = parts[8].lower()
    except (IndexError, ValueError):
        return None

    if "north" in hemisphere:
        return 32600 + utm_zone
    if "south" in hemisphere:
        return 32700 + utm_zone
    return None


def build_parquet_from_envi(
    envi_img: Path,
    envi_hdr: Path,
    parquet_path: Path,
    chunk_size: int = 2048,
) -> None:
    """Read an ENVI cube and emit a wide per-pixel Parquet table."""

    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.windows import Window
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cross_sensor_cal.exports.schema_utils import (
        SENSOR_WAVELENGTHS_NM,
        ensure_coord_columns,
        infer_stage_from_name,
        parse_envi_wavelengths_nm,
        sort_and_rename_spectral_columns,
    )

    envi_img = Path(envi_img)
    envi_hdr = Path(envi_hdr)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        parquet_path.unlink()

    stage_key = infer_stage_from_name(parquet_path.name)

    header_text: str | None = None
    wavelengths_nm: list[int] | None = None
    try:
        header_text = envi_hdr.read_text(encoding="utf-8", errors="ignore")
        wavelengths_nm = parse_envi_wavelengths_nm(header_text)
    except OSError:
        header_text = None

    if wavelengths_nm is None:
        for sensor_key, centers in SENSOR_WAVELENGTHS_NM.items():
            if sensor_key in parquet_path.name.lower():
                wavelengths_nm = centers
                break

    with rasterio.open(envi_img) as src:
        band_count = src.count
        height = src.height
        width = src.width
        transform = src.transform

        crs_candidate: int | str | None = None
        if src.crs is not None:
            try:
                crs_candidate = src.crs.to_epsg()
            except Exception:
                crs_candidate = None
            if crs_candidate is None:
                try:
                    crs_candidate = src.crs.to_string()
                except Exception:
                    crs_candidate = None
        if crs_candidate is None and header_text:
            crs_candidate = _crs_from_header_text(header_text)
        if crs_candidate is None:
            raise ValueError(f"Cannot determine CRS for {envi_img}")

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

        if wavelengths_nm:
            band_wavelengths: list[int] = []
            for idx in range(band_count):
                if idx < len(wavelengths_nm):
                    band_wavelengths.append(int(round(wavelengths_nm[idx])))
                elif band_wavelengths:
                    band_wavelengths.append(band_wavelengths[-1] + 1)
                else:
                    band_wavelengths.append(idx + 1)
        else:
            band_wavelengths = [idx + 1 for idx in range(band_count)]

        nodata = src.nodata
        writer: pq.ParquetWriter | None = None
        wrote_rows = False
        try:
            for row_start in range(0, height, block_y):
                row_end = min(row_start + block_y, height)
                for col_start in range(0, width, block_x):
                    col_end = min(col_start + block_x, width)
                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    block = src.read(window=window)
                    if block.size == 0:
                        continue

                    rows = row_end - row_start
                    cols = col_end - col_start
                    spectra = block.reshape(band_count, -1).T

                    mask = np.zeros(spectra.shape[0], dtype=bool)
                    if nodata is not None:
                        mask |= np.any(spectra == nodata, axis=1)
                    mask |= ~np.isfinite(spectra).all(axis=1)
                    valid_mask = ~mask
                    if not valid_mask.any():
                        continue

                    spectra = spectra[valid_mask].astype("float32", copy=False)
                    row_indices = np.repeat(np.arange(row_start, row_end), cols)[valid_mask]
                    col_indices = np.tile(np.arange(col_start, col_end), rows)[valid_mask]

                    df_chunk = pd.DataFrame(
                        spectra,
                        columns=[f"wl{band_wavelengths[idx]:04d}" for idx in range(band_count)],
                    )
                    df_chunk["row"] = row_indices.astype("int32", copy=False)
                    df_chunk["col"] = col_indices.astype("int32", copy=False)
                    df_chunk["pixel_id"] = (
                        row_indices.astype("int64", copy=False) * width
                        + col_indices.astype("int64", copy=False)
                    )
                    df_chunk["source_image"] = pd.Series(
                        envi_img.name, index=df_chunk.index, dtype="string"
                    )
                    if isinstance(crs_candidate, int):
                        df_chunk["epsg"] = pd.Series(
                            crs_candidate, index=df_chunk.index, dtype="Int64"
                        )
                        df_chunk["crs"] = pd.Series(pd.NA, index=df_chunk.index, dtype="string")
                    else:
                        df_chunk["epsg"] = pd.Series(pd.NA, index=df_chunk.index, dtype="Int64")
                        df_chunk["crs"] = pd.Series(
                            str(crs_candidate) if crs_candidate else pd.NA,
                            index=df_chunk.index,
                            dtype="string",
                        )

                    df_chunk = ensure_coord_columns(
                        df_chunk,
                        transform=transform,
                        crs_epsg=crs_candidate,
                    )

                    if "lon" not in df_chunk or "lat" not in df_chunk:
                        raise ValueError(f"Unable to compute lon/lat for {parquet_path}")

                    df_chunk = sort_and_rename_spectral_columns(
                        df_chunk,
                        stage_key=stage_key,
                        wavelengths_nm=band_wavelengths,
                    )

                    wrote_rows = True
                    table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, table.schema)
                    writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    if not parquet_path.exists() or not wrote_rows:
        meta_fields = [
            ("pixel_id", pa.int64()),
            ("row", pa.int32()),
            ("col", pa.int32()),
            ("x", pa.float64()),
            ("y", pa.float64()),
            ("lon", pa.float64()),
            ("lat", pa.float64()),
            ("source_image", pa.string()),
            ("epsg", pa.int32()),
            ("crs", pa.string()),
        ]
        spectral_fields = [
            (
                f"{stage_key}_b{idx + 1:03d}_wl{band_wavelengths[idx]:04d}nm",
                pa.float32(),
            )
            for idx in range(len(band_wavelengths))
        ]
        schema = pa.schema(meta_fields + spectral_fields)
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
