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
    "pq_read_table",
    "pq_write_table",
    "repair_lonlat_in_place",
]


def pq_read_table(path: Path):
    import pyarrow.parquet as pq

    return pq.read_table(path)


def pq_write_table(obj, path: Path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    if isinstance(obj, pa.Table):
        table = obj
    elif hasattr(obj, "to_pandas") and not isinstance(obj, pd.DataFrame):
        table = obj
    else:
        if isinstance(obj, pd.DataFrame):
            df = obj
        else:
            df = pd.DataFrame(obj)
        table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    return path


def _add_lonlat_columns(df, img_path: Path, *, transform=None, crs=None):
    import numpy as np
    import rasterio
    from rasterio.transform import xy
    import pyproj

    img_path = Path(img_path)

    if transform is None or crs is None:
        with rasterio.open(img_path) as ds:
            return _add_lonlat_columns(
                df,
                img_path,
                transform=ds.transform,
                crs=ds.crs,
            )

    if "x" in df.columns and "y" in df.columns:
        cols = df["x"].to_numpy()
        rows = df["y"].to_numpy()
    elif {"col", "row"}.issubset(df.columns):
        cols = df["col"].to_numpy()
        rows = df["row"].to_numpy()
    else:
        raise RuntimeError(
            "Missing pixel coordinates (x/y or row/col) for lon/lat export"
        )

    xs, ys = xy(transform, rows, cols, offset="center")
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    epsg = None
    if crs is not None:
        try:
            epsg = crs.to_epsg()
        except Exception:
            epsg = None

    if crs is not None and epsg and epsg != 4326:
        to_wgs84 = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lons, lats = to_wgs84.transform(xs, ys)
    else:
        lons, lats = xs, ys

    df["lon"] = lons
    df["lat"] = lats
    return df


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
        src_crs = src.crs

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

                    df_chunk = _add_lonlat_columns(
                        df_chunk,
                        envi_img,
                        transform=transform,
                        crs=src_crs,
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


def repair_lonlat_in_place(folder: Path):
    from cross_sensor_cal.merge_duckdb import EXCLUDE_PATTERNS

    folder = Path(folder)
    for pq_path in sorted(folder.glob("*.parquet")):
        name = pq_path.name
        if any(ex in name for ex in EXCLUDE_PATTERNS):
            continue
        tbl = pq_read_table(pq_path)
        if {"lat", "lon"}.issubset(tbl.column_names):
            continue
        base = (
            name.replace(".parquet", "")
            .replace("_envi", "")
            .replace("_brdfandtopo_corrected", "")
        )
        candidates = list(folder.glob(f"{base}*_envi.img"))
        if not candidates:
            continue
        img_path = candidates[0]
        df = tbl.to_pandas()
        df = _add_lonlat_columns(df, img_path)
        pq_write_table(df, pq_path)


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
