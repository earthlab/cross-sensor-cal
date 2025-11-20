"""Parquet export utilities for ENVI reflectance products.

The helpers in this module are intentionally lightweight at import time so that
unit tests can import :mod:`cross_sensor_cal.parquet_export` without requiring
heavy geospatial dependencies. Expensive libraries (``rasterio``, ``pandas``,
``pyarrow``) are imported lazily inside :func:`build_parquet_from_envi`.
"""

from __future__ import annotations

import inspect
from contextlib import suppress
from pathlib import Path
from typing import NamedTuple
import re

from cross_sensor_cal._ray_utils import ray_map

__all__ = [
    "parquet_exists_and_valid",
    "build_parquet_from_envi",
    "ensure_parquet_from_envi",
    "ensure_parquet_for_envi",
    "read_envi_in_chunks",
    "pq_read_table",
    "pq_write_table",
    "repair_lonlat_in_place",
]


class _ParquetChunkJob(NamedTuple):
    row_start: int
    row_end: int
    col_start: int
    col_end: int


class _ParquetChunkContext(NamedTuple):
    envi_img: str
    band_count: int
    band_wavelengths: tuple[int, ...]
    width: int
    nodata: float | int | None
    crs_candidate: int | str | None
    transform: tuple[float, float, float, float, float, float]
    src_crs_wkt: str | None
    stage_key: str


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
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with suppress(FileNotFoundError):
        tmp_path.unlink()

    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(path)
    finally:
        with suppress(FileNotFoundError):
            tmp_path.unlink()

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


def _plan_chunk_jobs(
    envi_img: Path,
    envi_hdr: Path,
    parquet_name: str,
    stage_key: str,
    *,
    chunk_size: int,
) -> tuple[_ParquetChunkContext, list[_ParquetChunkJob], dict[str, object]]:
    import rasterio

    from cross_sensor_cal.exports.schema_utils import (
        SENSOR_WAVELENGTHS_NM,
        parse_envi_wavelengths_nm,
    )

    envi_img = Path(envi_img)
    envi_hdr = Path(envi_hdr)

    try:
        header_text = envi_hdr.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        header_text = None

    wavelengths_nm = None
    if header_text:
        wavelengths_nm = parse_envi_wavelengths_nm(header_text)

    parquet_name_lower = parquet_name.lower()

    with rasterio.open(envi_img) as src:
        band_count = src.count
        height = src.height
        width = src.width
        transform = src.transform
        src_crs = src.crs
        nodata = src.nodata

        crs_candidate = None
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
            matched_centers = None
            for sensor_key, centers in SENSOR_WAVELENGTHS_NM.items():
                if sensor_key in parquet_name_lower:
                    matched_centers = centers
                    break
            if matched_centers is not None:
                band_wavelengths = [int(round(value)) for value in matched_centers]
            else:
                band_wavelengths = [idx + 1 for idx in range(band_count)]

        # --- Ensure band_wavelengths and band_count are consistent ---
        if len(band_wavelengths) < band_count:
            # Pad by extending the last wavelength by 1 nm steps
            if band_wavelengths:
                last = band_wavelengths[-1]
            else:
                last = 0
            pad = [last + i + 1 for i in range(band_count - len(band_wavelengths))]
            band_wavelengths = band_wavelengths + pad
        elif len(band_wavelengths) > band_count:
            # Trim extra centers
            band_wavelengths = band_wavelengths[:band_count]

        if len(band_wavelengths) != band_count:
            raise RuntimeError(
                "Inconsistent band metadata: band_count="
                f"{band_count}, len(band_wavelengths)={len(band_wavelengths)} "
                f"for {envi_img}"
            )

        jobs: list[_ParquetChunkJob] = []
        for row_start in range(0, height, block_y):
            row_end = min(row_start + block_y, height)
            for col_start in range(0, width, block_x):
                col_end = min(col_start + block_x, width)
                jobs.append(
                    _ParquetChunkJob(
                        row_start=row_start,
                        row_end=row_end,
                        col_start=col_start,
                        col_end=col_end,
                    )
                )

    context = _ParquetChunkContext(
        envi_img=str(envi_img),
        band_count=band_count,
        band_wavelengths=tuple(band_wavelengths),
        width=width,
        nodata=nodata,
        crs_candidate=crs_candidate,
        transform=tuple(transform),
        src_crs_wkt=src_crs.to_wkt() if src_crs is not None else None,
        stage_key=stage_key,
    )

    shared: dict[str, object] = {
        "band_wavelengths": list(band_wavelengths),
        "crs_candidate": crs_candidate,
        "transform": transform,
        "src_crs": src_crs,
        "width": width,
        "header_text": header_text,
        "wavelengths_nm": wavelengths_nm,
        "stage_key": stage_key,
    }

    return context, jobs, shared


def _process_chunk_to_dataframe(
    job: _ParquetChunkJob, context: _ParquetChunkContext
):
    import numpy as np
    import pandas as pd
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.windows import Window

    from cross_sensor_cal.exports.schema_utils import (
        ensure_coord_columns,
        sort_and_rename_spectral_columns,
    )

    transform = Affine(*context.transform)
    src_crs = CRS.from_wkt(context.src_crs_wkt) if context.src_crs_wkt else None

    with rasterio.open(context.envi_img) as src:
        window = Window(
            job.col_start,
            job.row_start,
            job.col_end - job.col_start,
            job.row_end - job.row_start,
        )
        block = src.read(window=window)

    if block.size == 0:
        return pd.DataFrame()

    rows = job.row_end - job.row_start
    cols = job.col_end - job.col_start
    band_count = context.band_count
    spectra = block.reshape(band_count, -1).T

    mask = np.zeros(spectra.shape[0], dtype=bool)
    if context.nodata is not None:
        mask |= np.any(spectra == context.nodata, axis=1)
    mask |= ~np.isfinite(spectra).all(axis=1)
    valid_mask = ~mask
    if not valid_mask.any():
        return pd.DataFrame()

    spectra = spectra[valid_mask].astype("float32", copy=False)
    row_indices = np.repeat(np.arange(job.row_start, job.row_end), cols)[valid_mask]
    col_indices = np.tile(np.arange(job.col_start, job.col_end), rows)[valid_mask]

    band_wavelengths = list(context.band_wavelengths)
    df_chunk = pd.DataFrame(
        spectra,
        columns=[f"wl{band_wavelengths[idx]:04d}" for idx in range(band_count)],
    )
    df_chunk["row"] = row_indices.astype("int32", copy=False)
    df_chunk["col"] = col_indices.astype("int32", copy=False)
    df_chunk["pixel_id"] = (
        row_indices.astype("int64", copy=False) * context.width
        + col_indices.astype("int64", copy=False)
    )
    df_chunk["source_image"] = pd.Series(
        Path(context.envi_img).name, index=df_chunk.index, dtype="string"
    )
    if isinstance(context.crs_candidate, int):
        df_chunk["epsg"] = pd.Series(
            context.crs_candidate, index=df_chunk.index, dtype="Int64"
        )
        df_chunk["crs"] = pd.Series(pd.NA, index=df_chunk.index, dtype="string")
    else:
        df_chunk["epsg"] = pd.Series(pd.NA, index=df_chunk.index, dtype="Int64")
        df_chunk["crs"] = pd.Series(
            str(context.crs_candidate) if context.crs_candidate else pd.NA,
            index=df_chunk.index,
            dtype="string",
        )

    df_chunk = ensure_coord_columns(
        df_chunk,
        transform=transform,
        crs_epsg=context.crs_candidate if isinstance(context.crs_candidate, int) else None,
    )

    df_chunk = _add_lonlat_columns(
        df_chunk,
        Path(context.envi_img),
        transform=transform,
        crs=src_crs,
    )

    df_chunk = sort_and_rename_spectral_columns(
        df_chunk,
        stage_key=context.stage_key,
        wavelengths_nm=band_wavelengths,
    )

    return df_chunk


def read_envi_in_chunks(
    envi_img: Path,
    envi_hdr: Path,
    parquet_name: str,
    *,
    chunk_size: int = 50_000,
):
    """Yield DataFrame chunks from an ENVI cube along with shared metadata."""

    from cross_sensor_cal.exports.schema_utils import infer_stage_from_name

    stage_key = infer_stage_from_name(parquet_name)

    context, jobs, shared = _plan_chunk_jobs(
        Path(envi_img),
        Path(envi_hdr),
        parquet_name,
        stage_key,
        chunk_size=chunk_size,
    )

    def _iterator():
        for job in jobs:
            df_chunk = _process_chunk_to_dataframe(job, context)
            if not df_chunk.empty:
                yield df_chunk

    iterator = _iterator()
    try:
        setattr(iterator, "context", shared)
        return iterator
    except AttributeError:
        # Generators cannot always accept attributes; wrap to carry context for callers
        class _IteratorWrapper:
            def __init__(self, gen):
                self._gen = gen
                self.context = shared

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._gen)

        return _IteratorWrapper(iterator)


def _build_empty_parquet_table(stage_key: str, band_wavelengths: list[int]):
    import pyarrow as pa

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
    return pa.Table.from_arrays(
        [pa.array([], type=field.type) for field in schema],
        names=[field.name for field in schema],
    )


def _write_parquet_chunks(
    parquet_path: Path,
    chunk_iter,
    stage_key: str,
    *,
    context: dict[str, object] | None = None,
    row_group_size: int | None = None,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    with suppress(FileNotFoundError):
        tmp_path.unlink()

    writer = None
    wrote_rows = False
    try:
        for df_chunk in chunk_iter:
            if getattr(df_chunk, "empty", False):
                continue
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            try:
                from cross_sensor_cal.pipelines.pipeline import _to_canonical_table
            except ImportError:  # pragma: no cover - defensive fallback
                canonical = table
            else:
                canonical = _to_canonical_table(table)
            table = canonical
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table, row_group_size=row_group_size)
            wrote_rows = True
    finally:
        if writer is not None:
            writer.close()

    if not wrote_rows:
        ctx = context or getattr(chunk_iter, "context", {})
        band_wavelengths = list(ctx.get("band_wavelengths") or [])
        empty_table = _build_empty_parquet_table(stage_key, band_wavelengths)
        pq.write_table(empty_table, tmp_path, row_group_size=row_group_size)

    tmp_path.replace(parquet_path)


def parquet_exists_and_valid(parquet_path: Path) -> bool:
    """Return ``True`` when ``parquet_path`` exists and has a readable schema."""

    path = Path(parquet_path)
    try:
        if not (path.is_file() and path.stat().st_size > 0):
            return False
        import pyarrow.parquet as pq

        pq.read_schema(path)
        return True
    except Exception:
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
    chunk_size: int = 50_000,
    *,
    num_cpus: int | None = None,
) -> None:
    """Read an ENVI cube and emit a wide per-pixel Parquet table."""

    parquet_path = Path(parquet_path)
    parquet_name = parquet_path.name
    from cross_sensor_cal.exports.schema_utils import infer_stage_from_name

    stage_key = infer_stage_from_name(parquet_name)
    if num_cpus is not None and num_cpus <= 0:
        chunk_iter = read_envi_in_chunks(
            Path(envi_img),
            Path(envi_hdr),
            parquet_name,
            chunk_size=chunk_size,
        )
        write_kwargs: dict[str, object] = {}
        if hasattr(chunk_iter, "context"):
            write_kwargs["context"] = getattr(chunk_iter, "context")

        parameters = inspect.signature(_write_parquet_chunks).parameters
        if "row_group_size" in parameters:
            write_kwargs["row_group_size"] = chunk_size

        _write_parquet_chunks(parquet_path, chunk_iter, stage_key, **write_kwargs)
        return

    context, jobs, shared = _plan_chunk_jobs(
        Path(envi_img),
        Path(envi_hdr),
        parquet_name,
        stage_key,
        chunk_size=chunk_size,
    )

    def _worker(job: _ParquetChunkJob):
        return _process_chunk_to_dataframe(job, context)

    if jobs:
        chunk_tables = ray_map(_worker, jobs, num_cpus=num_cpus)
    else:
        chunk_tables = []

    filtered_tables = (
        df for df in chunk_tables if getattr(df, "empty", False) is False
    )

    write_kwargs: dict[str, object] = {"context": shared}
    parameters = inspect.signature(_write_parquet_chunks).parameters
    if "row_group_size" in parameters:
        write_kwargs["row_group_size"] = chunk_size

    _write_parquet_chunks(
        parquet_path,
        filtered_tables,
        stage_key,
        **write_kwargs,
    )


def ensure_parquet_from_envi(
    envi_img: Path,
    envi_hdr: Path,
    parquet_path: Path,
    *,
    chunk_size: int = 50_000,
    num_cpus: int | None = None,
) -> Path:
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        try:
            pq.read_schema(parquet_path)
            return parquet_path
        except Exception:
            with suppress(FileNotFoundError):
                parquet_path.unlink()

    build_parquet_from_envi(
        envi_img,
        envi_hdr,
        parquet_path,
        chunk_size=chunk_size,
        num_cpus=num_cpus,
    )
    return parquet_path


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


def ensure_parquet_for_envi(
    envi_img: Path,
    logger,
    *,
    chunk_size: int = 2048,
    ray_cpus: int | None = None,
) -> Path | None:
    """Ensure a ``.parquet`` sidecar exists for ``envi_img``.

    This treats the `_undarkened_envi.img/.hdr` convolution-only products the
    same as their brightness-adjusted counterparts so QA comparisons stay
    aligned.
    """

    envi_img = Path(envi_img)
    if not envi_img.exists() or envi_img.stat().st_size == 0:
        logger.warning(
            "⚠️ Cannot export Parquet for %s because .img is missing or empty",
            envi_img.name,
        )
        return None

    parquet_path = envi_img.with_suffix(".parquet")

    if parquet_path.exists():
        if parquet_exists_and_valid(parquet_path):
            logger.info(
                "⏭️ Parquet already present for %s -> %s (skipping)",
                envi_img.name,
                parquet_path.name,
            )
            return parquet_path

        issue = "missing or empty file"
        if parquet_path.is_file() and parquet_path.stat().st_size > 0:
            try:
                import pyarrow.parquet as pq

                pq.read_schema(parquet_path)
            except Exception as exc:  # pragma: no cover - exact errors vary
                issue = exc
            else:  # pragma: no cover - defensive, should not occur
                issue = "unknown validation issue"

        logger.warning(
            "⚠️ Existing Parquet for %s is invalid (%s); regenerating",
            envi_img.name,
            issue,
        )
        try:
            parquet_path.unlink()
        except OSError as unlink_exc:
            logger.warning(
                "⚠️ Cannot remove invalid Parquet for %s (%s)",
                envi_img.name,
                unlink_exc,
            )
            return None

    envi_hdr = envi_img.with_suffix(".hdr")
    if not envi_hdr.exists() or envi_hdr.stat().st_size == 0:
        logger.warning(
            "⚠️ Cannot export Parquet for %s because .hdr is missing or empty",
            envi_img.name,
        )
        return None

    try:
        ensure_parquet_from_envi(
            envi_img,
            envi_hdr,
            parquet_path,
            chunk_size=chunk_size,
            num_cpus=ray_cpus,
        )
        logger.info(
            "✅ Wrote Parquet for %s -> %s",
            envi_img.name,
            parquet_path.name,
        )
        return parquet_path
    except Exception as exc:  # pragma: no cover - surfaced via warning log
        logger.warning("⚠️ Failed Parquet export for %s: %s", envi_img.name, exc)
        return None
