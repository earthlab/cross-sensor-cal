"""Polygon extraction helpers for building spectral libraries.

This module provides utilities for extracting polygon-based subsets from the
per-pixel Parquet products that the Cross-Sensor Calibration pipeline already
produces.  The helpers are intentionally orthogonal to the default flightline
pipeline so that they can be orchestrated separately while the workflow is
stabilised.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Mapping

import duckdb
import numpy as np
import pandas as pd

from spectralbridge.exports.schema_utils import ensure_coord_columns

from ._optional import require_geopandas, require_rasterio
from .paths import FlightlinePaths

LOGGER = logging.getLogger(__name__)


def _quote_path(path: Path) -> str:
    """Escape a filesystem path for embedding inside DuckDB SQL strings."""

    return str(path).replace("'", "''")


def _quote_identifier(name: str) -> str:
    """Return a double-quoted SQL identifier."""

    return '"' + name.replace('"', '""') + '"'


def _sanitise_alias(name: str) -> str:
    """Return a DuckDB-safe identifier derived from ``name``."""

    alias = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not alias:
        alias = "tbl"
    if alias[0].isdigit():
        alias = f"t_{alias}"
    return alias


def _resolve_reference_raster(
    flight_paths: FlightlinePaths, reference_product: str
) -> tuple[Path, Path]:
    """Return ``(img, hdr)`` paths for the requested reference product."""

    reference_product = reference_product.strip()
    if reference_product == "brdfandtopo_corrected_envi":
        return flight_paths.corrected_img, flight_paths.corrected_hdr
    if reference_product == "envi":
        return flight_paths.envi_img, flight_paths.envi_hdr
    if reference_product.endswith("_envi"):
        sensor_name = reference_product[: -len("_envi")]
        sensor_paths = flight_paths.sensor_product(sensor_name)
        return sensor_paths.img, sensor_paths.hdr
    raise ValueError(
        "Unsupported reference_product value: "
        f"{reference_product!r}. Expected one of 'envi', "
        "'brdfandtopo_corrected_envi', or '<sensor>_envi'."
    )


def _available_product_parquets(flight_paths: FlightlinePaths) -> Dict[str, Path]:
    """Return known per-product parquet paths for ``flight_paths``."""

    products: Dict[str, Path] = {
        "envi": flight_paths.envi_parquet,
        "brdfandtopo_corrected_envi": flight_paths.corrected_parquet,
    }
    for sensor_name, sensor_paths in flight_paths.sensor_products.items():
        products[f"{sensor_name}_envi"] = sensor_paths.parquet
    return products


def _describe_parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    """Return column names for a parquet file using DuckDB DESCRIBE."""

    sql = f"DESCRIBE SELECT * FROM read_parquet('{_quote_path(path)}')"
    return [row[0] for row in con.execute(sql).fetchall()]


def _write_dataframe_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to parquet via DuckDB to avoid pyarrow dependency."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("df_view", df)
        con.execute(
            "COPY (SELECT * FROM df_view) TO '"
            + _quote_path(path)
            + "' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        con.close()


def build_polygon_pixel_index(
    flight_paths: FlightlinePaths,
    polygons_path: str | Path,
    output_path: Path | str | None = None,
    *,
    reference_product: str = "brdfandtopo_corrected_envi",
    overwrite: bool = False,
    all_touched: bool = False,
) -> Path:
    """Create a pixel→polygon lookup table for ``flight_paths``.

    The resulting Parquet table includes one row per pixel that intersects any
    polygon.  Columns comprise the canonical pixel identifiers plus polygon
    attributes and geometry stored as WKB.
    """

    polygons_path = Path(polygons_path)
    if not polygons_path.exists():
        raise FileNotFoundError(polygons_path)

    if output_path is None:
        output_path = (
            flight_paths.flight_dir
            / f"{flight_paths.flight_id}_polygon_pixel_index.parquet"
        )
    else:
        output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        LOGGER.info("[polygons-index] Reusing existing index → %s", output_path)
        return output_path

    geopandas = require_geopandas()
    rasterio = require_rasterio()
    from rasterio.features import rasterize
    from rasterio.transform import xy

    img_path, hdr_path = _resolve_reference_raster(flight_paths, reference_product)
    if not img_path.exists():
        raise FileNotFoundError(
            f"Reference image for {reference_product!r} not found: {img_path}"
        )
    if not hdr_path.exists():
        raise FileNotFoundError(
            f"Reference header for {reference_product!r} not found: {hdr_path}"
        )

    polygons = geopandas.read_file(polygons_path)
    if polygons.empty:
        raise ValueError(f"No polygons were found in {polygons_path}")

    with rasterio.open(img_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        dataset_crs = src.crs
        crs_epsg = dataset_crs.to_epsg() if dataset_crs else None

    if polygons.crs is None and dataset_crs is not None:
        LOGGER.warning(
            "[polygons-index] Polygon source has no CRS; assuming %s", dataset_crs
        )
        polygons = polygons.set_crs(dataset_crs)
    elif dataset_crs is not None and polygons.crs != dataset_crs:
        polygons = polygons.to_crs(dataset_crs)

    polygons = polygons.reset_index(drop=True).copy()
    if "polygon_id" in polygons and polygons["polygon_id"].is_unique:
        polygon_ids = polygons["polygon_id"].astype("int64", copy=False)
    else:
        polygon_ids = pd.Series(
            np.arange(1, len(polygons) + 1, dtype="int64"), name="polygon_id"
        )
        polygons["polygon_id"] = polygon_ids

    # Prepare rasterisation shapes (skip empties)
    shapes = []
    for geom, pid in zip(polygons.geometry, polygons["polygon_id"]):
        if geom is None or geom.is_empty:
            continue
        shapes.append((geom, int(pid)))

    if not shapes:
        raise ValueError("All polygons were empty; nothing to index")

    LOGGER.info(
        "[polygons-index] Rasterising %s polygons against %s", len(shapes), img_path
    )

    polygon_grid = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=all_touched,
    )

    mask = polygon_grid > 0
    if not mask.any():
        raise ValueError("No pixels intersected the supplied polygons")

    rows, cols = np.nonzero(mask)
    polygon_ids = polygon_grid[rows, cols].astype("int64", copy=False)
    xs, ys = xy(transform, rows, cols, offset="center")

    df = pd.DataFrame(
        {
            "pixel_id": rows.astype("int64") * width + cols.astype("int64"),
            "row": rows.astype("int32"),
            "col": cols.astype("int32"),
            "x": np.asarray(xs, dtype="float64"),
            "y": np.asarray(ys, dtype="float64"),
            "polygon_id": polygon_ids,
        }
    )
    df["flight_id"] = flight_paths.flight_id
    df["polygon_source"] = str(polygons_path)
    df["reference_product"] = reference_product
    if dataset_crs is not None:
        try:
            df["raster_crs"] = dataset_crs.to_string()
        except Exception:  # pragma: no cover - defensive
            df["raster_crs"] = str(dataset_crs)
    if crs_epsg is not None:
        df["epsg"] = pd.Series(crs_epsg, index=df.index, dtype="Int64")

    df = ensure_coord_columns(df, transform=transform, crs_epsg=crs_epsg or 0)

    attribute_columns = [
        col for col in polygons.columns if col != polygons.geometry.name
    ]
    polygon_attrs = polygons[attribute_columns].copy()
    polygon_attrs["polygon_geometry_wkb"] = polygons.geometry.to_wkb()
    df = df.merge(polygon_attrs, on="polygon_id", how="left")

    _write_dataframe_parquet(df, output_path)
    LOGGER.info(
        "[polygons-index] ✅ Wrote polygon pixel index → %s (%s rows)",
        output_path,
        len(df),
    )
    return output_path


def extract_polygon_parquets_for_flightline(
    flight_paths: FlightlinePaths,
    polygon_index_path: Path | str,
    products: list[str] | None = None,
    *,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Filter per-product Parquet tables down to polygon pixels."""

    polygon_index_path = Path(polygon_index_path)
    if not polygon_index_path.exists():
        raise FileNotFoundError(polygon_index_path)

    product_paths = _available_product_parquets(flight_paths)
    if products is None:
        target_products = list(product_paths.keys())
    else:
        target_products = products

    outputs: Dict[str, Path] = {}
    con = duckdb.connect()
    try:
        for product in target_products:
            parquet_path = product_paths.get(product)
            if parquet_path is None:
                raise ValueError(f"Unknown product key: {product}")
            if not parquet_path.exists():
                LOGGER.warning(
                    "[polygons-extract] Missing parquet for %s → %s", product, parquet_path
                )
                continue

            out_path = parquet_path.with_name(f"{parquet_path.stem}_polygons.parquet")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not overwrite:
                LOGGER.info(
                    "[polygons-extract] Reusing %s polygons parquet → %s",
                    product,
                    out_path,
                )
                outputs[product] = out_path
                continue

            LOGGER.info(
                "[polygons-extract] Filtering %s → %s", parquet_path.name, out_path.name
            )
            sql = (
                "COPY ("
                "SELECT p.* FROM read_parquet('"
                + _quote_path(parquet_path)
                + "') p "
                "INNER JOIN read_parquet('"
                + _quote_path(polygon_index_path)
                + "') idx USING (pixel_id)"
                ") TO '"
                + _quote_path(out_path)
                + "' (FORMAT PARQUET)"
            )
            con.execute(sql)
            outputs[product] = out_path
            LOGGER.info(
                "[polygons-extract] ✅ Wrote %s polygons parquet → %s", product, out_path
            )
    finally:
        con.close()

    return outputs


def merge_polygon_parquets_for_flightline(
    flight_paths: FlightlinePaths,
    polygon_index_path: Path | str,
    product_polygon_parquets: Mapping[str, Path],
    output_path: Path | str | None = None,
    *,
    overwrite: bool = False,
    row_group_size: int = 25_000,
) -> Path:
    """Merge polygon-filtered Parquet tables into one spectral library."""

    polygon_index_path = Path(polygon_index_path)
    if not polygon_index_path.exists():
        raise FileNotFoundError(polygon_index_path)

    if output_path is None:
        output_path = (
            flight_paths.flight_dir
            / f"{flight_paths.flight_id}_polygons_merged_pixel_extraction.parquet"
        )
    else:
        output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        LOGGER.info(
            "[polygons-merge] Reusing existing polygon merge → %s", output_path
        )
        return output_path

    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")

    filtered_products = {
        key: Path(path)
        for key, path in product_polygon_parquets.items()
        if Path(path).exists()
    }
    if not filtered_products:
        raise ValueError("No polygon parquet tables were provided for merging")

    LOGGER.info(
        "[polygons-merge] Start merge for %s with %d products",
        flight_paths.flight_id,
        len(filtered_products),
    )

    con = duckdb.connect()
    try:
        index_columns = _describe_parquet_columns(con, polygon_index_path)
        select_terms = [
            f"idx.{_quote_identifier(col)} AS {_quote_identifier(col)}"
            for col in index_columns
        ]
        joins: list[str] = []
        seen_columns = set(index_columns)

        for product, parquet_path in filtered_products.items():
            alias = _sanitise_alias(product)
            columns = _describe_parquet_columns(con, parquet_path)
            keep_cols = [
                col
                for col in columns
                if col != "pixel_id" and col not in seen_columns
            ]
            for col in keep_cols:
                select_terms.append(
                    f"{alias}.{_quote_identifier(col)} AS {_quote_identifier(col)}"
                )
                seen_columns.add(col)
            joins.append(
                "LEFT JOIN read_parquet('"
                + _quote_path(parquet_path)
                + f"') {alias} USING (pixel_id)"
            )

        select_sql = (
            "SELECT "
            + ", ".join(select_terms)
            + " FROM read_parquet('"
            + _quote_path(polygon_index_path)
            + "') idx "
            + " ".join(joins)
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        LOGGER.info("[polygons-merge] Writing merged parquet → %s", output_path)
        copy_sql = (
            "COPY ("
            + select_sql
            + ") TO '"
            + _quote_path(output_path)
            + "' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE "
            + str(int(row_group_size))
            + ")"
        )
        con.execute(copy_sql)
    finally:
        con.close()

    LOGGER.info("[polygons-merge] ✅ Polygon spectral library → %s", output_path)
    return output_path


def run_polygon_pipeline_for_flightline(
    flight_paths: FlightlinePaths,
    polygons_path: str | Path,
    *,
    products: list[str] | None = None,
    reference_product: str = "brdfandtopo_corrected_envi",
    overwrite: bool = False,
) -> Dict[str, object]:
    """Execute the polygon pipeline end-to-end for one flightline."""

    polygon_index_path = build_polygon_pixel_index(
        flight_paths,
        polygons_path,
        reference_product=reference_product,
        overwrite=overwrite,
    )
    product_polygon_parquets = extract_polygon_parquets_for_flightline(
        flight_paths,
        polygon_index_path,
        products=products,
        overwrite=overwrite,
    )
    merged_path = merge_polygon_parquets_for_flightline(
        flight_paths,
        polygon_index_path,
        product_polygon_parquets,
        overwrite=overwrite,
    )
    return {
        "polygon_index_path": polygon_index_path,
        "product_polygon_parquets": product_polygon_parquets,
        "polygon_merged_parquet": merged_path,
    }


__all__ = [
    "build_polygon_pixel_index",
    "extract_polygon_parquets_for_flightline",
    "merge_polygon_parquets_for_flightline",
    "run_polygon_pipeline_for_flightline",
]

