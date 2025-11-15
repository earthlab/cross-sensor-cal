from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Polygon

import duckdb

from cross_sensor_cal.paths import FlightlinePaths
from cross_sensor_cal.polygons import (
    build_polygon_pixel_index,
    extract_polygon_parquets_for_flightline,
    merge_polygon_parquets_for_flightline,
    run_polygon_pipeline_for_flightline,
)


geopandas = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")


def _write_envi_raster(path: Path, transform, crs, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="ENVI",
        width=data.shape[-1],
        height=data.shape[-2],
        count=data.shape[0],
        dtype=data.dtype,
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(data)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("df_view", df)
        con.execute(
            "COPY (SELECT * FROM df_view) TO '"
            + str(path).replace("'", "''")
            + "' (FORMAT PARQUET)"
        )
    finally:
        con.close()


def _read_parquet(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(
            "SELECT * FROM read_parquet(?)", [str(path)]
        ).df()
    finally:
        con.close()


def _prepare_flightline(tmp_path: Path) -> tuple[FlightlinePaths, Path]:
    flight_id = "TEST_FLIGHT"
    base_dir = tmp_path / "flight"
    flight_paths = FlightlinePaths(base_dir, flight_id)
    flight_paths.flight_dir.mkdir(parents=True, exist_ok=True)

    transform = from_origin(0.0, 3.0, 1.0, 1.0)
    crs = "EPSG:32613"
    raster_data = np.arange(12, dtype=np.float32).reshape(1, 3, 4)

    _write_envi_raster(flight_paths.envi_img, transform, crs, raster_data)
    _write_envi_raster(flight_paths.corrected_img, transform, crs, raster_data)

    rows, cols = np.meshgrid(np.arange(3, dtype=np.int32), np.arange(4, dtype=np.int32), indexing="ij")
    rows_flat = rows.flatten()
    cols_flat = cols.flatten()
    pixel_ids = rows_flat * 4 + cols_flat
    xs, ys = rasterio.transform.xy(transform, rows_flat, cols_flat, offset="center")

    base_columns = {
        "pixel_id": pixel_ids.astype("int64"),
        "row": rows_flat.astype("int32"),
        "col": cols_flat.astype("int32"),
        "x": np.asarray(xs, dtype="float64"),
        "y": np.asarray(ys, dtype="float64"),
    }

    raw_df = pd.DataFrame(base_columns)
    raw_df["raw_b001_wl0450nm"] = np.linspace(0.0, 1.0, len(raw_df))
    _write_parquet(raw_df, flight_paths.envi_parquet)

    corr_df = pd.DataFrame(base_columns)
    corr_df["corr_b001_wl0450nm"] = np.linspace(1.0, 2.0, len(corr_df))
    _write_parquet(corr_df, flight_paths.corrected_parquet)

    landsat_df = pd.DataFrame(base_columns)
    landsat_df["landsat_tm_b001_wl0485nm"] = np.linspace(2.0, 3.0, len(landsat_df))
    landsat_path = flight_paths.sensor_product("landsat_tm").parquet
    _write_parquet(landsat_df, landsat_path)

    polygon = Polygon([(0.0, 3.0), (2.0, 3.0), (2.0, 1.0), (0.0, 1.0)])
    polygons = geopandas.GeoDataFrame(
        {"species": ["fir"], "polygon_id": [101]},
        geometry=[polygon],
        crs=crs,
    )
    polygons_path = tmp_path / "polygons.gpkg"
    polygons.to_file(polygons_path, driver="GPKG")

    return flight_paths, polygons_path


def test_build_polygon_pixel_index(tmp_path: Path) -> None:
    flight_paths, polygons_path = _prepare_flightline(tmp_path)

    index_path = build_polygon_pixel_index(flight_paths, polygons_path)
    assert index_path.exists()
    df = _read_parquet(index_path)
    assert set(df["pixel_id"]) == {0, 1, 4, 5}
    assert "polygon_geometry_wkb" in df.columns
    assert (df["polygon_id"].unique() == [101]).all()


def test_extract_polygon_parquets_for_flightline(tmp_path: Path) -> None:
    flight_paths, polygons_path = _prepare_flightline(tmp_path)
    index_path = build_polygon_pixel_index(flight_paths, polygons_path)

    outputs = extract_polygon_parquets_for_flightline(
        flight_paths,
        index_path,
        products=["envi", "brdfandtopo_corrected_envi", "landsat_tm_envi"],
    )
    assert set(outputs) == {"envi", "brdfandtopo_corrected_envi", "landsat_tm_envi"}
    for path in outputs.values():
        df = _read_parquet(path)
        assert set(df["pixel_id"]) <= {0, 1, 4, 5}


def test_merge_polygon_parquets_for_flightline(tmp_path: Path) -> None:
    flight_paths, polygons_path = _prepare_flightline(tmp_path)
    index_path = build_polygon_pixel_index(flight_paths, polygons_path)
    outputs = extract_polygon_parquets_for_flightline(
        flight_paths,
        index_path,
        products=["envi", "brdfandtopo_corrected_envi", "landsat_tm_envi"],
    )

    merged_path = merge_polygon_parquets_for_flightline(
        flight_paths,
        index_path,
        outputs,
    )
    assert merged_path.exists()
    merged = _read_parquet(merged_path)
    assert set(merged["pixel_id"]) == {0, 1, 4, 5}
    assert "corr_b001_wl0450nm" in merged.columns
    assert "landsat_tm_b001_wl0485nm" in merged.columns
    assert "polygon_id" in merged.columns


def test_run_polygon_pipeline_for_flightline(tmp_path: Path) -> None:
    flight_paths, polygons_path = _prepare_flightline(tmp_path)

    result = run_polygon_pipeline_for_flightline(
        flight_paths,
        polygons_path,
        products=["envi", "brdfandtopo_corrected_envi", "landsat_tm_envi"],
        overwrite=True,
    )
    assert Path(result["polygon_index_path"]).exists()
    merged_path = Path(result["polygon_merged_parquet"])
    assert merged_path.exists()
    merged = _read_parquet(merged_path)
    assert "raw_b001_wl0450nm" in merged.columns
