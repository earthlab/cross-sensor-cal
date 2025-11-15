# Polygon spectral library pipeline

The polygon workflow augments the standard flightline processing by producing
per-polygon spectral libraries from the existing pixel-level Parquet products.
The pipeline is optional today and can be invoked programmatically without
changing the default `cscal-pipeline` behaviour.

## Overview

The polygon pipeline performs three steps for each flightline:

1. **Polygon–pixel index** – rasterises a polygon layer against a chosen NEON
   product (BRDF/topography corrected ENVI by default) and records the pixels
   that intersect each polygon, including spatial metadata and polygon
   attributes, in ``*_polygon_pixel_index.parquet``.
2. **Polygon-only Parquets** – filters the existing per-product Parquet tables
   so that only pixels found in the index are retained.  Outputs follow the
   pattern ``*_envi_polygons.parquet``,
   ``*_brdfandtopo_corrected_envi_polygons.parquet`` and
   ``*_landsat_tm_envi_polygons.parquet`` (etc.).
3. **Merged polygon spectral library** – joins the polygon-only Parquets with
   the polygon index to produce a compact spectral library for all polygons in a
   flightline: ``*_polygons_merged_pixel_extraction.parquet``.

The helper functions live in :mod:`cross_sensor_cal.polygons` and are available
for bespoke workflows while we evaluate the approach.

## Data requirements

* A processed flightline directory with the usual per-product Parquet tables.
* A polygon vector file readable by GeoPandas (e.g. GeoPackage, Shapefile,
  GeoJSON).  The repository ships with a sample data set at
  ``Datasets/niwot_aop_polygons_2023_12_8_23_analysis_ready_half_diam.gpkg``.

Polygons are reprojected automatically to match the reference raster used for
the index.  A ``polygon_id`` column is honoured if present; otherwise a unique
identifier is generated.

## Example usage

```python
from cross_sensor_cal.paths import FlightlinePaths
from cross_sensor_cal.polygons import run_polygon_pipeline_for_flightline

flight_paths = FlightlinePaths("/data/flightlines", "NEON_D12_NIWO_2021")
polygons_path = "Datasets/niwot_aop_polygons_2023_12_8_23_analysis_ready_half_diam.gpkg"

result = run_polygon_pipeline_for_flightline(
    flight_paths,
    polygons_path,
    products=[
        "envi",
        "brdfandtopo_corrected_envi",
        "landsat_tm_envi",
        "landsat_oli_envi",
        "micasense_envi",
    ],
)

print(result["polygon_index_path"])
print(result["polygon_merged_parquet"])
```

Each helper is also available individually for advanced scenarios:

* :func:`build_polygon_pixel_index` – create the polygon pixel lookup table.
* :func:`extract_polygon_parquets_for_flightline` – generate polygon-only
  Parquet tables for a subset of products.
* :func:`merge_polygon_parquets_for_flightline` – merge polygon Parquets into a
  single spectral library.

## Status and roadmap

The polygon pipeline is **opt-in** today and does not run as part of the
standard flightline orchestration.  The utilities introduced here allow data
teams to experiment with polygon spectral libraries ahead of tighter
integration in a future release.
