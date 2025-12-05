# Jupyter Notebook Example

This page shows a complete Jupyter-style workflow using cross-sensor-cal:

1. Run the pipeline from Python  
2. Preview the merged Parquet table  
3. Compute a simple NDVI diagnostic  

You can copy these cells directly into a notebook.

---

## 1. Imports and configuration

```python
import os
import duckdb
import pandas as pd

from cross_sensor_cal import go_forth_and_multiply

base_folder = "output_notebook_demo"
site_code = "NIWO"
year_month = "2023-08"
product_code = "DP1.30006.001"
flight_lines = ["NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"]
2. Run the pipeline from Python
go_forth_and_multiply(
    base_folder=base_folder,
    site_code=site_code,
    year_month=year_month,
    product_code=product_code,
    flight_lines=flight_lines,
    max_workers=2,
    engine="thread",  # start with thread; switch to "ray" if configured
)
This cell will:
download NEON HDF5 tiles if needed
export ENVI cubes
apply topographic + BRDF correction
convolve to Landsat (if configured)
write Parquet tables
generate QA PNG, PDF, and JSON
3. Locate the merged Parquet file
fl_prefix = flight_lines[0]
fl_dir = os.path.join(base_folder, fl_prefix)

merged = os.path.join(
    fl_dir,
    f"{fl_prefix}_merged_pixel_extraction.parquet",
)
merged
4. Preview the merged table with DuckDB
con = duckdb.connect()

head_df = con.execute(
    f"SELECT * FROM read_parquet('{merged}') LIMIT 5"
).df()
head_df
5. Compute a quick NDVI diagnostic
This example assumes Landsat-style band names Red and NIR are present in the merged table.
ndvi_df = con.execute(
    """
    SELECT
      Red,
      NIR,
      (NIR - Red) / NULLIF(NIR + Red, 0) AS ndvi
    FROM read_parquet($merged)
    LIMIT 5000
    """,
    {"merged": merged},
).df()

ndvi_df["ndvi"].describe()
You can extend this pattern to:
filter by masks
aggregate by plot or polygon
export subsets for modeling
6. Visualize NDVI distribution (optional)
If you have matplotlib installed:
import matplotlib.pyplot as plt

ndvi_df["ndvi"].plot.hist(bins=50)
plt.xlabel("NDVI")
plt.ylabel("Count")
plt.title("NDVI distribution (sample)")
plt.show()
Next steps
Quickstart
Working with Parquet outputs
Pipeline overview & stages

---
