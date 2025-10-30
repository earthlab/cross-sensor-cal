# Previewing precursor and merged Parquets

Use DuckDB to peek at the merged pixel extraction table alongside its precursor Parquet files.
Update `fl_dir` to the flightline of interest.

```python
import duckdb, glob, os

fl_dir = "/path/to/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"

merged = os.path.join(
    fl_dir,
    "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance_merged_pixel_extraction.parquet",
)
precursors = sorted(glob.glob(os.path.join(fl_dir, "*_envi.parquet"))) + \
             sorted(glob.glob(os.path.join(fl_dir, "*_brdfandtopo_corrected_envi.parquet"))) + \
             sorted(glob.glob(os.path.join(fl_dir, "*_landsat_*_envi.parquet"))) + \
             sorted(glob.glob(os.path.join(fl_dir, "*_micasense*_envi.parquet")))

con = duckdb.connect()
# merged head
con.execute(f"SELECT * FROM read_parquet('{merged}') LIMIT 5").df()

# peek one precursor from each stage/sensor
for p in precursors:
    print("→", os.path.basename(p))
    df = con.execute(f"SELECT * FROM read_parquet('{p}') LIMIT 5").df()
    display(df.head())

# quick schema check: verify spectral names and order
schema_check = con.execute(
    """
    WITH cols AS (
      SELECT column_name
      FROM parquet_schema($path)
    )
    SELECT *
    FROM cols
    WHERE column_name ~ '.*_b[0-9]{3}_wl[0-9]{4}nm$'
    LIMIT 10
    """,
    {'path': precursors[0]},
).df()
schema_check
```
