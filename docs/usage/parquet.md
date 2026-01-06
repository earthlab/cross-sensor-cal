# Working with Parquet Outputs

The SpectralBridge pipeline writes Parquet files for each ENVI product it generates. These tables contain one row per pixel and are optimized for high-performance analytics with DuckDB, pandas, or xarray.

---

## Why Parquet?

- Supports efficient columnar reads  
- Compresses well for large datasets  
- Easily queryable using SQL (DuckDB)  
- Allows out-of-core or streaming access  

---

## File structure

Typical Parquet file naming:

*_brdfandtopo_corrected.parquet
*_landsat_convolved.parquet
*_merged_pixel_extraction.parquet

Each file contains columns for:

- reflectance values  
- band metadata  
- masks  
- pixel coordinates  
- optional ancillary variables  

---

## Quick preview using DuckDB

```python
import duckdb

duckdb.query("""
    SELECT *
    FROM '..._brdfandtopo_corrected.parquet'
    LIMIT 5
""").df()
DuckDB provides efficient SQL queries without needing to load the entire dataset into memory.
Checking dimensions
duckdb.query("""
    SELECT COUNT(*) AS nrows
    FROM '..._landsat_convolved.parquet'
""").df()
Loading with pandas
import pandas as pd
df = pd.read_parquet("..._merged_pixel_extraction.parquet")
df.head()
Use with caution for large flight lines.
Loading with xarray
Parquet → xarray workflows work best after pivoting or aggregating data. For full spatial cubes, ENVI files remain easier to load.
Streaming large files
DuckDB can scan files lazily:
duckdb.query("""
    SELECT AVG(NIR), AVG(Red)
    FROM '..._landsat_convolved.parquet'
""").df()
This avoids loading the full table.
Next steps
CLI usage
Pipeline outputs

---
