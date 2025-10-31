# Parquet Preview

=== "pandas"
    ```python
    import pandas as pd
    df = pd.read_parquet("merged/demo_merged_pixel_extraction.parquet")
    df.head()
    print([c for c in df.columns if c.startswith("band_")][:10])
    ```

=== "DuckDB"
    ```python
    import duckdb
    con = duckdb.connect()
    con.execute("SELECT * FROM 'merged/demo_merged_pixel_extraction.parquet' LIMIT 5").df()
    ```

=== "polars"
    ```python
    import polars as pl
    df = pl.read_parquet("merged/demo_merged_pixel_extraction.parquet")
    df.head(5)
    ```
