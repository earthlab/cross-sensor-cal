# Python API

> Use these functions when you need fine-grained control beyond the CLI.

## Quick example
```python
from cross_sensor_cal import merge_duckdb
merge_duckdb(["parquet/a.parquet","parquet/b.parquet"], "merged/all.parquet")
```

::: cross_sensor_cal
    options:
      members: true
    filters:
      - "!^_"
