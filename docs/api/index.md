# Python API

> Use these functions when you need fine-grained control beyond the CLI.

## Quick example
```python
from spectralbridge import merge_duckdb
merge_duckdb(["parquet/a.parquet","parquet/b.parquet"], "merged/all.parquet")
```

## Brightness correction entry point

### `apply_brightness_correction(cube, mask=None, method='percentile_match', ...)`
Normalizes per-band brightness for hyperspectral cubes before BRDF/topo stages.
The docstring walks through the affine model, parameter choices, and examples.
Use it when you need to harmonise tiles prior to the full pipeline; the QA JSON
will surface the per-band gain/offsets when this stage runs.

::: spectralbridge
    options:
      members: true
    filters:
      - "!^_"
