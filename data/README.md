# Data

## Overview
The `data` directory holds working datasets produced or consumed by the
pipeline. You can organize NEON downloads, intermediate ENVI files, and final
CSV exports here using the directory structure shown in the project README.

## Prerequisites
- Adequate disk space for hyperspectral imagery
- Optional: access credentials for NEON or remote storage endpoints

## Step-by-step tutorial
1. Create a new project directory inside `data/` named `SITE_YYYY_MM`.
2. Download NEON flightlines into `raw_h5/`:

```python
from src.envi_download import download_neon_flight_lines

download_neon_flight_lines(
    out_dir="data/NIWO_2023_08/raw_h5",
    site_code="NIWO",
    product_code="DP1.30006.001",
    year_month="2023-08",
    flight_lines=["NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance"],
)
```

3. Run the processing steps to populate `envi/`, `corrected/`, `resampled/`,
   and `csv/` subfolders.

## Reference
- `aop_macrosystems_data_1_7_25.geojson` – example polygon layer
- `hyperspectral_bands.json` – sensor band definitions
- `landsat_band_parameters.json` – Landsat resampling parameters

## Next steps
Clean up intermediate files when they are no longer needed to conserve disk
space.

Last updated: 2025-08-14
