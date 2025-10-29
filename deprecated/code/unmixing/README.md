# Unmixing

## Overview
The `unmixing` directory provides Python tools and demonstrations for spectral
unmixing. You can download Landsat data, run endmember extraction, and compare
results with the accompanying R notebook.

## Prerequisites
- Python 3.10+
- Libraries: `rasterio`, `numpy`, `pandas`
- Optional: R for the `unmixing.Rmd` example

## Step-by-step tutorial
1. Download Landsat imagery for a target region:

```bash
python unmixing/download_landsat.py --path 34 --row 32 --year 2023
```

2. Execute the MESMA implementation:

```python
from unmixing.el_mesma import run_mesma

run_mesma("/path/to/landsat.tif", endmembers=[...])
```

## Reference
- `download_landsat.py` – fetches Landsat scenes
- `el_mesma.py` – Implements Multiple Endmember Spectral Mixture Analysis
- `unmixing.py` – helper functions
- `unmixing.Rmd` – R counterpart to the Python utilities

## Next steps
Integrate unmixing outputs with the main calibration pipeline to compare sensor
responses.

Last updated: 2025-08-14
