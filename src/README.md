# Source Code

## Overview
The `src` directory contains the Python modules that implement the cross-sensor
calibration workflow. Each module handles a specific stage, from downloading
NEON data to resampling spectra for other sensors.

## Prerequisites
- Python 3.10+
- Dependencies listed in `requirements.txt`

## Step-by-step tutorial
1. Download a NEON flightline:

```python
from src.envi_download import download_neon_flight_lines

download_neon_flight_lines(
    out_dir="data/NIWO_2023_08",
    site_code="NIWO",
    product_code="DP1.30006.001",
    year_month="2023-08",
    flight_lines=["NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance"],
)
```

2. Convert, correct, and resample using subsequent modules:

```python
from src.neon_to_envi import neon_to_envi
from src.topo_and_brdf_correction import topo_and_brdf_correction
from src.convolution_resample import resample
```

## Reference
- `envi_download.py` – data acquisition utilities
- `neon_to_envi.py` – HDF5 to ENVI conversion
- `topo_and_brdf_correction.py` – BRDF and topographic correction
- `convolution_resample.py` – spectral resampling
- `extraction.py` – pixel extraction and tabular export

## Next steps
Explore the module docstrings for detailed parameter descriptions and extend
the workflow with custom code.

Last updated: 2025-08-14
