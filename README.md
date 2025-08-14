[![DOI](https://zenodo.org/badge/647048266.svg)](https://zenodo.org/doi/10.5281/zenodo.11167876)

# Cross-Sensor Calibration

## Overview
Cross-Sensor Calibration is a scriptable Python workflow for processing NEON AOP
hyperspectral flightlines and resampling them to simulate other sensors. You can
convert raw `.h5` data to ENVI, apply BRDF and topographic corrections, extract
pixel spectra, and organize the results in a reproducible directory structure.

## Prerequisites
- Python 3.10+
- `ray`, `numpy`, `rasterio`, `h5py`, `hytools`, `tqdm`, `pandas`
- NEON flightline identifiers and optional polygon layers for masking

Install dependencies:

```bash
pip install -r requirements.txt
```

## Step-by-step tutorial
### 1. Raster processing
1. **Download** NEON flightlines.
2. **Convert** HDF5 to ENVI format.
3. **Generate** BRDF and topographic correction configs.
4. **Apply** corrections.
5. **Resample** to alternate sensor specifications.

```python
from pathlib import Path
from src.envi_download import download_neon_flight_lines
from src.neon_to_envi import neon_to_envi
from src.topo_and_brdf_correction import generate_config_json, topo_and_brdf_correction
from src.convolution_resample import resample
from src.file_types import (
    NEONReflectanceConfigFile,
    NEONReflectanceBRDFCorrectedENVIFile,
)

data_dir = Path("data/NIWO_2023_08")
flight_lines = [
    "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance",
    "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance",
]

# Download
download_neon_flight_lines(
    out_dir=data_dir,
    site_code="NIWO",
    product_code="DP1.30006.001",
    year_month="2023-08",
    flight_lines=flight_lines,
)

# Convert to ENVI
for h5_file in data_dir.rglob("*.h5"):
    neon_to_envi(images=[str(h5_file)], output_dir=str(data_dir), anc=True)

# Generate configs
generate_config_json(data_dir)

# Apply corrections
for cfg in NEONReflectanceConfigFile.find_in_directory(data_dir):
    topo_and_brdf_correction(str(cfg.file_path))

# Resample
for corrected in NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(data_dir):
    resample(corrected.directory)
```

### 2. Pixel extraction
1. **Mask** rasters with polygons (optional).
2. **Extract** pixel spectra into CSV files.
3. **Sort and sync** outputs to remote storage.

```python
from src.extraction import process_base_folder, process_all_subdirectories, sort_and_sync_files

base_folder = "data/NIWO_2023_08"
polygon_layer = "inputs/site_polygons.geojson"

process_base_folder(base_folder, polygon_layer)
process_all_subdirectories(Path(base_folder), polygon_layer)
sort_and_sync_files(base_folder, "i:/iplant/home/shared/earthlab/macrosystems/cross-sensor-cal", True)
```

## Reference
- `src/` – core Python modules
- `bin/` – command-line interfaces
- `data/` – working datasets
- `docs/` – documentation and style guide
- `tests/` – automated test suite

## Next steps
Explore the README in each subdirectory for detailed instructions and extend the
pipeline to support additional sensors or workflows.

Last updated: 2025-08-14
