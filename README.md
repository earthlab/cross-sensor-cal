[![DOI](https://zenodo.org/badge/647048266.svg)](https://zenodo.org/doi/10.5281/zenodo.11167876)

# cross-sensor-cal
Python tool for cross-sensor calibration (This tool development is part of the NSF Award #DEB 2017889)

![Macrosystems Disturbance Resilience - Sensor-convolution (4)](https://github.com/earthlab/cross-sensor-cal/assets/67020853/90b08cf3-b9ca-494e-80a0-32dccadaefd4)


# NEON Hyperspectral Raster Processing Pipeline

This package provides a modular Python workflow for processing NEON AOP hyperspectral flightline data. It automates key preprocessing steps: converting `.h5` reflectance files to ENVI format, applying topographic and BRDF corrections, and resampling the hyperspectral data to match other sensors (e.g., Landsat, MicaSense).  

The workflow is fully scriptable, scalable across datasets, and customizable via JSON configuration.  

---

## Overview of Step 1: Raster Processing

This module performs the following steps:

1. **Download** NEON hyperspectral `.h5` files for selected flightlines
2. **Convert** HDF5 files to ENVI format (optionally including ancillary layers)
3. **Apply** BRDF and topographic corrections using HyTools
4. **Resample** hyperspectral data to match target sensors via spectral convolution or band sampling
5. **Organize** outputs into a consistent folder structure for downstream use

Users may override the default resampling behavior using editable JSON files, or run the pipeline with default settings for common sensors.

---

## Quickstart Example

```python
from pathlib import Path
from src.envi_download import download_neon_flight_lines
from src.neon_to_envi import neon_to_envi
from src.topo_and_brdf_correction import generate_config_json, topo_and_brdf_correction
from src.file_types import NEONReflectanceConfigFile, NEONReflectanceBRDFCorrectedENVIFile
from src.convolution_resample import resample

# Define working directory and flightline info
data_dir = Path("data/NIWO_2023_08")
site_code = "NIWO"
product_code = "DP1.30006.001"
year_month = "2023-08"
flight_lines = [
    "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance",
    "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"
]

# Step 1: Download NEON flight lines
download_neon_flight_lines(
    out_dir=data_dir,
    site_code=site_code,
    product_code=product_code,
    year_month=year_month,
    flight_lines=flight_lines
)

# Step 2: Convert HDF5 to ENVI
h5_files = list(data_dir.rglob("*.h5"))
for h5_file in h5_files:
    neon_to_envi(
        images=[str(h5_file)],
        output_dir=str(data_dir),
        anc=True
    )

# Step 3: Generate BRDF/topo config files
generate_config_json(data_dir)

# Step 4: Apply BRDF and topographic corrections
config_files = NEONReflectanceConfigFile.find_in_directory(data_dir)
for config_file in config_files:
    topo_and_brdf_correction(str(config_file.file_path))

# Step 5: Resample to sensor specifications
corrected_files = NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(data_dir)
for corrected_file in corrected_files:
    resample(corrected_file.directory)

