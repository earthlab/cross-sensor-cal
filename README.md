[![DOI](https://zenodo.org/badge/647048266.svg)](https://zenodo.org/doi/10.5281/zenodo.11167876)

# cross-sensor-cal
Python tool for cross-sensor calibration (This tool development is part of the NSF Award #DEB 2017889)

![Macrosystems Disturbance Resilience - Sensor-convolution (4)](https://github.com/earthlab/cross-sensor-cal/assets/67020853/90b08cf3-b9ca-494e-80a0-32dccadaefd4)

# NEON Hyperspectral Processing Pipeline

This package provides a modular, scriptable Python workflow for processing NEON AOP hyperspectral flightline data and extracting pixel-level reflectance values. The pipeline is divided into two stages:

- **Step 1: Raster Processing** — converts `.h5` files to ENVI format, applies topographic and BRDF corrections, and resamples to multiple sensor specifications.
- **Step 2: Pixel Extraction** — extracts spectral data from corrected rasters into tabular CSVs, optionally filtered by polygons.

Each step is automated but customizable, enabling users to scale from a single flightline to large collections across sensors and sites.

---

## 🛰 Step 1: Raster Processing

This step performs the following operations:

1. **Download** NEON `.h5` hyperspectral reflectance files for selected flightlines  
2. **Convert** to ENVI format with optional ancillary data  
3. **Generate** correction configs (BRDF, topography)  
4. **Apply** topographic and BRDF corrections using HyTools  
5. **Resample** hyperspectral reflectance to simulate other sensors (e.g., Landsat, MicaSense)  

Outputs are written to a structured folder hierarchy for reuse in extraction workflows.

### Quickstart Example

<pre><code>from pathlib import Path
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
</code></pre>

---

## 📊 Step 2: Pixel Extraction

This step extracts tabular reflectance data from the corrected ENVI files, optionally masking by user-defined vector geometries.

1. **Mask** raster data with a polygon layer (optional)  
2. **Extract** pixel values from each flightline and sensor variant  
3. **Process** entire directory trees automatically  
4. **Sort and sync** output tables to remote locations or cloud storage  

### Example Usage

<pre><code>from pathlib import Path
from src.extraction import (
    process_base_folder,
    process_all_subdirectories,
    sort_and_sync_files
)

# Define input parameters
base_folder = "data/NIWO_2023_08"
polygon_layer_path = "inputs/site_polygons.geojson"
remote_prefix = "i:/iplant/home/shared/earthlab/macrosystems/cross-sensor-cal"
sync_files = True

# Step 1: Mask and extract from the base folder
process_base_folder(
    base_folder=base_folder,
    polygon_layer=polygon_layer_path,
    raster_crs_override="EPSG:4326",       # Optional override for raster CRS
    polygons_crs_override="EPSG:4326",     # Optional override for polygon CRS
    output_masked_suffix="_masked",        # Optional suffix for masked outputs
    plot_output=False,                     # Skip diagnostic plots
    dpi=300                                # Plot resolution (if enabled)
)

# Step 2: Recursively process all subdirectories
process_all_subdirectories(Path(base_folder), polygon_layer_path)

# Step 3: Sort and optionally sync files to remote storage
sort_and_sync_files(base_folder, remote_prefix, sync_files)
</code></pre>

---

## 📁 Output Organization

Each step of the pipeline writes outputs to nested folders, with consistent naming for easy traceability:

data/
└── SITE_DATE/
    ├── raw_h5/                  # Raw .h5 NEON flightlines
    ├── envi/                    # ENVI-converted hyperspectral data
    ├── ancillary/               # Ancillary ENVI files (e.g., slope, path length)
    ├── corrected/               # BRDF and topographically corrected ENVI
    ├── resampled/               # Sensor-convolved versions (e.g., Landsat, MicaSense)
    ├── csv/                     # Pixel-level reflectance tables
    └── config/                  # JSON files for correction parameters


---

## 🧩 Extensibility

- JSON configs can be modified to simulate additional sensors  
- Polygon masks can define specific regions for extraction  
- Custom pipelines can extend or remix steps for new use cases  

---

## 🔗 Requirements

- Python 3.10+
- `ray`, `numpy`, `rasterio`, `h5py`, `hytools`, `tqdm`, `pandas`

Install dependencies:

pip install -r requirements.txt


## 📬 Contact

For questions or contributions, open an issue or submit a pull request on GitHub.

