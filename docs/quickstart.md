# Quickstart

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
## Small Dataset Quickstart

This guide walks through processing a single NEON Airborne Observation Platform (AOP) hyperspectral flight line using the Cross‑Sensor Calibration workflow. It assumes only one flight line from the NIWO site in August 2023, which makes it a "small dataset" suitable for experimentation or learning.

The workflow below demonstrates both the command‑line interface (CLI) and the equivalent Python API. All commands are copy‑paste ready and use the placeholder flight line:

```
NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance
```

---

### 1. Environment Setup

1. Clone the repository and create a virtual environment.
2. Activate the environment.
3. Install the package in editable mode.

```bash
git clone https://github.com/example/cross-sensor-cal.git
cd cross-sensor-cal
python -m venv .venv
source .venv/bin/activate          # On Windows use: .venv\Scripts\activate
pip install -e .
```

---

### 2. Prepare the Data Directory

Create a dedicated base folder for this run. The pipeline will download all intermediate files into this directory.

```bash
BASE=data/NIWO_2023-08
mkdir -p "$BASE"
```

Optional: If you already have an HDF5 flight line, copy it into `"$BASE"`. Otherwise the download step will populate the directory.

---

### 3. Run the CLI Pipeline

The `jefe.py` script orchestrates downloading, converting, correcting, and resampling. The following command processes a single flight line and skips remote syncing.

```bash
FLIGHT_LINE=NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance

python bin/jefe.py \
    "$BASE" \
    NIWO \
    2023-08 \
    "$FLIGHT_LINE" \
    --no-sync
```

Key options:

- `"$BASE"` – output directory for all generated files.
- `NIWO` – NEON site code.
- `2023-08` – year and month of the flight.
- `--no-sync` – generate results without uploading to iRODS.

The script emits progress messages for each stage: download, HDF5→ENVI conversion, BRDF/topographic correction, and sensor resampling.

---

### 4. Python API Equivalent

The same workflow can be scripted in Python for additional customization.

```python
from pathlib import Path

from cross_sensor_cal.convolution_resample import resample
from cross_sensor_cal.envi_download import download_neon_flight_lines
from cross_sensor_cal.file_types import (
    NEONReflectanceConfigFile,
    NEONReflectanceBRDFCorrectedENVIFile,
)
from cross_sensor_cal.neon_to_envi import flight_lines_to_envi
from cross_sensor_cal.topo_and_brdf_correction import (
    generate_config_json,
    topo_and_brdf_correction,
)

base = Path("data/NIWO_2023-08")
base.mkdir(parents=True, exist_ok=True)

flight_lines = [
    "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance",
]

download_neon_flight_lines(
    out_dir=base,
    site_code="NIWO",
    product_code="DP1.30006.001",
    year_month="2023-08",
    flight_lines=flight_lines,
)

flight_lines_to_envi(
    input_dir=base,
    output_dir=base,
)

generate_config_json(base)

for cfg in NEONReflectanceConfigFile.find_in_directory(base, "envi"):
    topo_and_brdf_correction(cfg.file_path)

for hdr in NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(base, "envi"):
    resample(hdr.directory)
```

---

### 5. Expected Folder Structure

Before running the pipeline:

```text
data/
└── NIWO_2023-08/
```

After successful completion:

```text
data/
└── NIWO_2023-08/
    ├── NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance.h5
    ├── NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance/
    │   ├── brdf_corrected/
    │   ├── convolution/
    │   └── diagnostic_plots/
    ├── envi_file_move_list.csv
    └── logs/
```

The actual directory names may differ slightly depending on optional steps (e.g., polygon masking or resampling to additional sensors).

---

### 6. Minimal Validation Checklist

- BRDF‑corrected `.hdr` files exist in `brdf_corrected/`.
- Resampled products appear under `convolution/`.
- `envi_file_move_list.csv` lists every generated raster.

---

All paths are relative to the repository root. Replace the placeholder flight line and site details as needed for other datasets.
<!-- FILLME:END -->
