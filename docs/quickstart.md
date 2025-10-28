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

The `jefe.py` script orchestrates downloading, converting, correcting, and resampling in the new four-stage order. The following command processes a single flight line and skips remote syncing.

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

The script now reports idempotent skips such as:

```
✅ ENVI export already complete for NEON_D13_NIWO_DP1_L019-1..., skipping
✅ Correction JSON already complete for NEON_D13_NIWO_DP1_L019-1..., skipping
✅ BRDF+topo correction already complete for NEON_D13_NIWO_DP1_L019-1..., skipping
✅ Landsat 8 OLI convolution already complete, skipping
```

These messages confirm that outputs passed validation and the stage moved on without recomputing.

---

### 4. Python API Equivalent

The same workflow can be scripted in Python for additional customization while preserving idempotent behaviour.

```python
from pathlib import Path

from cross_sensor_cal.pipelines.pipeline import go_forth_and_multiply

base = Path("data/NIWO_2023-08")
base.mkdir(parents=True, exist_ok=True)

go_forth_and_multiply(
    base_folder=base,
    site_code="NIWO",
    year_month="2023-08",
    flight_lines=[
        "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance",
    ],
)

# Rerunning the exact command is safe; completed stages log "skipping" and are validated automatically.
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
    │   ├── NEON_D13_NIWO_DP1_L019-1_20230815_brdfandtopo_corrected_envi.img
    │   ├── NEON_D13_NIWO_DP1_L019-1_20230815_brdfandtopo_corrected_envi.hdr
    │   ├── NEON_D13_NIWO_DP1_L019-1_20230815_brdfandtopo_corrected_envi.json
    │   ├── Convolution_Reflectance_Resample_Landsat_8_OLI/
    │   │   └── NEON_D13_NIWO_DP1_L019-1_20230815_resampled_Landsat_8_OLI.img/.hdr
    ├── envi_file_move_list.csv
    └── logs/
```

The actual directory names may differ slightly depending on optional steps (e.g., polygon masking or resampling to additional sensors).

---

### 6. Minimal Validation Checklist

- `_brdfandtopo_corrected_envi.img/.hdr/.json` exist for each processed flightline.
- Resampled products appear under the corresponding `Convolution_Reflectance_Resample_*` directories.
- `envi_file_move_list.csv` lists every generated raster.

---

All paths are relative to the repository root. Replace the placeholder flight line and site details as needed for other datasets.
<!-- FILLME:END -->
