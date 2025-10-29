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

Create a dedicated base folder for this run. The pipeline will download each NEON `.h5` into this directory automatically and write derived products into per-flightline subfolders.

```bash
BASE=data/NIWO_2023-08
mkdir -p "$BASE"
```

Optional: If you already have an HDF5 flight line, copy it into `"$BASE"`. Otherwise `stage_download_h5()` will stream it with a progress bar the first time you run the pipeline.

---

### 3. Run the CLI Pipeline

The `jefe.py` script orchestrates downloading, converting, correcting, and resampling in the refreshed five-stage order. The following command processes a single flight line and skips remote syncing.

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

The script now shows per-flightline log prefixes and tqdm progress bars instead of the old `GRGRGR...` spam. Reruns surface
idempotent skips such as:

```
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] 📥 stage_download_h5() found existing .h5 (skipping)
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] ✅ ENVI export already complete -> ..._envi.img / ..._envi.hdr (skipping heavy export)
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] ✅ BRDF+topo correction already complete -> ..._brdfandtopo_corrected_envi.img / ..._brdfandtopo_corrected_envi.hdr (skipping)
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] 📊 Sensor convolution summary | succeeded=['landsat_tm', 'micasense'] skipped=['landsat_oli'] failed=[]
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
    product_code="DP1.30006.001",
    flight_lines=[
        "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance",
    ],
    max_workers=2,
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
    │   ├── NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.img/.hdr/.parquet
    │   ├── NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_brdfandtopo_corrected_envi.img/.hdr/.json/.parquet
    │   ├── NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_tm_envi.img/.hdr/.parquet
    │   ├── NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_micasense_envi.img/.hdr/.parquet
    │   └── NIWO_brdf_model.json
    └── <other flightlines follow the same pattern>
```

The actual directory names may differ slightly depending on optional steps (e.g., polygon masking or resampling to additional sensors).

---

### 6. Minimal Validation Checklist

- `<flight_stem>.h5` exists at the base folder root for each requested line.
- `<base>/<flight_stem>/` contains ENVI, JSON, and Parquet products for that line.
- Logs include `[flight_stem]` prefixes with `✅ ... (skipping)` when rerunning on completed stages.

---

All paths are relative to the repository root. Replace the placeholder flight line and site details as needed for other datasets.
<!-- FILLME:END -->
