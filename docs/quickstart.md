# Quickstart

This Quickstart gives you two ways to run the pipeline end-to-end:

1. **CLI path** – run `cscal-pipeline` from a terminal  
2. **Notebook path** – run the pipeline inside Jupyter

Both produce the same ENVI, Parquet, and QA artifacts.

---

## Install

Install from PyPI:

```bash
pip install cross-sensor-cal
For Ray support:
pip install "cross-sensor-cal[ray]"
1. CLI path
Use the CLI if you run jobs on your laptop or HPC cluster.
Run a NEON flight line
# Choose an output directory
BASE=output_quickstart
mkdir -p "$BASE"

# Run the pipeline
cscal-pipeline \
  --base-folder "$BASE" \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2 \
  --engine thread
The first time this runs, it will:
download NEON HDF5 tiles
export ENVI cubes
apply topographic + BRDF correction
convolve to Landsat style reflectance
write Parquet tables
produce QA PNG, PDF, and JSON summaries
If rerun, completed stages are skipped safely.
Inspect QA files
open $BASE/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/*_qa.png
open $BASE/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/*_qa.pdf
2. Notebook path (Jupyter)
Use this if you want a reproducible, interactive workflow.
Python example
from cross_sensor_cal import go_forth_and_multiply

base = "output_quickstart_py"

go_forth_and_multiply(
    base_folder=base,
    site_code="NIWO",
    year_month="2023-08",
    product_code="DP1.30006.001",
    flight_lines=["NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"],
    max_workers=2,
    engine="thread",
)
Preview merged Parquet
import duckdb, os

fl = "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"
merged = os.path.join(base, fl, f"{fl}_merged_pixel_extraction.parquet")

duckdb.query(f"SELECT * FROM '{merged}' LIMIT 5").df()
For a complete notebook example, see:
Usage → Jupyter notebook example
Next steps
Why calibration?
Tutorials
Pipeline overview
Working with Parquet

---
