# Quickstart

This Quickstart gives you two ways to run the pipeline end-to-end:

1. **CLI path** – use the `cscal-pipeline` command from a terminal.  
2. **Notebook path** – call the pipeline from Python inside Jupyter.

Both paths produce the same corrected ENVI, harmonized products, Parquet tables, and QA artifacts.

---

## 1. CLI path

Use this if you are comfortable with a terminal or running batch jobs.

```bash
# Choose an output base
BASE=output_quickstart
mkdir -p "$BASE"

# Run the pipeline on one NIWO flight line
cscal-pipeline \
  --base-folder "$BASE" \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2 \
  --engine thread
What this does:
Downloads the NEON HDF5 tiles (if needed)
Exports ENVI cubes
Applies topographic + BRDF correction
Convolves to Landsat bandspace (if configured)
Writes Parquet tables
Generates QA PNG, PDF, and JSON
Re-running the same command is safe: completed stages are skipped.
2. Notebook path (Python / Jupyter)
Use this if you want to orchestrate runs from Python and immediately inspect outputs in a notebook.
First, install the package into the environment where you run Jupyter:

pip install cross-sensor-cal
Then, in a notebook cell:
from cross_sensor_cal import go_forth_and_multiply

base_folder = "output_quickstart_py"
site_code = "NIWO"
year_month = "2023-08"
product_code = "DP1.30006.001"
flight_lines = ["NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"]

go_forth_and_multiply(
    base_folder=base_folder,
    site_code=site_code,
    year_month=year_month,
    product_code=product_code,
    flight_lines=flight_lines,
    max_workers=2,
    engine="thread",
)
After this cell finishes, you will have the same ENVI, Parquet, and QA artifacts as in the CLI example, but orchestrated from Python.
To explore the merged Parquet in the same notebook:

import os
import duckdb
import pandas as pd

fl_prefix = "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"
fl_dir = os.path.join(base_folder, fl_prefix)

merged = os.path.join(
    fl_dir,
    f"{fl_prefix}_merged_pixel_extraction.parquet",
)

con = duckdb.connect()
df = con.execute(
    f"SELECT * FROM read_parquet('{merged}') LIMIT 5"
).df()
df
For a more complete notebook-style walkthrough, see the Jupyter notebook example.
Tips and common pitfalls
Start with --max-workers 2 or 4 unless you are sure you have plenty of RAM.
If /dev/shm is small (common in containers), avoid very large concurrency.
Use --engine thread first; switch to --engine ray only after cross-sensor-cal[ray] is installed and working.
Where to go next
Understand each pipeline stage: Pipeline overview & stages
Inspect outputs: Outputs & file structure
Work with Parquet tables: Working with Parquet

---
