# Quickstart

This Quickstart walks you through installing cross-sensor-cal, running a single NEON flight line through the pipeline, and inspecting the outputs. It is designed to give you one successful end-to-end example before diving into tutorials or the conceptual background.

---

## Install

The package is available on PyPI:

```bash
pip install cross-sensor-cal
For development installations or optional Ray support, see the Reference section.
Run the pipeline on one NEON flight line
BASE=output_quickstart && mkdir -p "$BASE"

cscal-pipeline \
  --base-folder "$BASE" \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2 \
  --engine thread
The first time this runs, it will:
fetch the required NEON HDF5 tiles
export ENVI cubes
apply BRDF + topographic correction
create Landsat-equivalent reflectance
write Parquet tables
generate QA outputs
If the command is rerun, completed stages will be skipped automatically.
Inspect the QA outputs
Look in your output folder:
output_quickstart/
    NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/
        *_brdfandtopo_corrected_envi.img
        *_landsat_convolved_envi.img
        *_merged_pixel_extraction.parquet
        *_qa.png
        *_qa.pdf
        *_qa.json
Open the PNG QA panel:
open output_quickstart/..._qa.png
This shows reflectance ranges, masks, brightness shifts, and diagnostic metrics.
More detail is available in the QA metrics page.
What to explore next
Understand why calibration is necessary:
Why cross-sensor calibration?
Follow detailed workflows:
NEON → corrected ENVI
NEON → Landsat-style reflectance
Work with outputs in analysis tools:
Working with Parquet


---
