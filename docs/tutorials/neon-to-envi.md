# Tutorial: NEON → Corrected ENVI (BRDF + Topographic)

This tutorial walks through converting NEON hyperspectral HDF5 files into physically corrected ENVI reflectance cubes. The output is the foundational product used in all downstream harmonization workflows (e.g., Landsat-style reflectance).

---

## Overview

You will learn how to:

1. download NEON directional reflectance tiles  
2. export them to ENVI format  
3. apply topographic correction  
4. apply BRDF correction  
5. inspect corrected outputs and QA artifacts  

This tutorial assumes you have installed cross-sensor-cal and have run through the [Quickstart](../quickstart.md).

---

## 1. Set up a working directory

```bash
BASE=output_neon_to_envi
mkdir -p "$BASE"
2. Run the pipeline for one flight line
Here we process a single NEON hyperspectral flight line at NIWO.
cscal-pipeline \
  --base-folder "$BASE" \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --engine thread \
  --max-workers 2
The command will:
fetch the necessary HDF5 files
export directional reflectance to ENVI (*_envi.img/.hdr)
apply topographic correction
apply BRDF correction
write corrected ENVI reflectance (*_brdfandtopo_corrected_envi.img)
generate QA PNG, PDF, and JSON files
Re-running the command will skip completed stages.
3. What the corrected ENVI product contains
A corrected flight line directory contains files such as:
*_directional_reflectance_envi.img
*_topocorrected_envi.img
*_brdfandtopo_corrected_envi.img
*_qa.png
*_qa.pdf
*_qa.json
The key output is:
*_brdfandtopo_corrected_envi.img
This file contains physically corrected reflectance values suitable for sensor harmonization or modeling.
4. Inspect the corrected ENVI cube
You can view the ENVI product using:
ENVI/IDL
QGIS (with raster bands exposed)
Python libraries such as rioxarray or spectral
Example in Python:
import rioxarray as rxr

cube = rxr.open_rasterio("path/to/..._brdfandtopo_corrected_envi.img")
cube
This loads the corrected reflectance cube into an xarray DataArray.
5. Inspect QA outputs
Open the QA PNG:
open "$BASE/..._qa.png"
The QA panel includes:
reflectance range checks
brightness differences across correction stages
mask summary statistics
wavelength and band metadata
BRDF / topographic coefficient summaries
The PDF version includes multi-page diagnostics and expanded panels.
6. Next steps
Proceed to:
NEON → Landsat reflectance
Pipeline stages for in-depth explanations
Working with Parquet to extract and analyze pixel spectra

---
