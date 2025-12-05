# Tutorial: MicaSense → Landsat Harmonization

This tutorial demonstrates how to harmonize drone-scale multispectral reflectance (e.g., MicaSense RedEdge) into Landsat-equivalent band values using the cross-sensor-cal regression workflow.

---

## Overview

You will learn how to:

1. prepare MicaSense reflectance inputs  
2. apply band mapping and wavelength matching  
3. use regression tables to harmonize reflectance  
4. inspect harmonized Landsat-style products  

This allows direct comparison between drone and satellite observations.

---

## 1. Inputs

You need reflectance values from a calibrated drone multispectral system. These can be:

- stacked reflectance TIFFs  
- ENVI-formatted band images  
- Parquet tables exported from your workflow  

Each band should have known center wavelengths.

---

## 2. Harmonization workflow

cross-sensor-cal uses regression relationships linking MicaSense band values to Landsat OLI bands. These regressions are derived from calibrated field and NEON comparisons.

Run the following command:

```bash
cscal-micasense-to-landsat \
  --input your_micasense_input \
  --output ms_to_ls_output \
  --regression-table data/regression/micasense_to_landsat.csv
Outputs include:
Landsat-equivalent reflectance table
diagnostic statistics
optional ENVI export
3. Inspecting harmonized reflectance
Example:
import pandas as pd

df = pd.read_parquet("ms_to_ls_output/micasense_landsat_harmonized.parquet")
df.head()
Columns will match Landsat bands:
Blue
Green
Red
NIR
SWIR1
SWIR2
4. NDVI sanity check
df["ndvi"] = (df["NIR"] - df["Red"]) / (df["NIR"] + df["Red"])
df["ndvi"].describe()
If harmonization worked correctly, NDVI should fall within typical vegetation ranges (0.1–0.9 for most cases).
5. Next steps
Integrate drone → NEON → Landsat comparisons
Combine harmonized products with NEON-derived spectra
Use the merged output in ecological modeling workflows
See also:
NEON → Landsat tutorial
Pipeline outputs

---
