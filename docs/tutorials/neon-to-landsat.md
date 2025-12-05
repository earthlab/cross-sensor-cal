# Tutorial: NEON → Landsat-Style Reflectance

This tutorial shows how to convert physically corrected NEON ENVI reflectance into **Landsat-equivalent reflectance** using sensor spectral response functions (SRFs).

The output is a set of ENVI cubes and Parquet tables representing NEON reflectance expressed in the Landsat bandspace.

---

## Prerequisites

Before running this tutorial, complete:

- [NEON → corrected ENVI](neon-to-envi.md)

You will need the `*_brdfandtopo_corrected_envi.img` output for convolution.

---

## 1. Select a corrected ENVI cube

Your corrected data should look like:

.../NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/
NEON_D13_NIWO_..._brdfandtopo_corrected_envi.img

This file is the input to Landsat bandpass convolution.

---

## 2. Run the convolution stage

If you used the main pipeline, convolution runs automatically after BRDF+topo correction.  
To run only the convolution stage manually:

```bash
cscal-pipeline \
  --base-folder "$BASE" \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2 \
  --engine thread \
  --start-at convolution
Outputs include:
*_landsat_convolved_envi.img
*_landsat_convolved_envi.hdr
*_landsat_convolved.parquet
3. Understanding Landsat bandpass integration
The convolution stage integrates each corrected spectrum against Landsat OLI SRFs:
Coastal / Blue / Green / Red
NIR
SWIR1
SWIR2
This process produces reflectance values aligned to Landsat definitions, enabling:
direct comparison to satellite imagery
validation exercises
cross-scale modeling
harmonized NDVI and other vegetation indices
4. Inspecting Landsat-like ENVI outputs
Example in Python:
import rioxarray as rxr

landsat = rxr.open_rasterio("path/to/..._landsat_convolved_envi.img")
landsat
Band order, wavelengths, and metadata will match the Landsat OLI sensor.
5. Checking the QA metrics
The QA JSON and QA PNG will now include:
brightness adjustments used in Landsat harmonization
bandwise statistics after convolution
wavelength alignment checks
mask and no-data diagnostics
Use this to verify successful harmonization.
6. Using the Parquet tables
The *_landsat_convolved.parquet file contains:
one row per pixel
columns for each Landsat-equivalent band
pixel geometry and mask indicators
Example:
import duckdb

duckdb.query("""
    SELECT Red, NIR, (NIR - Red) / (NIR + Red) AS ndvi
    FROM '..._landsat_convolved.parquet'
    LIMIT 5
""").df()
7. Next steps
Continue to:
MicaSense → Landsat harmonization
Working with Parquet
Pipeline QA

---
