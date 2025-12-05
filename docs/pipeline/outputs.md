# Outputs & File Structure

The cross-sensor-cal pipeline produces a consistent directory structure for each flight line. This page describes every artifact, what it contains, and how to use it.

---

## Flight line directory layout

A typical directory looks like:

NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/
directional/
topo/
brdf/
convolved/
parquet/
qa/

Within each folder, ENVI images, Parquet tables, and sidecar JSON files share common prefixes.

---

## Key outputs

### 1. Corrected ENVI products

**Files:**

*_directional_reflectance_envi.img
*_topocorrected_envi.img
*_brdfandtopo_corrected_envi.img

**Contents:**

- reflectance bands  
- wavelength metadata  
- masks (cloud, shadow, water, snow, invalid)  
- CRS and pixel geometry  

---

### 2. Sensor-harmonized ENVI cubes

**Files:**

*_landsat_convolved_envi.img

One file is produced for each requested sensor type.

**Contents:**

- band-averaged reflectance values for the sensor  
- metadata documenting SRFs used  
- brightness adjustment coefficients (if applicable)  

---

### 3. Per-product Parquet tables

Every ENVI file has a corresponding Parquet table:

*_brdfandtopo_corrected.parquet
*_landsat_convolved.parquet

Each row = one pixel.  
Columns include:

- reflectance values  
- masks  
- wavelengths  
- pixel coordinates  

These tables are ideal for large-scale analysis using DuckDB, pandas, or xarray.

---

### 4. Merged pixel extraction table

*_merged_pixel_extraction.parquet

This contains all extracted pixel-level data for the flight line, merged across products.

---

### 5. QA artifacts

*_qa.png
*_qa.pdf
*_qa.json

See the [QA page](qa.md) for details.

---

## Naming conventions

Files follow a consistent pattern:

<NEON ID><date><stage><sensor?><format>.{img|hdr|parquet}

Examples:

- `NEON_D13_NIWO_DP1_L020-1_20230815_brdfandtopo_corrected_envi.img`  
- `NEON_D13_NIWO_DP1_L020-1_20230815_landsat_convolved_parquet`  

The prefixes and suffixes are designed for predictable sorting and automation.

---

## Using outputs in analysis

Example with DuckDB:

```python
import duckdb
duckdb.query("SELECT NIR, Red FROM '..._landsat_convolved.parquet' LIMIT 10").df()
Example with rioxarray:
import rioxarray as rxr
cube = rxr.open_rasterio("..._brdfandtopo_corrected_envi.img")
Next steps
Pipeline stages
QA panels & metrics

---
