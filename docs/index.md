# Cross-Sensor Calibration

Earth Lab’s **cross-sensor-cal** is a Python package for producing physically corrected and sensor-harmonized reflectance from NEON hyperspectral data and other fine-resolution imagery. It provides a reproducible pipeline that:

- converts NEON HDF5 directional reflectance into clean ENVI cubes  
- applies topographic and BRDF corrections  
- harmonizes corrected spectra into Landsat, MicaSense, and other sensor bandspaces  
- produces analysis-ready Parquet tables  
- emits QA panels, PDF reports, and JSON metrics  

The goal is to make it easy and repeatable to move from *raw NEON tiles* to *cross-sensor comparable reflectance* across sites, years, and platforms.

---

## Why this package exists

Ecologists increasingly need to align data from different platforms:

- NEON hyperspectral imaging  
- drone multispectral cameras (e.g., MicaSense)  
- moderate-resolution satellites (e.g., Landsat OLI/OLI-2, Sentinel-2)  

These sensors differ in band definitions, viewing and illumination geometry, radiometric assumptions, and file formats. Without harmonization, differences between sensors can overwhelm differences in vegetation or ecosystem state.

cross-sensor-cal provides a **transparent, auditable, restart-safe** workflow that implements:

1. physical corrections that reduce geometry-driven variation  
2. sensor-bandpass harmonization  
3. consistent, interpretable outputs for analysis or model ingestion  

---

## What problems does cross-sensor-cal solve?

Use this package when you need to:

- compare NEON hyperspectral reflectance to satellite reflectance  
- translate drone-scale multispectral data into Landsat-equivalent bands  
- build cross-scale ecological models (plot → drone → NEON → satellite)  
- process multiple NEON flight lines reproducibly in the cloud  
- generate standardized QA artifacts for calibration decisions  

If you only need to open a single NEON HDF5 file in a viewer, this package is more than you need. If you need **many flight lines** and **scientifically interpretable harmonization**, this pipeline is designed for that workflow.

---

## A minimal example: run one NEON flight line (CLI)

The fastest way to see the pipeline working is via the command-line interface.

```bash
# 1. Choose an output base
BASE=output_demo
mkdir -p "$BASE"

# 2. Run the pipeline on one NEON flight line
cscal-pipeline \
  --base-folder "$BASE" \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2 \
  --engine thread

# 3. Inspect QA outputs (macOS example)
open "$BASE/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance_qa.png"
open "$BASE/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance_qa.pdf"
For a Python/Jupyter version of this example, see the Jupyter notebook example.
How the pipeline works (conceptual view)
HDF5 → ENVI export
Convert directional reflectance into ENVI .img/.hdr with clean metadata.
Topographic + BRDF correction
Use DEM and per-pixel sun/view geometry to create reflectance aligned to a common geometry.
Convolution / harmonization
Integrate corrected spectra into target sensor bandpasses (e.g., Landsat OLI).
Parquet generation & merging
Produce analysis-ready tables with wavelengths, masks, and derived indices.
Quality assurance
Emit a QA panel, multi-page PDF, and JSON metrics for each flight line.
Detailed descriptions are in the Pipeline overview & stages page.
What you get out
Each processed flight line produces:
corrected ENVI cubes
Landsat- and/or MicaSense-equivalent reflectance cubes
per-product Parquet tables
merged pixel extraction tables
PNG QA panel
multi-page QA PDF
JSON validation metrics
The structure is described in Outputs & file structure.
Next steps
GoalWhere to go
Run your first full workflowQuickstart
Understand scientific contextWhy calibration?
Work through examplesTutorials
Use outputs in analysisWorking with Parquet
Try it in a notebooks/Jupyter notebook example
Inspect and validate resultsQA panels & metrics
Tune or extend the pipelineReference
Something looks offTroubleshooting guide

---
