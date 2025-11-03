# Validation

> **Purpose:** This section explains the QA tests in quantitative terms.

Validation routines in `cross_sensor_cal.qa_plots` compute physical and statistical
diagnostics designed to be *interpretable by scientists*. They check not only numeric stability
but physical plausibility (reflectance bounds, wavelength order, spectral coherence).

For each metric, expected ranges are derived from NEON calibration standards and cross-sensor
comparisons (Landsat OLI, Sentinel-2 MSI, and MicaSense). These ranges were validated empirically
across 2021–2024 flight lines and provide confidence that corrected products maintain
absolute reflectance fidelity within ±2 %.

### ΔReflectance Test
Measures the magnitude of change introduced by correction; large uniform shifts imply over- or under-correction.

### Convolution Accuracy Test
Computes RMSE and SAM between expected and computed spectral bands; high errors indicate sensor response mismatch.

### Brightness Gain/Offset Check
Evaluates brightness normalization stability; flags gain < 0.85 or > 1.15.

### Landsat↔MicaSense brightness adjustment

When the pipeline convolves corrected NEON cubes to Landsat bands, it now applies a
small per-band brightness adjustment so the results align with a MicaSense reference.

- Coefficients live in `data/brightness/landsat_to_micasense.json` and are shipped with the
  package.
- The applied values are written into QA JSON files under `brightness_coefficients` and are
  also rendered on Page 3 of the QA PDF alongside other issues.
- The Landsat ENVI headers record the same coefficients so downstream tools can reproduce
  the exact multiplicative adjustment (`L_adj = L_raw * (1 + coeff / 100)`).

[See detailed interpretation →](../pipeline/qa_panel.md)

> **When do I need this?** When a stage fails or a QA smell appears; validate inputs/outputs against known-good schema.

## Purpose
Provide targeted checks for [Pipeline Stages](../pipeline/stages.md) that frequently fail—especially Stage 3 corrections and Stage 6 merges.

## Inputs
- Paths to ENVI `.img/.hdr` or Parquet files from [Outputs](../pipeline/outputs.md)
- Schema definitions or expected ranges for QA metrics

## Outputs
Console reports or CSV summaries highlighting missing bands, mismatched wavelengths, or schema drift.

## Run it
```bash
python scripts/check_envi_headers.py corrected/*_brdfandtopo_corrected_envi.hdr
python scripts/validate_schema.py merged/demo_merged_pixel_extraction.parquet schemas/merged_schema.json
```

```python
from cross_sensor_cal.validation import validate_parquet

validate_parquet("merged/demo_merged_pixel_extraction.parquet", strict=True)
```

## Pitfalls
- Skipping validation can hide silent failures; automate checks in CI before trusting Stage 7 QA images.
- Keep Ray workers pinned to the same package version to avoid mixed schema outputs.
- When QA histograms look wrong, verify both the input parquet and the ENVI wavelength metadata.
