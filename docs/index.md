# Cross-Sensor Calibration

Earth Lab’s **cross-sensor-cal** is a Python package for producing physically corrected and sensor-harmonized reflectance from NEON hyperspectral data and other fine-resolution imagery.

It provides a reproducible workflow that:

- exports NEON HDF5 to ENVI  
- applies topographic and BRDF correction  
- harmonizes reflectance to Landsat / MicaSense bandspaces  
- writes Parquet tables  
- emits QA PNG, PDF, and JSON summaries  

---

## Minimal example (CLI)

```bash
BASE=output_demo
mkdir -p "$BASE"

cscal-pipeline \
  --base-folder "$BASE" \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2 \
  --engine thread
Inspect QA outputs:
open $BASE/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/*_qa.png
For a Python/Jupyter version, see:
Usage → Jupyter notebook example
How the pipeline works
HDF5 → ENVI export
Topographic correction
BRDF correction
Sensor harmonization (bandpass convolution)
Parquet extraction + merging
QA reporting
See the Pipeline overview for details.
Next steps
Quickstart
Jupyter notebook example
Tutorials
Reference

---
