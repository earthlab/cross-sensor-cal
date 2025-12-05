# Troubleshooting Guide

This page lists common problems encountered when running the cross-sensor-cal pipeline, along with likely causes and recommended solutions. It is organized by pipeline stage.

---

## General issues

### The pipeline crashes or silently stops
Likely causes:
- out-of-memory (OOM) errors  
- temporary directory filling up  
- unexpected NEON file structure  

Solutions:
- reduce `--max-workers`  
- set `CSCAL_TMPDIR` to a larger scratch disk  
- run stages independently using `--start-at` and `--end-at`

---

## Download & HDF5 access issues

### Missing or corrupted HDF5 tiles
- NEON occasionally updates directory structures  
- Cloud storage sessions may time out  

Solutions:
- manually verify the HDF5 path  
- re-run pipeline; downloads are restart-safe  
- use a fresh working directory  

---

## HDF5 → ENVI export issues

### ENVI header not recognized or wavelengths missing
Causes:
- malformed or incomplete HDF5 metadata  
- older NEON tiles with inconsistent naming conventions  

Solutions:
- verify the HDF5 product name includes `directional_reflectance`  
- ensure correct product code (usually `DP1.30006.001`)  
- re-run the export stage only:

```bash
cscal-pipeline --start-at export-envi --end-at export-envi ...
ENVI export produces extremely large or slow files
This stage is I/O intensive.
Solutions:
avoid running many exports concurrently
use local SSD scratch storage
use thread engine instead of Ray for single-tile workflows
Topographic correction issues
Dark or clipped areas after topo correction
Causes:
deep shadows
DEM mismatch
slope/aspect irregularities
Solutions:
check DEM resolution
visually inspect slope/aspect rasters
mask problematic areas if needed
BRDF correction issues
BRDF correction produces NaNs or extreme values
Causes:
unstable BRDF coefficient fitting
missing or unrealistic geometry values
low-SNR or noisy bands
Solutions:
inspect BRDF coefficients in QA JSON
limit BRDF correction to certain bands
check for invalid view/solar geometry
Sudden brightness shift after BRDF
Possible causes:
incorrect per-band scaling
extreme solar/view geometry
BRDF coefficients failing to converge
Check the QA PNG or PDF to verify brightness changes.
Convolution (sensor harmonization) issues
Landsat-convolved reflectance looks wrong
Causes:
wavelength mismatch
empty or incorrect SRF tables
bright or dark artifacts from BRDF stage
Solutions:
inspect SRF metadata in QA JSON
confirm NEON wavelengths match expected ranges
compare band means to expected Landsat reflectance ranges
Brightness coefficients are unusually large
This indicates poor alignment between corrected spectra and sensor response functions.
Check:
reflectance scaling
BRDF coefficient stability
QA brightness plots
Parquet extraction issues
Memory errors during extraction or merge
Solutions:
reduce number of workers
increase scratch space
use DuckDB for large-table operations rather than pandas
QA issues
QA PNG or PDF missing
Causes:
pipeline interrupted before QA stage
insufficient permissions in output directory
Re-run:
cscal-pipeline --start-at qa ...
When to reach out for help
If the pipeline produces persistent artifacts, consider:
sharing the QA PNG/PDF
sharing a small snippet of metadata
describing environment, RAM, and tile size
These provide critical clues about where failure occurs.
Next steps
Pipeline stages
QA metrics

---
