# Refactor Notes: HyTools-Free Pipeline

## Purpose
These notes document the current cross-sensor-cal processing pipeline after removing the runtime dependency on HyTools. The steps below describe what the code does today so collaborators can reproduce results and audit intermediate products.

## End-to-End Pipeline
1. **Acquire NEON reflectance flightlines**  
   Download or copy the required NEON Airborne Observation Platform (AOP) reflectance HDF5 files to a local workspace before running the pipeline.

2. **Convert HDF5 to ENVI without HyTools**  
   `neon_to_envi_no_hytools()` opens the HDF5 file with `NeonCube`, streams the cube out in spatial tiles, and writes a float32 BSQ ENVI dataset via `EnviWriter`. It simultaneously exports ancillary rasters (solar/sensor geometry, slope, aspect, etc.) needed for correction. The result is an uncorrected directional reflectance `.img/.hdr` pair for each flightline. HyTools and Ray are not invoked in this stage—the conversion logic is entirely internal.

3. **Persist correction parameters**
   `build_and_write_correction_json()` (in `brdf_topo.py`) inspects the flightline geometry, fits BRDF coefficients, and serialises the results as `<flightline>_brdfandtopo_corrected_envi.json`. The helper validates the JSON via `is_valid_json()` and reuses it on reruns when intact.

4. **Topographic and BRDF correction**
   The pipeline allocates a new corrected cube and uses `EnviWriter` to persist it. For every spatial tile it:
   - Reads the tile from the uncorrected ENVI export.
   - Applies topographic correction using slope, aspect, and solar geometry rasters.
   - Applies BRDF correction using the saved coefficient JSON. When coefficients are missing, unreadable, or poorly conditioned, the code logs a warning and falls back to neutral BRDF terms so the tile still receives topographic correction.
   - Optionally adds a `brightness_offset` before writing.  
   The corrected output `<flightline>_brdfandtopo_corrected_envi.img/.hdr` carries full spatial metadata plus the wavelength list, FWHM list, and wavelength units required for spectral resampling.

5. **Spectral convolution / sensor simulation**
   `convolve_resample_product()` opens the corrected cube as a BSQ memmap, reads spatial tiles, transposes them to `(y, x, bands)`, and multiplies each tile by sensor-specific spectral response functions (SRFs). SRFs are loaded from JSON files under `cross_sensor_cal/data/` via package-relative paths. Each simulated sensor produces its own float32 BSQ ENVI product and header. Existing resampled outputs are validated with `is_valid_envi_pair()` and skipped when already complete.

6. **Downstream consumers (optional)**
   Additional tooling can derive pixel stacks, polygon summaries, or parquet tables from the corrected and resampled rasters. These consumers still function but are documented separately and are not detailed here.

Every step performs the same validation checks on reruns so the pipeline is safe to resume after interruptions or partial failures.

7. **Recommended artifact retention**  
   Keep the following per flightline so downstream analyses and cross-sensor comparisons remain reproducible:
   - `<flightline>_directional_reflectance.img/.hdr`
   - `<flightline>_brdfandtopo_corrected_envi.json`
   - `<flightline>_brdfandtopo_corrected_envi.img/.hdr`
   - `<flightline>_resampled_<sensor>.img/.hdr`

## Module Structure
- `cross_sensor_cal/neon_cube.py`
  - `NeonCube` class
  - Opens NEON HDF5 reflectance, exposes dimensions, wavelengths, ancillary angles, etc.
  - Iterates spatial tiles without requiring HyTools.
- `cross_sensor_cal/envi_writer.py`
  - `EnviWriter` class
  - Writes BSQ float32 rasters (`.img/.hdr`).
  - Used for uncorrected export, corrected cubes, and resampled products.
- `cross_sensor_cal/corrections.py`
  - `fit_and_save_brdf_model()`
  - `apply_topo_correct()`
  - `apply_brdf_correct()`
  - Includes helpers to load and apply BRDF coefficients.
- `cross_sensor_cal/resample.py`
  - `resample_chunk_to_sensor()`
  - SRF loading utilities
  - Convolution-friendly helpers for chunk-wise processing.
- `cross_sensor_cal/pipelines/pipeline.py`
  - `go_forth_and_multiply()`
  - Orchestrates downloads, H5→ENVI export (no HyTools), BRDF fitting, topographic+BRDF correction, and spectral convolution.
- `cross_sensor_cal/data/`
  - SRF JSON files for Landsat, Sentinel, etc.
  - Accessed via package-relative paths at runtime.

## Licensing
Several algorithms and data-handling conventions in `NeonCube`, the correction routines, and ENVI export logic were adapted from the HyTools project (GPLv3). Although the refactored pipeline no longer imports HyTools at runtime, we continue to credit the original HyTools authors and comply with GPLv3 obligations for the adapted code.
