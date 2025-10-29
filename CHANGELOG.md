## [2.2.0] – 2025-10-29
### Added
- Automatic NEON HDF5 download (`stage_download_h5`) with live progress bar.
- Per-flightline subdirectories containing all derived products (ENVI, corrected, convolved, parquet).
- QA summary panel generator (`cscal-qa`) that validates export, correction, convolution, and parquet output.
- Parallel flightline processing with `--max-workers` / `max_workers`.
- Progress bars for download, ENVI export tiling, and BRDF+topo correction tiling.

### Changed
- Pipeline now restores the documented behavior: `cscal-pipeline` / `go_forth_and_multiply()` handles download, correction, resampling, and export in one call.
- Logging now includes per-flightline prefixes during parallel execution for readability.
- Reproducibility documentation and CLI quickstart updated.

### Improved
- Idempotent skip logic preserved across all new stages.
- Organized output layout for long-term storage (keep corrected outputs, discard raw `.h5` if desired).
- Clearer environment setup instructions (conda or pip).

### Fixed
- Eliminated `GRGRGRGR...` spam; replaced with tqdm-style progress bars.
- Made output paths consistent between stages so downstream steps don't guess filenames.

## [Unreleased] – Pipeline refactor for idempotent, ordered execution (October 2025)

- Pipeline is now restart-safe / idempotent: each major stage checks for valid existing outputs
  and skips heavy recompute, logging `✅ ... (skipping)`.
- Introduced canonical output naming via `get_flightline_products()`.
- All per-sensor convolution products are now written as ENVI `.img/.hdr` pairs using the
  pattern `<flight_stem>_<sensor_name>_envi.img/.hdr`.
- Removed `.tif` GeoTIFF outputs from the advertised workflow.
- Convolution now ALWAYS reads the BRDF+topo corrected ENVI cube, never the raw NEON `.h5` directly.
- Added per-sensor success/skip/fail accounting and a final summary line:
  `📊 Sensor convolution summary ... | succeeded=[...] skipped=[...] failed=[...]`
- Pipeline no longer hard-stops if one sensor fails; it finishes the flight line as long as at
  least one sensor succeeded (or had a valid preexisting output).
- Added final site-level completion log: `✅ All requested flightlines processed.`
