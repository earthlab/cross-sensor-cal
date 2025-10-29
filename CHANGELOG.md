## [2.2.0] – 2025-10-29

### Added
- Automatic download stage (`stage_download_h5`) with live progress bar.
- Per-tile progress bars for ENVI export and BRDF+topo correction.
- Parallel flightline processing with configurable concurrency (`max_workers`).

### Changed
- Output layout: all derived files for each flightline are now written to `<base>/<flight_stem>/`.
- `.h5` files remain at the base folder for easier cleanup.
- Logs now include per-flightline prefixes during parallel execution.

### Improved
- Cleaner, more readable runtime output (replaced `GRGRGR...` with progress bars).
- More informative logs for downloads, exports, and corrections.
- Better idempotence safety and isolation between flightlines.

### Fixed
- Restored consistent skip logic for partially completed runs.
- Corrected missing download stage when running `go_forth_and_multiply()` on a new folder.

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
