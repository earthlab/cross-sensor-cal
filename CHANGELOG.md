## [2.3.0] – 2025-11-03
### Added
- Config-driven brightness coefficients for Landsat→MicaSense (`landsat_to_micasense.json`) and helper loader.
- Automatic per-band brightness adjustment applied to Landsat-convolved products, recorded in QA JSON and brightness tables.
- Multi-page QA report (`*_qa.pdf`) with:
  - Page 1: ENVI product overview (one row, one panel per ENVI file).
  - Page 2: topographic and BRDF diagnostics (two rows).
  - Page 3: remaining QA diagnostics (convolution accuracy, header/mask summaries, issues, brightness coefficients).
- Expanded QA JSON metrics, including header integrity, mask coverage, Δ reflectance, convolution error, and brightness coefficients.

### Changed
- ENVI export and pipeline logs now use affirmative, progress-oriented wording (e.g., “creating new ENVI export” instead of “not found or invalid”).
- CI simplified to four main checks on PRs: `CI / lite`, `CI / unit`, `Docs Drift Check / audit`, and `QA quick check / qa`.

### Fixed
- Duplicate QA/pytest workflows removed; QA quick check now runs once per PR (and optionally once per push to `main`).

## [2025-10-30] Added Merge Stage + Restored QA Panel

- Added new DuckDB-based merge step combining original, corrected, and resampled pixel tables.
- Merged output uses naming convention: `<prefix>_merged_pixel_extraction.parquet`.
- QA panel (`<prefix>_qa.png`) is now rendered automatically after each merge, even in parallel runs.
- Documentation and examples updated accordingly.

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
