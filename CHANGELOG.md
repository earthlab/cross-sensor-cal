## [Unreleased] – Pipeline refactor for idempotent, ordered execution (October 2025)

### Added
- New restart-safe pipeline stages with per-stage validation and skip logic:
  - ENVI export
  - BRDF/topo correction parameter JSON
  - BRDF + topographic correction
  - Sensor convolution/resampling
- Centralized path resolver `get_flightline_products()` that defines all expected file locations
  (raw ENVI, correction JSON, corrected ENVI, and per-sensor products).
- Logging now surfaces "✅ ... skipping" when a stage is already complete and valid.
- Automatic recovery from partial runs: if a previous run was interrupted mid-stage,
  the pipeline will re-run *just that* stage to repair missing/corrupt outputs.
- Per-sensor convolution now emits a summary log of succeeded, skipped, and failed sensors,
  tolerating partial success while still reporting detailed outcomes.

### Changed
- The pipeline no longer guesses filenames on the fly. All stage logic now pulls canonical
  paths from `get_flightline_products()` instead of hand-building names.
- Convolution/resampling now ALWAYS uses the BRDF+topo corrected ENVI product
  (`*_brdfandtopo_corrected_envi.img/.hdr`). It can no longer run "too early"
  on uncorrected reflectance or raw `.h5`.
- The correction JSON (`*_brdfandtopo_corrected_envi.json`) is now explicitly
  generated before the BRDF/topo correction step. The correction step reads it.
- Legacy GeoTIFF per-sensor exports were removed; every persistent sensor output is now an ENVI
  `.img/.hdr` pair.

### Fixed
- Removed a race/ordering bug where convolution could be attempted before BRDF+topo
  correction finished writing its outputs.
- Removed repeated large (~20+ GB) ENVI exports on reruns. If a valid export already
  exists, the pipeline skips re-exporting and will not blow up the kernel.
- Clearer, more specific error messages when expected outputs are missing or invalid,
  including guidance to align `get_flightline_products()` with actual on-disk filenames.

### Developer Notes
- `process_one_flightline()` is now the canonical workflow for a single flight line.
- `go_forth_and_multiply()` coordinates multiple flight lines and passes through
  options like `brightness_offset` and resampling mode.
- `get_flightline_products()` is now considered authoritative for naming and layout.
  If filenames change, update that function instead of editing every stage.
