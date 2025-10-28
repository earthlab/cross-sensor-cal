## [Unreleased] – Pipeline Refactor and Idempotent Execution (October 2025)
### Added
- New stage-level skipping logic for all major pipeline steps (ENVI export, correction JSON, BRDF/topo correction, convolution).
  Each step now checks if outputs already exist and are valid before recomputing.
  This makes the pipeline fully **idempotent** and restart-safe.
- Validation utilities:
  - `_nonempty_file(p)`
  - `is_valid_envi_pair(img, hdr)`
  - `is_valid_json(json_path)`
  These prevent reprocessing incomplete or corrupted outputs.
- Structured per-flightline orchestration (`process_one_flightline`) with explicit step order.
- Detailed logging at every step, including "✅ already complete, skipping" messages and improved error reporting.

### Changed
- **go_forth_and_multiply()** now orchestrates the full four-stage workflow:
  1. ENVI export  
  2. Correction JSON creation  
  3. BRDF + topographic correction  
  4. Convolution/resampling
  This replaces the previous inline logic where convolution could run before correction finished.
- Convolution now runs **only** on the corrected ENVI (suffix `_brdfandtopo_corrected_envi`) instead of on raw `.h5` or uncorrected ENVI files.
- Stage outputs and logs are standardized: corrected products use the canonical suffix
  `_brdfandtopo_corrected_envi.img/.hdr/.json`.

### Fixed
- Bug where convolution sometimes began before corrected ENVI files were fully written.
- Misleading log messages referencing the `.h5` file during resampling.
- Partial run corruption: incomplete files from interrupted runs are now detected and recomputed.

### Developer Notes
- The pipeline is now resumable: you can rerun `go_forth_and_multiply()` safely without overwriting valid data.
- Each stage validates its outputs before skipping, ensuring reproducibility and stability for long runs.
- Downstream sorting and file discovery continue to rely on the `_brdfandtopo_corrected_envi` suffix; this remains unchanged.
