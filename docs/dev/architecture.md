# Package Architecture

This page describes how cross-sensor-cal is organized internally. Understanding this structure helps contributors extend the pipeline or integrate new sensors without breaking guarantees.

---

## Design philosophy and invariants

- **Reproducibility first.** Pipeline behavior is predictable and restart-safe; stages skip when valid outputs already exist instead of recomputing.
- **Fixed ordering.** `process_one_flightline` and `go_forth_and_multiply` orchestrate the same sequence: ENVI export → BRDF/topographic parameter build → BRDF+topo correction → sensor convolution/resampling → Parquet exports → DuckDB merge → QA panels.
- **File contracts.** `FlightlinePaths` centralizes filenames and directories; stages communicate only through these artifacts. Downstream docs and CI treat the merged Parquet and QA products as required outputs.
- **Outputs are the API.** Functions return little; correctness is expressed through on-disk ENVI/Parquet and QA files that can be inspected or reused.

---

## Pipeline architecture (high level)

- **Stages** are pure file transforms. Each consumes a known input set and writes ENVI, Parquet, and JSON/PNG sidecars. Stages do not mutate shared state.
- **Orchestration** happens in `pipelines/pipeline.py` via `process_one_flightline` (single flightline) and `go_forth_and_multiply` (batch). Both rely on `FlightlinePaths` to resolve paths and naming before delegating work.
- **Communication** between stages is file-based. The BRDF+topo-corrected ENVI is always the source for sensor resampling; Parquet exports derive from both raw and corrected ENVI; the merged Parquet and QA summaries consume all earlier artifacts.
- **Restart safety** is achieved because each stage validates its outputs and returns early when they already exist. Partial runs can be resumed without recomputation or corrupting prior files.

---

## Directory structure (Python package)

- `cross_sensor_cal/pipelines/`: orchestration entry points and Ray helpers
- `cross_sensor_cal/exports/`: ENVI export helpers
- `cross_sensor_cal/io/`: schema and I/O helpers (e.g., NEON schema resolution)
- `cross_sensor_cal/utils/`: shared utilities, naming/path helpers, memory management
- `cross_sensor_cal/data/`: spectral metadata and calibration tables
  - `landsat_band_parameters.json`: band centers/FWHM used for resampling
  - `brightness/*.json`: brightness adjustments between Landsat and MicaSense
  - `hyperspectral_bands.json`: reference metadata for hyperspectral inputs
- `cross_sensor_cal/qa_plots.py` and `cross_sensor_cal/sensor_panel_plots.py`: QA visualization utilities
- `cross_sensor_cal/standard_resample.py`: spectral resampling and coefficients

---

## Extending the system safely

### Adding or modifying a target sensor
- Update spectral definitions in `cross_sensor_cal/data/landsat_band_parameters.json` (or analogous table for the new sensor) and ensure resampling logic in `standard_resample.py` knows how to consume them.
- Confirm `get_flightline_products` and `FlightlinePaths` generate filenames for the new sensor; outputs must still include merged Parquet and QA artifacts.
- Add tests that validate band definitions and resampled outputs; do not bypass the existing stage ordering.

### Updating brightness or calibration coefficients
- Brightness and regression tables live under `cross_sensor_cal/data/brightness/` and are loaded via `brightness_config`. Changes here affect downstream cross-sensor harmonization.
- Keep JSON schema and key names stable; update any dependent tests and documentation describing the coefficients.
- Validate against Landsat-referenced QA outputs to confirm calibrations remain within expected bounds.

### Modifying QA outputs
- QA panels and JSON summaries are produced after merging outputs. Filenames such as `<flight_id>_qa.png` and `<flight_id>_qa.json` are assumed by docs and CI.
- If adding metrics or changing formats, ensure `_qa.png` and `_qa.json` remain available and update the QA tests under `tests/test_qa` accordingly.
- Maintain quick-mode rendering used in CI fixtures so drift checks continue to pass.

---

## Relationship to scientific reproducibility

- The repository encodes the workflow described in the RSE manuscript; artifacts (ENVI, Parquet, QA) are the evidence trail for analyses.
- Centralized naming, stage ordering, and idempotent execution make runs auditable and repeatable across environments.
- Contributors are expected to preserve these invariants so published and future analyses can be reproduced from the same on-disk products.

---

## Next steps

- [Contributing & development workflow](contributing.md)
- [Guidelines for AI/Codex edits](codex-guidelines.md)
