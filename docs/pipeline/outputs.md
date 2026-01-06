# Outputs & File Structure

The pipeline writes every artefact into a per-flightline folder using naming rules shared by `FlightlinePaths` and `cross_sensor_cal.utils.naming`. All patterns below come directly from `src/cross_sensor_cal/paths.py` and are reused by the CLI and QA tools.

## Per-flightline outputs (canonical names)

Naming is centralized in `cross_sensor_cal.paths.FlightlinePaths` and `cross_sensor_cal.utils.naming.get_flightline_products`.

| Output | Pattern |
| --- | --- |
| Raw ENVI | `<flight_id>_envi.(img|hdr)` with optional `<flight_id>_envi.parquet` |
| Corrected ENVI | `<flight_id>_brdfandtopo_corrected_envi.(img|hdr|json|parquet)` |
| Sensor outputs | `<flight_id>_<sensor>_envi.(img|hdr|parquet)` where `sensor` is one of `landsat_tm`, `landsat_etm+`, `landsat_oli`, `landsat_oli2`, `micasense`, `micasense_to_match_tm_etm+`, `micasense_to_match_oli_oli2` |
| Merged Parquet | `<flight_id>_merged_pixel_extraction.parquet` |
| QA | `<flight_id>_qa.png`, `<flight_id>_qa.json`, `<flight_id>_qa.pdf` (when rendered) |

Docs drift CI expects `_merged_pixel_extraction.parquet` and `_qa.png` to be mentioned; see `tools/doc_drift_audit.py`.

## Where outputs come from

- ENVI exports are produced in `stage_export_envi_from_h5` and validated in `process_one_flightline`.
- BRDF/topo correction plus the correction JSON are written by `stage_build_and_write_correction_json` and `stage_apply_brdf_and_topo`.
- Sensor-specific cubes follow the resample method passed to `process_one_flightline` and use the sensor list from `FlightlinePaths.sensor_products`.
- Parquet sidecars for raw, corrected, and resampled cubes are emitted by `_export_parquet_stage` and merged by `merge_flightline` into the canonical merged parquet.
- QA panels and metrics (`render_flightline_panel`) align with the same naming stems, including `_qa.png` and `_qa.json`.

