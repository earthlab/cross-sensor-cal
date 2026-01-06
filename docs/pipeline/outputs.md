# Outputs & File Structure

## Purpose of this page
- Outputs on disk are the primary interface of SpectralBridge.
- Downstream analyses should rely on these artefacts, not return values from the Python API.

## Canonical per-flightline outputs
Naming stems come from `spectralbridge.paths.FlightlinePaths` and `spectralbridge.utils.naming.get_flightline_products`; sensor-specific stems come from `SensorProductPaths`.

| Output type | Canonical filename pattern | Description | Notes / guarantees |
| --- | --- | --- | --- |
| Raw ENVI (when available) | `<flight_id>_envi.(img|hdr|parquet)` | Direct export of the NEON HDF5 reflectance cube. | Used when present to seed later stages; parquet sidecar is written when exported. |
| BRDF + topographic corrected ENVI | `<flight_id>_brdfandtopo_corrected_envi.(img|hdr|json|parquet)` | Physics-informed normalization and correction JSON produced before sensor resampling. | One corrected set per flightline; parquet mirrors the corrected ENVI cube. |
| Sensor-resampled ENVI + Parquet | `<flight_id>_<sensor>_envi.(img|hdr|parquet)` where `<sensor>` is one of `landsat_tm`, `landsat_etm+`, `landsat_oli`, `landsat_oli2`, `micasense`, `micasense_to_match_tm_etm+`, `micasense_to_match_oli_oli2` | Reflectance cubes resampled into the Landsat-referenced frame and MicaSense variants. | Each sensor has its own ENVI/Parquet trio; stems are consistent with `FlightlinePaths.sensor_products`. |
| Merged Parquet | `<flight_id>_merged_pixel_extraction.parquet` | Master table that merges Parquet sidecars across stages into one analysis-ready spectral library. | Exactly one per flightline; treated as the primary success signal. |
| QA artefacts | `<flight_id>_qa.png`, `<flight_id>_qa.json`, optional `<flight_id>_qa.pdf` | Visual and numeric QA summaries aligned to the merged outputs. | PNG and JSON are expected for every completed run; PDF is produced when rendering is enabled. |
| QA metrics parquet | `<flight_id>_qa_metrics.parquet` | Structured QA metrics by band and sensor. | Emitted alongside QA JSON/PNG when QA calculation runs. |

## What “success” means
- The merged parquet exists and is readable: `<flight_id>_merged_pixel_extraction.parquet`.
- The QA PNG renders: `<flight_id>_qa.png` (with matching `<flight_id>_qa.json`).
- Sensor-specific ENVI/Parquet products exist as configured; absence may reflect configuration rather than failure.
- If the merged parquet and QA PNG are present, the pipeline completed successfully for that flightline.

## Idempotence and restart-safety
`process_one_flightline` and `go_forth_and_multiply` skip stages whose outputs already exist and validate, so re-running the pipeline will not recompute completed products. This skip-if-valid behavior makes restarts safe and is relied upon in notebook workflows and batch processing alike.

## CI and documentation guarantees
- Documentation drift checks (`tools/doc_drift_audit.py`) assert that `_merged_pixel_extraction.parquet` and `_qa.png` remain part of the documented contract.
- QA expectations and output stems are shared between user-facing docs and automated validation to protect reproducibility.

## How to rely on these outputs
- Load Parquet products directly (especially the merged parquet) for analysis; they are the authoritative API.
- Inspect QA PNG/JSON before downstream modeling to confirm spectral health and calibration quality.
- Treat intermediate ENVI products as optional diagnostics; most workflows can ignore them once Parquet and QA artefacts are present.
