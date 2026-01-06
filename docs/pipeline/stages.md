# Pipeline Overview & Stages

The cross-sensor-cal pipeline transforms NEON HDF5 directional reflectance into physically corrected and sensor-harmonized reflectance products. Each stage is restart-safe and produces structured, auditable outputs.

This page describes every stage of the pipeline, what it consumes, what it produces, and what can go wrong.

---

## Pipeline stages and idempotence

The orchestrators `process_one_flightline` and `go_forth_and_multiply` enforce the following order:

1. Download HDF5 (via `stage_download_h5`).
2. Export ENVI.
3. Build correction JSON.
4. Apply BRDF + topo correction.
5. Resample/convolve all sensors.
6. Export Parquet sidecars.
7. DuckDB merge to merged parquet.
8. Render QA panel + metrics.

Each stage checks whether its expected outputs already exist and are valid, logs a skip message when they do, and recomputes missing or corrupted artefacts. Recovery mode exists for raw ENVI exports when corrected outputs are present (`stage_export_envi_from_h5` supports `recover_missing_raw`, used by `cscal-recover-raw`).

---

<a id="data-acquisition"></a>
## 1. Data acquisition

**Inputs:**
- NEON API paths or local HDF5 files

**Outputs:**
- cached HDF5 tiles stored under the selected `--base-folder`

Downloads are handled by `stage_download_h5` and are triggered automatically by `go_forth_and_multiply`.

---

<a id="hdf5-to-envi"></a>
## 2. HDF5 → ENVI export

**Inputs:**
- `*_directional_reflectance.h5`
- per-pixel geometry and metadata

**Outputs:**
- `*_envi.img/.hdr` (see [Outputs](outputs.md))

`stage_export_envi_from_h5` creates the ENVI pair using the canonical naming in `FlightlinePaths`, with optional brightness offsets.

---

<a id="topographic-correction"></a>
## 3. Topographic + BRDF correction

**Inputs:**
- directional or raw ENVI exports
- DEM-derived slope and aspect
- solar/view geometry

**Outputs:**
- `<flight_id>_brdfandtopo_corrected_envi.(img|hdr|json)`

`stage_build_and_write_correction_json` writes the parameter JSON, and `stage_apply_brdf_and_topo` applies the combined correction before downstream resampling.

---

<a id="sensor-harmonization"></a>
## 4. Sensor harmonization (spectral convolution)

**Inputs:**
- BRDF+topo corrected ENVI
- sensor spectral response functions (SRFs)

**Outputs:**
- `<flight_id>_<sensor>_envi.(img|hdr|parquet)`

`stage_convolve_all_sensors` delegates to the configured resample method (convolution, legacy, or resample) and iterates through `FlightlinePaths.sensor_products`.

---

<a id="parquet-extraction-merging"></a>
## 5. Parquet extraction & merging

**Inputs:**
- any ENVI cube produced by earlier stages

**Outputs:**
- Parquet files for raw, corrected, and resampled cubes
- `<flight_id>_merged_pixel_extraction.parquet`

`_export_parquet_stage` builds the per-product Parquet sidecars, and `merge_flightline` (DuckDB) consolidates them with schema validation before optional QA rendering.

---

<a id="quality-assurance"></a>
## 6. Quality assurance (QA)

**Inputs:**
- merged parquet and supporting ENVI files

**Outputs:**
- `<flight_id>_qa.png`
- `<flight_id>_qa.json`
- `<flight_id>_qa.pdf` (when rendered)

QA artefacts come from `render_flightline_panel` and mirror the canonical stems listed in [Outputs](outputs.md).

---

## Next steps

- [Outputs & file structure](outputs.md)
- [QA panels & metrics](qa.md)

---
