# Cross-Sensor Calibration

Earth Lab’s **cross-sensor-cal** is a Python package for producing physically corrected and sensor-harmonized reflectance from NEON hyperspectral data and other fine-resolution imagery.

## What cross-sensor-cal does

- Converts NEON AOP directional reflectance HDF5 flightlines into ENVI products (see `stage_export_envi_from_h5`).
- Builds a correction JSON, applies BRDF + topo correction, then resamples to target sensors via convolution or a configured resample method (`process_one_flightline` in `cross_sensor_cal.pipelines.pipeline`).
- Exports Parquet sidecars for raw/corrected/resampled products, then merges them with DuckDB into `<flight_id>_merged_pixel_extraction.parquet` (`merge_flightline` in `cross_sensor_cal.merge_duckdb`).
- Generates QA outputs `<flight_id>_qa.png` and `<flight_id>_qa.json` (and PDF when produced) via `render_flightline_panel`.

It provides a reproducible workflow that:

- exports NEON HDF5 to ENVI
- applies topographic and BRDF correction
- harmonizes reflectance to Landsat / MicaSense bandspaces
- writes Parquet tables
- emits QA PNG, PDF, and JSON summaries

---

## Minimal example (CLI)

```bash
BASE=output_demo
mkdir -p "$BASE"

cscal-pipeline \
  --base-folder "$BASE" \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2 \
  --engine thread
```

Inspect QA outputs:

```bash
open $BASE/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/*_qa.png
```

For a Python/Jupyter version, see:

- Usage → [Jupyter notebook example](usage/notebook-example.md)

## How the pipeline works

1. [HDF5 → ENVI export](pipeline/stages.md#hdf5-to-envi)
2. [Topographic correction](pipeline/stages.md#topographic-correction)
3. [BRDF correction](pipeline/stages.md#brdf-correction)
4. [Sensor harmonization (bandpass convolution)](pipeline/stages.md#sensor-harmonization)
5. [Parquet extraction + merging](pipeline/stages.md#parquet-extraction-merging)
6. [QA reporting](pipeline/stages.md#quality-assurance)

See the [Pipeline overview](pipeline/stages.md) for details.

## Next steps

- [Quickstart](quickstart.md)
- [Jupyter notebook example](usage/notebook-example.md)
- [Tutorials](tutorials/neon-to-envi.md)
- [Reference](reference/configuration.md)

---
