# Configuration

Most users can rely on default configuration, but the pipeline allows fine-grained control over processing stages, performance, and file handling.

This page documents the available configuration parameters and how they affect pipeline behavior.

---

## Configuration sources

1. Command-line options (highest priority)  
2. Environment variables  
3. Default internal settings  

---

## Common settings

### `base-folder`

Directory where:

- downloaded HDF5 tiles  
- ENVI exports  
- Parquet tables  
- QA artifacts  

are written.

---

### `engine`

Execution backend:

- `thread` (default)  
- `ray` (distributed/hyperparallel workflows)

---

### `max-workers`

Controls concurrency in:

- ENVI export  
- BRDF and topo correction  
- Parquet extraction  

Use cautiously when memory is limited.

---

### Pipeline scope

The CLI currently runs the full stage sequence; idempotent checks in `process_one_flightline` skip work when outputs already exist.

### Environment variables

| Variable | Meaning |
| --- | --- |
| `CSCAL_TMPDIR` | Override temporary directory |
| `CSCAL_LOGLEVEL` | Set logging verbosity |
| `CSCAL_RAY_ADDRESS` | Use an existing Ray cluster |

Advanced configuration
These settings primarily matter for large-scale workflows:

- chunk sizes for Parquet extraction
- memory thresholds for Ray worker processes
- default CRS assignments
- sensor SRF paths

Details of internal architecture appear in the Developer section.

---

## Sensor support & configuration sources

Supported sensor outputs (from `FlightlinePaths.sensor_products`):

- `landsat_tm`
- `landsat_etm+`
- `landsat_oli`
- `landsat_oli2`
- `micasense`
- `micasense_to_match_tm_etm+`
- `micasense_to_match_oli_oli2`

Spectral configuration comes from:

- `cross_sensor_cal/data/landsat_band_parameters.json`
- `cross_sensor_cal/data/hyperspectral_bands.json`
- brightness coefficients loaded via `load_brightness_coefficients(system_pair)` in `cross_sensor_cal/brightness_config.py` with tables stored under `cross_sensor_cal/data/brightness/*.json`

If you update brightness coefficients, refresh the JSON tables accordingly. Provenance/citation for brightness regression tables: (needs project decision).
Next steps
JSON schemas
Validation metrics
Pipeline stages

---
