# Package Architecture

This page describes how cross-sensor-cal is organized internally. Understanding this structure helps contributors extend the pipeline or integrate new sensors.

---

## High-level design

The package is built around a **stage-based pipeline**. Each stage:

- consumes well-defined inputs  
- writes ENVI/Parquet outputs  
- logs metadata to JSON sidecar files  
- is restart-safe  

Stages are orchestrated by the main CLI driver.

---

## Directory structure (Python package)

cross_sensor_cal/
pipeline/
download.py
export_envi.py
topo.py
brdf.py
convolution.py
parquet.py
qa.py
utils/
data/
srf/
regression/

---

## Pipeline modules

### `download.py`
Fetches NEON HDF5 tiles.

### `export_envi.py`
Extracts directional reflectance and writes ENVI + metadata.

### `topo.py`
Applies DEM-based topographic correction.

### `brdf.py`
Computes BRDF coefficients and corrects reflectance.

### `convolution.py`
Performs sensor bandpass integration.

### `parquet.py`
Extracts pixel-level data and merges tables.

### `qa.py`
Generates PNG, PDF, and JSON QA outputs.

---

## Data assets

Stored under `cross_sensor_cal/data/`:

- spectral response functions (SRFs)  
- MicaSense → Landsat regression tables  
- wavelength lookup and metadata reference files  

---

## Adding a new sensor

1. Add SRF tables under `data/srf/`  
2. Update convolution stage mappings  
3. Add tests and metadata checks  
4. Document the new sensor in tutorials and reference pages  

---

## Next steps

- [Contributing & development workflow](contributing.md)  
- [Guidelines for AI/Codex edits](codex-guidelines.md)
