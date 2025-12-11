# Pipeline Overview & Stages

The cross-sensor-cal pipeline transforms NEON HDF5 directional reflectance into physically corrected and sensor-harmonized reflectance products. Each stage is restart-safe and produces structured, auditable outputs.

This page describes every stage of the pipeline, what it consumes, what it produces, and what can go wrong.

---

## Pipeline summary

1. **Data acquisition** (download NEON HDF5 tiles)
2. **HDF5 → ENVI export**
3. **Topographic correction**
4. **BRDF correction**
5. **Sensor harmonization (spectral convolution)**
6. **Parquet extraction + merging**
7. **Quality assurance (QA PNG, PDF, JSON)**

Each stage can be run independently using the `--start-at` and `--end-at` flags.

---

<a id="data-acquisition"></a>
## 1. Data acquisition

**Inputs:**
- NEON API paths or local HDF5 files

**Outputs:**
- cached HDF5 tiles stored under the selected `--base-folder`

The pipeline fetches *only* the tiles required for the selected flight line.

**Common issues:**
- missing HDF5 files in NEON storage
- interrupted downloads in cloud environments
- insufficient space in temporary directories

---

<a id="hdf5-to-envi"></a>
## 2. HDF5 → ENVI export

**Inputs:**
- `*_directional_reflectance.h5`
- per-pixel geometry and metadata

**Outputs:**
- `*_directional_reflectance_envi.img/.hdr`
- sidecar JSON documenting extracted wavelengths, masks, and scaling

This stage produces an ENVI image that mirrors the HDF5 directional reflectance dataset.

**What the ENVI file contains:**
- reflectance (scaled NEON values)
- wavelength metadata
- per-pixel masks (cloud, cloud shadow, water, snow, invalid)

**Common issues:**
- mismatch between HDF5 metadata and ENVI header
- extremely large tile sizes causing I/O delays
- NaN bands due to malformed HDF5 datasets

---

<a id="topographic-correction"></a>
## 3. Topographic correction

**Inputs:**
- directional reflectance ENVI
- DEM-derived slope and aspect
- solar geometry

**Outputs:**
- `*_topocorrected_envi.img`

Topographic correction reduces slope- and aspect-driven variation in illumination.

The method assumes surface reflectance behaves consistently with simple terrain-adjustment models.

**Common issues:**
- DEM resolution mismatch
- strong terrain shadows that remain after correction
- negative reflectance in deep shadows (masked)

---

<a id="brdf-correction"></a>
## 4. BRDF correction

**Inputs:**
- topographically corrected ENVI reflectance
- view geometry (sensor zenith / azimuth)
- solar geometry

**Outputs:**
- `*_brdfandtopo_corrected_envi.img`
- BRDF coefficient tables in the QA JSON

BRDF correction adjusts reflectance to a consistent view/illumination angle, making spectra across the flight line more comparable.

**Common issues:**
- instabilities in BRDF coefficient fitting
- extreme reflectance values that must be masked
- spatial artifacts in low-SNR bands

---

<a id="sensor-harmonization"></a>
## 5. Sensor harmonization (spectral convolution)

**Inputs:**
- BRDF+topo corrected ENVI
- sensor spectral response functions (SRFs)

**Outputs:**
- `*_landsat_convolved_envi.img` or other sensor-equivalent ENVI products
- bandpass-harmonized Parquet files

This stage integrates the corrected spectrum against the target sensor's SRFs.
Supported sensors include Landsat OLI/OLI-2; others can be added.

**Common issues:**
- wavelength misalignment
- missing SRF tables
- sensor bands with near-zero response across NEON wavelengths

---

<a id="parquet-extraction-merging"></a>
## 6. Parquet extraction & merging

**Inputs:**
- any ENVI cube produced by earlier stages

**Outputs:**
- a Parquet file per cube (one row per pixel)
- a merged pixel extraction table for the whole flight line

This step makes downstream analysis easy in Python, R, or DuckDB.

**Common issues:**
- extremely large tables (billions of rows)
- insufficient memory for merges
- incorrect CRS metadata in ENVI headers

---

<a id="quality-assurance"></a>
## 7. Quality assurance (QA)

**Inputs:**
- all previous outputs

**Outputs:**
- `*_qa.png`
- `*_qa.pdf`
- `*_qa.json`

QA artifacts summarize reflectance distributions, masks, wavelength metadata, and BRDF/brightness statistics.
See the [QA page](qa.md) for details.

---

## Running a partial pipeline

You can run only part of the pipeline:

```bash
cscal-pipeline \
  --start-at brdf \
  --end-at convolution
```

Or run a single stage manually if needed.

## Next steps

- [Outputs & file structure](outputs.md)
- [QA panels & metrics](qa.md)

---
