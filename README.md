# Cross-Sensor Calibration

Cross-Sensor Calibration provides a Python pipeline for processing NEON Airborne Observation Platform hyperspectral flight lines and resampling them to emulate alternate sensors in a reproducible, scriptable workflow.

![Pipeline diagram](docs/img/pipeline.png)

## Quickstart

> As of v2.0, cross-sensor-cal no longer depends on HyTools. All NEON handling is internal and GPL-compatible.

Install the lightweight base package:

```bash
pip install cross-sensor-cal
```

Create a workspace and download a flight line:

```bash
mkdir -p data/SITE
cscal-download SITE --year-month 2021-06 --flight FLIGHT_LINE --output data
```

Run the end-to-end processing pipeline (see `--help` for all options):

```bash
cscal-pipeline --help
```

If you need the full geospatial/hyperspectral toolchain (Rasterio, GeoPandas, Spectral, Ray, HDF5),
install the optional extras:

```bash
pip install "cross-sensor-cal[full]"
```

Feature availability by install type:

| Feature | Base | `[full]` |
|---|---|---|
| Core array ops (NumPy/Scipy) | ✅ | ✅ |
| Raster I/O (rasterio) | ⚠️ (not included) | ✅ |
| Vector I/O/ops (GeoPandas) | ⚠️ | ✅ |
| ENVI/HDR (spectral) | ⚠️ | ✅ |
| HDF5 (h5py) | ⚠️ | ✅ |
| Ray parallelism | ⚠️ | ✅ |

Replace `SITE` with a NEON site code and `FLIGHT_LINE` with an actual line identifier.

## Running the pipeline

```python
from pathlib import Path
from cross_sensor_cal.pipelines.pipeline import go_forth_and_multiply

go_forth_and_multiply(
    base_folder=Path("output_tester"),
    site_code="NIWO",
    year_month="2023-08",
    flight_lines=[
        "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance",
        "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance",
    ],
)
```

This will:

1. Export the NEON hyperspectral `.h5` cube for each flightline into ENVI format.
2. Generate a BRDF + topographic correction parameter JSON.
3. Apply BRDF + topographic correction to produce:
   `*_brdfandtopo_corrected_envi.img/.hdr`
4. Convolve the corrected reflectance into a set of sensor-like products
   (e.g. Landsat-style bands).

### Idempotent / restart-safe

You can safely rerun the same command. The pipeline is stage-aware:

- If a stage already produced a valid output, that stage is skipped.
- If an output is missing or looks corrupted/empty, only that stage is recomputed.
- If you crashed halfway through a long run, you can just rerun the same cell to resume.

Typical log output on a rerun looks like:

```
12:01:03 | INFO | 🚀 Processing NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance ...
12:01:03 | INFO | 🔎 ENVI export target for ... is NEON_D13_..._envi.img / NEON_D13_..._envi.hdr
12:01:03 | INFO | ✅ ENVI export already complete for ..._envi.img / ..._envi.hdr (skipping heavy export)
12:01:04 | INFO | ✅ Correction JSON already complete for ..._brdfandtopo_corrected_envi.json (skipping)
12:01:05 | INFO | ✅ BRDF+topo correction already complete for ..._brdfandtopo_corrected_envi.img/.hdr (skipping)
12:01:06 | INFO | 🎯 Convolving corrected reflectance for ...
12:01:06 | INFO | ✅ landsat_oli product already complete ... (skipping)
12:01:06 | INFO | 🎉 Finished pipeline for NEON_D13_...
```

### Memory safety

The new pipeline will NOT keep re-loading 20+ GB hyperspectral cubes into memory on every rerun.
The ENVI export step now checks for an existing, valid ENVI pair before doing any heavy work.
If it's already there, it logs "✅ ... skipping heavy export" and moves on.

### Data products

After a successful run you should see, for each flight line:

- `<flight_stem>.h5`
- `<flight_stem>_envi.img/.hdr`
  (Uncorrected reflectance cube in ENVI format; first stage output)
- `<flight_stem>_brdfandtopo_corrected_envi.json`
  (JSON parameters used for physical correction)
- `<flight_stem>_brdfandtopo_corrected_envi.img/.hdr`
  (Corrected reflectance cube in ENVI format; used for all downstream work)
- Per-sensor convolution/resample outputs (e.g. Landsat-like products); each file
  is generated from the corrected ENVI cube, not from the raw `.h5`.

The `_brdfandtopo_corrected_envi` suffix is now guaranteed and should be considered
the canonical "final" reflectance for analysis and downstream comparisons.

### Pipeline stages

The pipeline is now organized into four explicit stages that always run in order:

1. **ENVI Export**  
   Converts the NEON `.h5` directional reflectance cube to ENVI (`*_envi.img/.hdr`).  
   Large and expensive. Skipped on rerun if valid output exists.

2. **Correction JSON Build**  
   Computes and writes `<flight_stem>_brdfandtopo_corrected_envi.json`, which contains
   BRDF and topographic correction parameters.  
   This file is required for the next step.

3. **BRDF + Topographic Correction**  
   Uses the correction JSON to generate physically corrected reflectance in ENVI format:  
   `<flight_stem>_brdfandtopo_corrected_envi.img/.hdr`.  
   These corrected products are now the "truth" for downstream use.

4. **Sensor Convolution / Resampling**  
   Convolves the corrected reflectance cube to sensor-specific bandsets
   (e.g. Landsat TM, Landsat OLI, etc.).  
   Each sensor product is generated from the corrected ENVI, never from the raw `.h5`,
   and each product is skipped if it already exists.

This enforced order prevents earlier bugs where convolution could run on uncorrected data.

### Developer notes

- `process_one_flightline()` is now the canonical per-flightline workflow.
- `go_forth_and_multiply()` loops over flightlines and handles options like `brightness_offset`.
- `get_flightline_products()` is the single source of truth for naming and layout of:
  - the `.h5` input,
  - the uncorrected ENVI export,
  - the correction JSON,
  - the corrected ENVI (`*_brdfandtopo_corrected_envi.*`),
  - the per-sensor convolution outputs.

  All pipeline stages call `get_flightline_products()` instead of guessing filenames.
  If file naming changes, update `get_flightline_products()`, not each stage.

- Each stage validates its outputs (non-empty files, parseable JSON, etc.). If outputs are valid,
  that stage logs "✅ ... skipping" and returns immediately.  
  If outputs are missing or corrupted, that stage recomputes them.  
  This is what makes the pipeline resumable after a crash or partial run.

## Install

Cross-Sensor Calibration depends on GDAL, PROJ, Ray, h5py, and optional geospatial stacks such as Rasterio and GeoPandas. We recommend the Conda workflow below because it installs the required native libraries automatically. If you prefer a pure `pip` workflow, install system packages for GDAL/PROJ first (e.g., `brew install gdal` on macOS or `apt-get install gdal-bin libgdal-dev proj-bin` on Debian/Ubuntu).

### Conda

```bash
conda create -n cscal python=3.10 gdal proj
conda activate cscal
pip install -e .
```

### uv/pip

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

If GDAL wheels are unavailable for your platform, install it from Conda-forge and then point `pip` at the Conda environment by exporting `CPLUS_INCLUDE_PATH` and `C_INCLUDE_PATH`.

## Documentation

Browse the full documentation site at
[earthlab.github.io/cross-sensor-cal](https://earthlab.github.io/cross-sensor-cal).
The site is built with MkDocs Material and automatically deployed to GitHub
Pages.

Key entry points:

- [Overview](docs/overview.md)
- [Quickstart](docs/quickstart.md)
- [Stage 01 Raster Processing](docs/stage-01-raster-processing.md)
- [Stage 02 Sorting](docs/stage-02-sorting.md)
- [Stage 03 Pixel Extraction](docs/stage-03-pixel-extraction.md)
- [Stage 04 Spectral Library](docs/stage-04-spectral-library.md)
- [Stage 05 MESMA](docs/stage-05-mesma.md)

## Support Matrix

| Python | OS            | GDAL | Ray |
|--------|---------------|------|-----|
| 3.10+  | Linux, macOS  | 3.4+ | 2.0+ |

## How to cite

If you use Cross-Sensor Calibration in your research, please cite the project:

```
Earth Lab Data Innovation Team. (2025). Cross-Sensor Calibration (Version 0.1.0) [Software]. University of Colorado Boulder. https://github.com/earthlab/cross-sensor-cal
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## License and Citation

Distributed under the GPLv3 License. Please cite the project using [CITATION.cff](CITATION.cff).

