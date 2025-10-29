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

## Pipeline overview

Cross-Sensor Calibration processes every flight line through four ordered,
restart-safe stages. Each stage writes ENVI (`.img/.hdr`) artifacts using
canonical names from `get_flightline_products()` and logs whether it is doing
work or skipping because valid outputs already exist.

1. **ENVI export** – Loads the NEON hyperspectral `.h5` flight line, logs the
   target ENVI pair, and writes `<flight_stem>_envi.img/.hdr`. On rerun, if that
   pair validates, the stage emits `✅ ENVI export already complete ... (skipping heavy export)`.
2. **Correction JSON build** – Computes BRDF + topographic parameters and writes
   `<flight_stem>_brdfandtopo_corrected_envi.json`. Valid JSON on disk triggers
   `✅ Correction JSON already complete ... (skipping)`.
3. **BRDF + topographic correction** – Uses the correction JSON to produce the
   canonical reflectance cube
   (`<flight_stem>_brdfandtopo_corrected_envi.img/.hdr`). When the corrected ENVI
   pair already validates the stage logs
   `✅ BRDF+topo correction already complete ... (skipping)`.
4. **Sensor convolution / resampling** – Reads only the corrected ENVI cube and
   produces per-sensor ENVI pairs named `<flight_stem>_<sensor_name>_envi.img/.hdr`.
   Each sensor logs either `✅ Wrote ...` when newly generated or `✅ ... already complete ... (skipping)`
   when the product validates. The stage concludes with
   `📊 Sensor convolution summary ... | succeeded=[...] skipped=[...] failed=[...]` and
   then `🎉 Finished pipeline for <flight_stem>`.

Every persistent output (raw, corrected, and sensor-specific) is an ENVI pair
plus a single correction JSON. GeoTIFF products are no longer part of the
advertised workflow.

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

This will execute the four stages above for every flight line and leave ENVI
products in `base_folder` using the canonical naming patterns. After the last
flight line completes, the pipeline logs `✅ All requested flightlines processed.`.

### Idempotent / restart-safe

You can safely rerun the same command. The pipeline is stage-aware and
restart-safe:

- If a stage already produced a valid output, that stage logs a `✅ ... (skipping)`
  message and returns immediately.
- If an output is missing or looks corrupted/empty, only that stage is recomputed.
- If you crashed halfway through a long run, you can rerun the same call to resume where
  work is still needed.

A realistic rerun for one flight line now looks like:

```
🚀 Processing NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance ...
🔎 ENVI export target for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance is NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.hdr
✅ ENVI export already complete for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.hdr (skipping heavy export)
✅ Correction JSON already complete for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_brdfandtopo_corrected_envi.json (skipping)
✅ BRDF+topo correction already complete for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_brdfandtopo_corrected_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_brdfandtopo_corrected_envi.hdr (skipping)
🎯 Convolving corrected reflectance for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance
✅ Wrote landsat_tm product for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_tm_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_tm_envi.hdr
✅ Wrote landsat_etm+ product for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_etm+_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_etm+_envi.hdr
✅ Wrote landsat_oli product for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_oli_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_oli_envi.hdr
✅ Wrote landsat_oli2 product for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_oli2_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_oli2_envi.hdr
✅ Wrote micasense product for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_micasense_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_micasense_envi.hdr
📊 Sensor convolution summary for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance | succeeded=['landsat_tm', 'landsat_etm+', 'landsat_oli', 'landsat_oli2', 'micasense'] skipped=[] failed=[]
🎉 Finished pipeline for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance
```

After all requested flight lines finish, the run concludes with
`✅ All requested flightlines processed.`

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
- Per-sensor convolution/resample outputs (e.g. Landsat-style bandstacks). Each
  sensor emits an ENVI pair named `<flight_stem>_<sensor>_envi.img` and
  `<flight_stem>_<sensor>_envi.hdr`, generated directly from the corrected ENVI
  cube. GeoTIFF sensor exports are no longer produced. Successful reruns may
  log these as `skipped` when valid products already exist.

The `_brdfandtopo_corrected_envi` suffix is now guaranteed and should be considered
the canonical "final" reflectance for analysis and downstream comparisons.

### Pipeline stages

Each stage uses `get_flightline_products()` to discover its inputs/outputs and
performs a restart-safe validation before doing work. Valid ENVI pairs or JSON
artifacts are reused rather than recomputed, ensuring reruns only fill in
missing or corrupted pieces.

#### Sensor convolution / resampling behavior

- The final stage turns the corrected reflectance cube
  (`*_brdfandtopo_corrected_envi.img/.hdr`) into simulated sensor products
  (e.g. Landsat-style band stacks).
- Each target sensor is attempted independently. Missing/unknown sensor definitions
  are logged with a warning and skipped.
- Each simulated sensor writes an ENVI `.img/.hdr` pair named
  `<flight_stem>_<sensor>_envi.*`. GeoTIFFs are no longer emitted by this stage.
- If a sensor product already exists on disk and validates as an ENVI pair, it is skipped with
  a `✅ ... already complete ... (skipping)` log.
- At the end of the stage, the pipeline logs a summary of which sensors succeeded,
  which were skipped (already done), and which failed.
- The pipeline only raises a runtime error if *all* sensors failed to produce usable
  output for that flight line. Otherwise, partial success is allowed and the
  pipeline continues.

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

