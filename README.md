# Cross-Sensor Calibration

Cross-Sensor Calibration provides a Python pipeline for processing NEON Airborne Observation Platform hyperspectral flight lines and resampling them to emulate alternate sensors in a reproducible, scriptable workflow.

![Pipeline diagram](docs/img/pipeline.png)

## Quickstart

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

## Install

Cross-Sensor Calibration depends on GDAL, PROJ, Ray, and HyTools. We recommend the Conda workflow below because it installs the required native libraries automatically. If you prefer a pure `pip` workflow, install system packages for GDAL/PROJ first (e.g., `brew install gdal` on macOS or `apt-get install gdal-bin libgdal-dev proj-bin` on Debian/Ubuntu).

### HyTools install (choose ONE)

**PyPI (recommended for CI and most dev):**
```bash
python -m pip install -U "pip<25" "setuptools<75" wheel
pip install "hy-tools==1.6.0"
```

**Conda (if you need GDAL/PROJ prebuilt):**
```bash
conda create -n cscal python=3.10 gdal proj hytools -c conda-forge
conda activate cscal
```

> Don’t mix conda `hytools` with PyPI `hy-tools`. In CI we pin PyPI `hy-tools==1.6.0`.

### Conda

```bash
conda create -n cscal python=3.10 gdal proj hytools
conda activate cscal
pip install -e .
```

### uv/pip

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
pip install "hy-tools==1.6.0"
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

| Python | OS            | GDAL | HyTools | Ray |
|--------|---------------|------|--------|-----|
| 3.10+  | Linux, macOS  | 3.4+ | 1.0+   | 2.0+ |

## How to cite

If you use Cross-Sensor Calibration in your research, please cite the project:

```
Earth Lab Data Innovation Team. (2025). Cross-Sensor Calibration (Version 0.1.0) [Software]. University of Colorado Boulder. https://github.com/earthlab/cross-sensor-cal
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## License and Citation

Distributed under the GPLv3 License. Please cite the project using [CITATION.cff](CITATION.cff).

