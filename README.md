# Cross-Sensor Calibration

Cross-Sensor Calibration provides a Python pipeline for processing NEON Airborne Observation Platform hyperspectral flight lines and resampling them to emulate alternate sensors in a reproducible, scriptable workflow.

![Pipeline diagram](docs/img/pipeline.png)

## 5-minute Quickstart

```bash
# 1. set up workspace
mkdir -p data/SITE

# 2. download a tiny flight line (placeholders shown)
python -m bin.download --site SITE --flight FLIGHT_LINE

# 3. run the full pipeline
python -m bin.pipeline data/SITE/FLIGHT_LINE

# 4. peek at resampled outputs
ls data/SITE/FLIGHT_LINE/resampled
```

Replace `SITE` with a NEON site code and `FLIGHT_LINE` with an actual line identifier.

## Install

Cross-Sensor Calibration depends on GDAL, PROJ, Ray, and HyTools. HyTools is a required dependency and must be installed at a known-good version to avoid circular import failures inside the correction stages. The 1.6.1 release currently publishes reliable wheels for Python 3.9, so start with a Python 3.9 environment unless you have confirmed newer interpreters locally. We recommend the Conda workflow below because it installs the required native libraries automatically. If you prefer a pure `pip` workflow, install system packages for GDAL/PROJ first (e.g., `brew install gdal` on macOS or `apt-get install gdal-bin libgdal-dev proj-bin` on Debian/Ubuntu).

### Conda

```bash
conda create -n cscal python=3.9 gdal proj
conda activate cscal
pip install -U pip
pip install -e . -c constraints/lock-hytools.txt
```

### uv/pip

```bash
uv python install 3.9
uv venv --python 3.9
source .venv/bin/activate
pip install -U pip
pip install -e . -c constraints/lock-hytools.txt
```

If GDAL wheels are unavailable for your platform, install it from Conda-forge and then point `pip` at the Conda environment by exporting `CPLUS_INCLUDE_PATH` and `C_INCLUDE_PATH`.

### Troubleshooting HyTools

If HyTools fails to import or BRDF/TOPO stages crash immediately:

1. Reinstall the project with the pinned constraints:

   ```bash
   pip install -U pip
   pip install -e . -c constraints/lock-hytools.txt
   ```

2. Verify the active environment reports `hy-tools 1.6.1`, `numpy 1.26.4`, and `h5py 3.10.0` while running on Python 3.9.
3. Confirm ancillary ENVI files referenced in the config JSON exist on disk.
4. Check that NEON metadata files include the expected Reflectance/Metadata groups and CRS fields.

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
| 3.9    | Linux, macOS  | 3.4+ | 1.6.1  | 2.0+ |

## How to cite

If you use Cross-Sensor Calibration in your research, please cite the project:

```
Earth Lab Data Innovation Team. (2025). Cross-Sensor Calibration (Version 0.1.0) [Software]. University of Colorado Boulder. https://github.com/earthlab/cross-sensor-cal
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## License and Citation

Distributed under the GPLv3 License. Please cite the project using [CITATION.cff](CITATION.cff).

