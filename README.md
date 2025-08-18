<!-- FILLME:START -->
# Cross-Sensor Calibration

Cross-Sensor Calibration provides a Python pipeline for processing NEON Airborne Observation Platform hyperspectral flight lines and resampling them to emulate alternate sensors in a reproducible, scriptable workflow.

![Pipeline diagram](docs/img/pipeline.png)
<!-- TODO: replace with final diagram -->

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

### Conda

```bash
conda create -n cscal python=3.10 gdal ray hytools
conda activate cscal
pip install -e .
```

### uv/pip

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Documentation

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

## License and Citation

Distributed under the GPLv3 License. Please cite the project using [CITATION.cff](CITATION.cff).

<!-- FILLME:END -->

