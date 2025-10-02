# Cross-Sensor Calibration

[![CI](https://github.com/earthlab/cross-sensor-cal/actions/workflows/ci.yml/badge.svg)](https://github.com/earthlab/cross-sensor-cal/actions/workflows/ci.yml)
[![Docs](https://github.com/earthlab/cross-sensor-cal/actions/workflows/gh-pages.yml/badge.svg)](https://earthlab.github.io/cross-sensor-cal/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/TODO.svg)](https://zenodo.org/)

Cross-Sensor Calibration provides a Python toolkit for harmonizing National Ecological
Observatory Network (NEON) airborne hyperspectral observations across multiple target
sensors. The library automates downloading, radiometric corrections, resampling, and
quality checks so remote-sensing scientists can produce comparable reflectance products
across instruments and campaigns.

![Pipeline diagram showing the major processing stages](docs/img/pipeline.png)

## Key features

- 🛰️ **Sensor harmonization** – resample NEON hyperspectral pixels to alternate
  satellite and airborne sensor response functions.
- 🧪 **Processing pipeline** – scripted workflow for downloading, masking,
  topographic + BRDF corrections, and spectral library creation.
- 📦 **Reusable components** – composable functions for each stage exposed via the
  `cross_sensor_cal` Python package.
- 📚 **Comprehensive documentation** – MkDocs site covering architecture, CLI usage,
  troubleshooting, and developer guidance.

## Quickstart

```bash
# Create and activate a virtual environment (example uses uv, but any tool works)
uv venv
source .venv/bin/activate

# Install the package with development extras
uv pip install -e .[dev]

# Run the synthetic end-to-end example
python examples/basic_calibration_workflow.py
```

For conda users, we provide a lightweight environment definition in
[`environment.yml`](environment.yml).

## Minimal example

The repository ships with a synthetic calibration workflow that exercises the
core API surface without large data dependencies. Run it directly from the repo
root:

```bash
python examples/basic_calibration_workflow.py
```

The script prints before/after statistics that demonstrate how the synthetic
observations are aligned. See [`examples/basic_calibration_workflow.py`](examples/basic_calibration_workflow.py)
for annotated code and [`docs/examples/basic-calibration-workflow.md`](docs/examples/basic-calibration-workflow.md)
for narrative context. Additional real-world usage patterns are documented in
the [MkDocs User Guide](docs/user-guide.md).

## Installation

Cross-Sensor Calibration targets Python 3.9 or newer and depends on widely used
scientific Python packages (`numpy`, `pandas`, `rasterio`, `spectral`, `hytools`,
etc.). Install from source with:

```bash
pip install -e .
```

Advanced GDAL/PROJ configuration tips live in the
[installation guide](docs/env-setup.md). When the project publishes wheels to
PyPI the command will simply be `pip install cross-sensor-cal`.

## Documentation

The full documentation site is published at
[earthlab.github.io/cross-sensor-cal](https://earthlab.github.io/cross-sensor-cal/)
with sections for overview material, stage-by-stage tutorials, developer notes,
and an API reference powered by `mkdocstrings`.

## What is cross-sensor calibration?

Cross-sensor calibration ensures measurements collected by different remote
sensing instruments can be compared directly. NEON flight line spectra are
resampled and corrected so that downstream ecological analyses can combine
observations from sensors with varying spectral responses, view angles, and
illumination conditions. This repository provides reusable Python building
blocks for those harmonization steps.

## Citing this software

Please cite the software using the metadata in [`CITATION.cff`](CITATION.cff).
A minimal BibTeX entry looks like:

```bibtex
@software{cross-sensor-cal,
  author    = {{TODO: Replace with project authors}},
  title     = {Cross-Sensor Calibration},
  year      = {2024},
  publisher = {Earth Lab},
  url       = {https://github.com/earthlab/cross-sensor-cal}
}
```

A DOI-backed citation will be provided after the first Zenodo archived release.

## Contributing and community standards

We welcome issues and pull requests! Please review the
[Code of Conduct](CODE_OF_CONDUCT.md) and [Contributing guidelines](CONTRIBUTING.md)
before getting started. For larger features, open a GitHub Discussion or issue
to align on scope before implementation. The repository follows Semantic
Versioning and enforces automated formatting, linting, type checking, and tests
in continuous integration.

## Support and maintenance

The project is maintained by the Earth Lab remote sensing team. We triage new
issues on a monthly cadence. If you rely on the software in published work,
please let us know—citations and success stories help justify continued
maintenance. Commercial support inquiries can be directed to the maintainers via
GitHub issues or email (**TODO: add contact email**).

## License

This project is distributed under the terms of the [GNU General Public License
v3.0](LICENSE). Ensure that your use is compatible with GPL requirements when
embedding the code into larger systems.

## Acknowledgements

Development is supported by the National Ecological Observatory Network (NEON)
and Earth Lab collaborators. Additional acknowledgements, including funding
sources, will be listed in the upcoming JOSS manuscript.

