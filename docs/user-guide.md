# User guide

This guide links together the most common tasks for scientists who are new to
cross-sensor calibration. It complements the step-by-step pipeline chapters and
focuses on reproducible workflows that can be run on a laptop with minimal
setup.

## Installation

1. Create a Python 3.9+ virtual environment using your preferred tool (`uv`,
   `venv`, `conda`, etc.).
2. Install the package in editable mode with optional development tools:

   ```bash
   pip install -e .[dev]
   ```

3. Verify the installation by running the import smoke test:

   ```bash
   python -c "import cross_sensor_cal; print(cross_sensor_cal.__version__)"
   ```

For platform-specific GDAL or PROJ notes consult the
[environment setup guide](env-setup.md).

## Running the pipeline

1. Download the required NEON flight line assets using the
   [`download_neon_flight_lines`](references.md#cross_sensor_cal.envi_download.download_neon_flight_lines)
   helper.
2. Convert the raw inputs to ENVI format with
   [`neon_to_envi`](stage-01-raster-processing.md).
3. Apply topographic and BRDF corrections via
   [`topo_and_brdf_correction`](stage-01-raster-processing.md#topographic-and-brdf-correction).
4. Use [`translate_to_other_sensors`](stage-04-spectral-library.md) to generate
   harmonized reflectance cubes.

Each of these steps is scriptable; see the [`bin/`](https://github.com/earthlab/cross-sensor-cal/tree/main/bin)
utilities for CLI wrappers and the [examples](examples/basic-calibration-workflow.md) section for
lightweight demonstrations.

## Tips for success

- **Work incrementally.** Start with the synthetic example, then scale to small
  subsets of NEON data before launching full campaigns.
- **Track metadata.** The `docs/schemas.md` page defines expected columns and
  attributes at each stage. Validate your outputs early to avoid surprises.
- **Cache downloads.** NEON flight lines can be large; reuse local caches when
  iterating on processing parameters.
- **Ask for help.** File issues or questions on GitHub when anything is unclear.
  We value community feedback and contributions.
