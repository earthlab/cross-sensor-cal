# cross-sensor-cal

Cross-Sensor Calibration turns raw [NEON AOP](https://www.neonscience.org/data-collection/airborne-remote-sensing) hyperspectral
flight lines into analysis-ready reflectance, Parquet tables, and QA artifacts in a single restart-safe workflow. The pipeline
covers download, ENVI export, BRDF + topographic correction, cross-sensor convolution, and reporting so teams can focus on
analysis instead of data wrangling.

!!! tip "Looking for the old quickstart?"
    The expanded quickstart now lives at [Quickstart](quickstart.md) with both CLI and Python API walkthroughs.

## Get the package

=== "Install from PyPI"
    ```bash
    python -m venv cscal-env
    source cscal-env/bin/activate  # Windows: cscal-env\Scripts\activate
    pip install --upgrade pip
    pip install cross-sensor-cal
    ```

    This installs the CLI entry points (`cscal-download`, `cscal-pipeline`, `cscal-qa`, and friends) plus the Python API. The
    wheels are published for Linux and macOS on Python 3.10–3.12; other platforms can build from source.

=== "Install from source"
    ```bash
    git clone https://github.com/earthlab/cross-sensor-cal.git
    cd cross-sensor-cal
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .[dev]
    ```

    Source installs expose the latest development features and editable code. See [Environment Setup](env-setup.md) for Conda
    instructions and notes about GDAL/PROJ, Ray shared memory, and documentation previews.

## Process your first flightline

1. Create a base folder where downloads and derived products will be written:
   ```bash
   BASE=output_demo
   mkdir -p "$BASE"
   ```
2. Run the consolidated pipeline on one or more NEON flight lines. Replace placeholders with the site, month, product, and
   flight IDs you need:
   ```bash
   cscal-pipeline \
     --base-folder "$BASE" \
     --site-code NIWO \
     --year-month 2023-08 \
     --product-code DP1.30006.001 \
     --flight-lines \
       NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance \
       NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
     --max-workers 2
   ```
3. Inspect results inside `$BASE/<flight_stem>/`—you should see ENVI exports, corrected cubes, sensor-specific resamples, Parquet
   tables, and the QA panel (`<flight_stem>_qa.png`). Re-run the same command at any time; completed stages log `✅ ... (skipping)`
   and are left untouched.
4. Optional: generate fresh QA panels without rerunning the full pipeline.
   ```bash
   cscal-qa --base-folder "$BASE"
   ```

Prefer a fully scripted flow? The [Quickstart](quickstart.md#4-python-api-equivalent) shows the matching Python API example using
`go_forth_and_multiply()`.

## Troubleshooting & support

- Start with the curated [Troubleshooting](troubleshooting.md) guide for Ray shared memory issues, filename validation errors,
  ENVI mismatches, iRODS hiccups, and more.
- Browse the [FAQ](faq.md) for configuration and workflow questions like resuming runs or adding new sensors.
- Use `--verbose` on CLI commands or read the per-flightline logs inside each output directory to pinpoint stage failures.
- When filing issues, include the console output (with the `[flight_stem]` prefixes) and your configuration overrides to speed up
  triage.

## Explore the documentation

| Topic | Start here |
| --- | --- |
| Step-by-step walkthrough | [Quickstart](quickstart.md) |
| Environment and dependencies | [Environment Setup](env-setup.md) |
| CLI usage and examples | [CLI & Examples](usage/cli.md) |
| Pipeline architecture | [Pipeline Stages](pipeline/stages.md) |
| Outputs and file formats | [Pipeline Outputs](pipeline/outputs.md) and [Schemas](schemas.md) |
| Extending sensors & stages | [Extending the pipeline](extending.md) |
| Validation and QA | [QA panel](pipeline/qa_panel.md), [Validation checks](validation.md) |

## Release highlights

- Per-flightline master table written as **`<prefix>_merged_pixel_extraction.parquet`**
- QA panel **`<prefix>_qa.png`** is emitted **after the merge** during full runs

root@d52d22c64421:/workspace/cross-sensor-cal#
