<!-- HOME:START -->
# Cross-Sensor Calibration

Turn raw **NEON hyperspectral** flight lines into corrected ENVI cubes, sensor-matched products, tidy Parquet, and a QA panel—**in one restart-safe command**.

## Run your first tile (3 steps)

```bash
# 1) Create an output base
BASE=output_demo && mkdir -p "$BASE"

# 2) Process one flight line (replace IDs as needed)
cscal-pipeline \
  --base-folder "$BASE" \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2

# 3) Open the QA image
open "$BASE/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance_qa.png"
```

New here? See the [Quickstart](quickstart.md).

Something broke? Jump to [Troubleshooting](troubleshooting.md).
<!-- HOME:END -->

---

## Why teams use cross-sensor-cal

- **Restart-safe orchestration** – rerun the same command after a crash; completed stages are skipped automatically.
- **Consistent artifacts** – every tile produces ENVI cubes, tidy Parquet, and a QA panel named predictably.
- **Python & CLI parity** – drive the pipeline from scripts or automate it in schedulers with the same arguments.

## Install in minutes

Choose a clean virtual environment (see the detailed [Environment](env.md) page for Conda tips and GDAL notes).

=== "venv + PyPI"
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -U pip
    pip install cross-sensor-cal
    ```

=== "From source"
    ```bash
    git clone https://github.com/earthlab/cross-sensor-cal.git
    cd cross-sensor-cal
    python -m venv .venv && source .venv/bin/activate
    pip install -U pip
    pip install -e .[dev]
    ```

## Understand the pipeline

1. **Download NEON HDF5** tiles.
2. **Export ENVI** cubes and apply **topographic + BRDF correction**.
3. **Convolve** to other sensors, **flatten to Parquet**, and finish with a **QA panel**.

See the [Pipeline Stages](pipeline/stages.md) overview for purpose, inputs, outputs, and pitfalls at every step.

## Next steps

| Goal | Start with |
| --- | --- |
| Rerun the full flow on multiple tiles | [CLI & examples](usage/cli.md) |
| Inspect Parquet outputs in notebooks | [Parquet preview](usage/parquet_preview.md) |
| Tune configuration for HPC or CI | [Reference → Configuration](reference/configuration.md) |
| Validate artifacts and schemas | [Reference → Schemas](reference/schemas.md) & [Reference → Validation](reference/validation.md) |
| Extend to a new sensor or reader | [Reference → Extending](reference/extending.md) |
| Something looks off | [Troubleshooting](troubleshooting.md) |

---

## Quality Assurance Overview

All processed tiles automatically undergo a **QA panel** and **JSON validation** stage.

✅ **Good:** reflectance within [0, 1.2], low ΔReflectance, high mask coverage.  
⚠️ **Needs Review:** 1–2 metrics outside thresholds.  
❌ **Fail:** large brightness shifts, missing wavelengths, or high spectral error.

[Read full QA interpretation →](pipeline/qa_panel.md)

## Release highlights

- Per-flightline master table written as **`<prefix>_merged_pixel_extraction.parquet`**
- QA panel **`<prefix>_qa.png`** is emitted **after the merge** during full runs
