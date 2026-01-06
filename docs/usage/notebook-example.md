# Start Here: Notebook Workflow

This page is the canonical notebook-first path for running SpectralBridge. It walks through one successful flightline run that produces harmonized, Landsat-referenced outputs on disk—not return values—so you can bridge UAS and NEON observations to the long-term Landsat record without touching the CLI.

---

## 1) Minimal setup

```python
import spectralbridge
from spectralbridge.pipelines.pipeline import (
    process_one_flightline,
    go_forth_and_multiply,
)
```

Use `process_one_flightline` when you are exploring or validating a single flightline. Use `go_forth_and_multiply` when you have a list of flightlines and want the same pipeline applied in batch.

---

## 2) Canonical single-flightline example

```python
from pathlib import Path

# Where all artifacts will be written
base_folder = Path("csc_output")

# Identify the NEON site and acquisition window for clarity
site_code = "NIWO"
year_month = "2023-08"
flightline_id = "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"

# Run the structured, skip-aware pipeline for one flightline
process_one_flightline(
    base_folder=base_folder,
    product_code="DP1.30006.001",  # NEON hyperspectral product
    flight_stem=flightline_id,
)
```

The defaults handle topographic + BRDF correction, spectral convolution into the Landsat frame, brightness adjustment, Parquet export, and QA generation. Re-running the cell will skip stages that are already valid so interrupted sessions can resume safely.

---

## 3) What gets created

A successful run produces a directory named after the flightline inside `base_folder` containing:

- Corrected ENVI reflectance cubes.
- Landsat-resampled ENVI cubes and Parquet sidecars.
- `{flightline_id}_merged_pixel_extraction.parquet` (the master table that merges all Parquet sidecars).
- `{flightline_id}_qa.png` and `{flightline_id}_qa.json` (visual + numeric QA summaries).

Existing, validated outputs are reused on subsequent runs, so partial runs can be resumed without starting over. Seeing both the merged Parquet and QA artifacts is a good signal that the end-to-end pipeline completed.

---

## 4) Inspect results in the notebook

Load the merged Parquet table to confirm the expected columns and a few rows:

```python
import pandas as pd

flight_dir = base_folder / flightline_id
merged_parquet = flight_dir / f"{flightline_id}_merged_pixel_extraction.parquet"

merged_df = pd.read_parquet(merged_parquet)
merged_df.head()
```

Check the available fields to guide downstream analysis:

```python
sorted(merged_df.columns)
```

Locate the QA outputs so you can open them in your notebook or file browser:

```python
qa_png = flight_dir / f"{flightline_id}_qa.png"
qa_json = flight_dir / f"{flightline_id}_qa.json"

qa_png, qa_json
```

Use your notebook environment to display the PNG or parse the JSON for validation details.

---

## 5) What to do next

- Browse notebook [recipes](recipes.md) for batch runs, QA refreshes, and quick exploration patterns.
- Review [Outputs & naming](../pipeline/outputs.md) to understand the file contract and where to find products.
- Read the [Concepts](../concepts/why-calibration.md) section for the rationale behind the Landsat-referenced normalization.
- If you prefer or need the command-line interface, see the [CLI reference](cli.md).
