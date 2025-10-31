# Validation

> **When do I need this?** When a stage fails or a QA smell appears; validate inputs/outputs against known-good schema.

## Purpose
Provide targeted checks for [Pipeline Stages](../pipeline/stages.md) that frequently fail—especially Stage 3 corrections and Stage 6 merges.

## Inputs
- Paths to ENVI `.img/.hdr` or Parquet files from [Outputs](../pipeline/outputs.md)
- Schema definitions or expected ranges for QA metrics

## Outputs
Console reports or CSV summaries highlighting missing bands, mismatched wavelengths, or schema drift.

## Run it
```bash
python scripts/check_envi_headers.py corrected/*_brdfandtopo_corrected_envi.hdr
python scripts/validate_schema.py merged/demo_merged_pixel_extraction.parquet schemas/merged_schema.json
```

```python
from cross_sensor_cal.validation import validate_parquet

validate_parquet("merged/demo_merged_pixel_extraction.parquet", strict=True)
```

## Pitfalls
- Skipping validation can hide silent failures; automate checks in CI before trusting Stage 7 QA images.
- Keep Ray workers pinned to the same package version to avoid mixed schema outputs.
- When QA histograms look wrong, verify both the input parquet and the ENVI wavelength metadata.
