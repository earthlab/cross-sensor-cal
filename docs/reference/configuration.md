# Configuration

> **When do I need this?** When you want to override defaults for correction/convolution or run headless in CI. Includes examples and links to producing/consuming stages.

## Purpose
Explain optional overrides that change how [Pipeline Stages](../pipeline/stages.md) behave—especially Stage 3 (correction), Stage 4 (convolution), and Stage 6 (merge).

## Inputs
- YAML or JSON files loaded via `--config`
- Environment variables read by orchestrators
- Inline CLI overrides such as `--sensor` or `--max-workers`

## Outputs
A normalized configuration object passed into each stage; affects artifact naming in [Outputs](../pipeline/outputs.md).

## Brightness configuration

Brightness adjustments between sensor systems (e.g., Landsat→MicaSense) are
defined via small JSON files shipped with the package:

- Location: `cross_sensor_cal/data/brightness/*.json`
- Loader: `cross_sensor_cal.load_brightness_coefficients(system_pair)`

Example (`landsat_to_micasense.json`):

```json
{
  "system_pair": "landsat_to_micasense",
  "unit": "percent",
  "bands": {
    "1": -7.40,
    "2": -2.75,
    "3": -6.94,
    "4": -10.12,
    "5": -6.65,
    "6": -2.74,
    "7": -1.11
  }
}
```

These coefficients are applied to Landsat convolution products as:
`L_adj = L_raw * (1 + coeff / 100)`
and are surfaced in:

- The brightness summary table (per-band values).
- The QA JSON (`brightness_coefficients`).
- The multi-page QA PDF (Page 3).

### How to change coefficients

1. Create a new JSON file under `cross_sensor_cal/data/brightness/`.
2. Set `"system_pair"` and a `"bands"` mapping of 1-based band indices to
   percent adjustments.
3. Call `load_brightness_coefficients("<your_system_pair>")` from your
   pipeline extension or configuration.

## Run it
```bash
cscal-pipeline --base-folder out --config configs/niwo_gpu.yaml \
  --site-code NIWO --year-month 2023-08 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance
```

```python
from cross_sensor_cal.config import load_config

cfg = load_config("configs/niwo_gpu.yaml")
print(cfg.pipeline.stages["convolution"].sensors)
```

## Pitfalls
- Missing sensor definitions lead to empty Stage 4 outputs—double-check `sensors:` blocks.
- Keep path references relative to the working directory used during Stage 2–7 execution.
- For CI, pin random seeds where applicable to keep QA metrics stable.
