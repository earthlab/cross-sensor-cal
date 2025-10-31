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
