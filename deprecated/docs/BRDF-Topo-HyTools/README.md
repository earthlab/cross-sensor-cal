# BRDF-Topo-HyTools

## Overview
This directory packages helper resources for applying BRDF and topographic
corrections with the HyTools library. You can use it to reproduce the
processing steps implemented in the `src/topo_and_brdf_correction.py` module.

## Prerequisites
- Python 3.10+
- `hytools` installed and available on your `PYTHONPATH`
- NEON hyperspectral data in `.h5` format

## Step-by-step tutorial
1. Place the NEON flightline `.h5` files in your working directory.
2. Generate configuration files using the main pipeline:

```python
from src.topo_and_brdf_correction import generate_config_json

generate_config_json("data/NIWO_2023_08")
```

3. Run the HyTools-based correction scripts contained here to apply BRDF and
topographic adjustments.

## Reference
- `Topo-and-Brdf-Corr/` – legacy HyTools correction scripts bundled for
  reproducibility.

## Next steps
After running the corrections, return to `src/topo_and_brdf_correction.py` to
continue the pipeline.

Last updated: 2025-08-14
