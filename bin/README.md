# bin

## Overview
The `bin` directory provides command-line entry points for common processing
tasks. Each script wraps functionality from `src/` so you can run the workflow
without writing your own Python code.

## Prerequisites
- Python 3.10+
- Required libraries installed from `requirements.txt`

## Step-by-step tutorial
1. Use the packaged console entry points:

```bash
cscal-download --help
cscal-pipeline --help
```

2. Apply BRDF and topographic correction from the command line when needed:

```bash
python bin/topo_and_brdf_correction.py config.json
```

## Reference
- `topo_and_brdf_correction.py` – applies corrections using generated configs
- `fetch_testdata` – placeholder helper for CI environments

## Next steps
Additional bespoke scripts should live alongside the production CLI entry points.

Last updated: 2025-08-14
