# bin

## Overview
The `bin` directory provides command-line entry points for common processing
tasks. Each script wraps functionality from `src/` so you can run the workflow
without writing your own Python code.

## Prerequisites
- Python 3.10+
- Required libraries installed from `requirements.txt`

## Step-by-step tutorial
1. Display help for the ENVI conversion script:

```bash
python bin/neon_to_envi.py --help
```

2. Apply BRDF and topographic correction from the command line:

```bash
python bin/topo_and_brdf_correction.py config.json
```

## Reference
- `jefe.py` – orchestrates job execution on clusters
- `neon_to_envi.py` – converts NEON HDF5 files to ENVI format
- `topo_and_brdf_correction.py` – applies corrections using generated configs

## Next steps
You can add new scripts here to expose additional modules as CLI tools.

Last updated: 2025-08-14
