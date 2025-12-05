# Tutorial: Cloud & HPC Workflows

This tutorial describes how to run the cross-sensor-cal pipeline efficiently in cloud or HPC environments where data access, storage, and memory constraints differ from a local workstation.

---

## Overview

You will learn:

- best practices for running the pipeline in object-storage environments  
- when to use thread mode vs. Ray mode  
- how to work with large NEON datasets without local persistence  
- strategies for scaling multi-flightline workflows  

---

## 1. Working with object storage (e.g., CyVerse)

NEON HDF5 tiles can be accessed using `gocmd` or iRODS commands.

Recommended workflow:

1. Stage a small number of HDF5 tiles into a temporary working directory  
2. Run the pipeline on those tiles  
3. Upload corrected ENVI + Parquet outputs to persistent storage  
4. Clean intermediate files to save space  

Example staging command:

```bash
gocmd get i:/iplant/home/.../NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance.h5 .
2. Engine selection: threads vs Ray
Thread engine
Best for:
small to medium flight lines
machines with limited memory
single-tile debugging
Ray engine
Best for:
many flight lines
distributed cloud environments
parallel extraction and merging
Enable Ray:
pip install cross-sensor-cal[ray]
Run:
cscal-pipeline ... --engine ray
3. Memory considerations
Large NEON flight lines may be tens of gigabytes. To avoid memory pressure:
reduce chunk sizes
use --max-workers conservatively
prefer per-tile processing rather than large batch merging
avoid keeping many ENVI cubes in memory simultaneously
In Ray mode, ensure worker memory matches expected tile size (e.g., 16–32 GB per worker).
4. Recommended HPC workflow
Submit one job per flight line
Request enough memory for a single NEON tile (~20–40 GB)
Use local scratch storage for temporary files
Upload final ENVI + Parquet products to shared storage
Merge results downstream using DuckDB or Python
This scales cleanly across sites and years.
5. Example SLURM script
#!/bin/bash
#SBATCH --job-name=cscal
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load python

BASE=$SCRATCH/cscal_${SLURM_JOB_ID}
mkdir -p "$BASE"

cscal-pipeline \
  --base-folder "$BASE" \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --engine thread \
  --max-workers 8
6. Next steps
Pipeline stages
Using Parquet outputs
Troubleshooting

---
