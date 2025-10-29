# Stage 02 Sorting

> **Deprecated:** Sorting is now handled by `generate_file_move_list()` and the
> optional stage 8 helpers inside `src/cross_sensor_cal/pipelines/pipeline.py`.
> This legacy write-up referenced missing CLI scripts and GeoTIFF outputs.

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
## Overview
Stage 02 sorts your raster products into a canonical folder tree and normalizes
flight line identifiers for downstream analysis. You run it after raster
processing to prepare data for cross‑sensor comparisons. The sort is idempotent;
rerunning the command recreates the same layout without duplicating files.

## Prerequisites
- Outputs from [Stage 01 Raster Processing](stage-01-raster-processing.md)
- Local workspace containing NEON flight line folders
- Access to an iRODS account and `gocmd` installed for syncing
- (Optional) project‑specific remote path prefix

## Step-by-step tutorial
1. Generate the move list and optionally sync files:

   ```bash
   python bin/jefe.py sort-and-sync-files \
       --base-folder /path/to/niwo_neon \
       --remote-prefix projects/neon/data \
       --sync-files
   ```

   The script normalizes flight line names (lowercase, zero padded) and builds
   `sorted_files/envi/<category>/` folders.

2. Inspect `envi_file_move_list.csv` in the base folder. This local cache stores
   source and destination paths so later runs can resume without rescanning.

3. Paths that sync to iRODS follow
   `i:/iplant/<remote_prefix>/sorted_files/envi/<category>/<filename>`.
   Omitting `--remote-prefix` places files under `i:/iplant/` directly.

4. Rerun the command at any time. `gocmd sync` compares timestamps and sizes, so
   reruns skip unchanged files and resume incomplete transfers.

## Reference
 - [`src/file_sort.py`](https://github.com/earthlab/cross-sensor-cal/blob/main/src/file_sort.py) implements sorting and iRODS path rules
 - [`bin/jefe.py`](https://github.com/earthlab/cross-sensor-cal/blob/main/bin/jefe.py) provides the `sort_and_sync_files` entry point
 - [CyVerse iRODS](cyverse-irods.md) covers authentication and environment setup

## Next steps
Proceed to [Stage 03 Pixel Extraction](stage-03-pixel-extraction.md) once files
are sorted and synced.
<!-- FILLME:END -->
Last updated: 2025-08-18
