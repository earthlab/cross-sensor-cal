# Command-Line Interface (CLI)

The cross-sensor-cal CLI provides a unified, restart-safe workflow for processing NEON flight lines from raw HDF5 to corrected and sensor-harmonized ENVI/Parquet outputs. This page documents the primary commands, flags, and usage patterns.

---

## Primary command: `cscal-pipeline`

This is the recommended entry point for processing NEON hyperspectral data.

### Example

```bash
cscal-pipeline \
  --base-folder output \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --engine thread \
  --max-workers 4
Required arguments
FlagDescription
--base-folderRoot directory for all outputs
--site-codeNEON site code (e.g., NIWO, KONZ)
--year-monthAcquisition month in YYYY-MM format
--product-codeNEON product code (usually DP1.30006.001)
--flight-linesOne or more NEON flight line identifiers
Optional arguments
FlagDescription
--engine {thread,ray}Execution engine
--max-workersNumber of parallel workers
--start-atStage to begin the pipeline
--end-atStage to stop the pipeline
--cleanRemove intermediate files
--dry-runPrint what would be run without executing
Pipeline stages
You may specify partial workflows:
cscal-pipeline \
  --start-at brdf \
  --end-at convolution
Valid stages:
download
export-envi
topo
brdf
convolution
parquet
qa
Engine selection
Thread engine (default)
Suitable for local machines and moderate-sized flight lines.
Ray engine
Install:
pip install cross-sensor-cal[ray]
Run:
cscal-pipeline --engine ray ...
Use Ray for:
batch processing of many flight lines
distributed cloud/HPC environments
large-scale Parquet extraction
Inspecting available options
cscal-pipeline --help
This prints detailed descriptions for every flag.
Other CLI tools
Brief mention (documented elsewhere)
cscal-micasense-to-landsat
debugging utilities (internal / experimental)
These experimental tools may change as the package evolves.
Next steps
Working with Parquet outputs
Pipeline stages

---
