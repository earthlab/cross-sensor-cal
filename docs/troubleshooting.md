# Troubleshooting

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Ray `/dev/shm` warning or crash | Shared-memory mount `/dev/shm` is too small for Ray workers | Re-run the container with a larger `--shm-size` (e.g., `--shm-size=8g`) or point `RAY_TMPDIR` at a filesystem with space such as `/tmp` |
| Download fails with HTTP 403 or 404 | NEON API credentials missing or flight line ID incorrect | Verify that the `--flight-lines` values match NEON naming exactly and export `NEON_API_TOKEN` before running `cscal-download` or `cscal-pipeline` |
| “strict filename parsing” errors | Input file names do not match the expected `<site>_<...>_<YYYYMMDD>_<suffix>` pattern | Rename files to conform to the expected pattern or set `strict_filenames: false` in the configuration when working with legacy data |
| “metadata injection” path mismatches | Paths embedded in metadata don't match the actual file layout on disk | Regenerate the metadata using `cscal-pipeline` or edit the JSON so `source_path` and `target_path` fields match the current structure |
| Non-monotonic wavelengths reported in QA JSON | ENVI header `wavelength` values are unsorted or missing | Fix the header ordering or regenerate the ENVI export so wavelengths increase monotonically |
| High `negatives_pct` in QA metrics | Mask coverage is low or BRDF parameters are mismatched | Inspect the mask raster, adjust the ROI, or re-run correction with tuned BRDF/topo inputs |
| Large convolution RMSE in QA panel | Sensor response functions or wavelength units are inconsistent | Confirm the sensor convolution tables and units (nm vs µm) before rerunning the resample stage |
| ENVI header/BSQ mismatches | `.hdr` and `.bsq` files are out of sync or corrupted | Delete the mismatched files and rerun `cscal-pipeline --stage export_envi` (or rerun the full pipeline) to regenerate them |
| Stage stops with `MemoryError` | Worker concurrency exceeds available RAM | Lower `--max-workers`, enable swap, or split the job into smaller batches so only one flight line is active at a time |
| 255 max-value columns in CSVs | Columns stored as unsigned 8-bit integers use 255 as a sentinel | Convert columns to a wider type (e.g., `uint16`) and replace 255 with a nodata value before analytics |
| iRODS transient failures | Temporary network or server issue when communicating with iRODS | Retry the operation after a delay; for automation enable exponential backoff or use the pipeline's built-in retry mechanisms |
| QA panel missing plots | Upstream stage failed or outputs moved after processing | Rerun `cscal-qa --base-folder <dir>` and inspect the logs for missing inputs; fix the upstream stage before regenerating |

### General debugging tips

1. **Check logs first.** Every command prefixes logs with the flight line stem. Search for `❌` markers to spot failing stages.
2. **Validate folder structure.** Each `<base>/<flight_stem>/` folder should include `*_envi.*`, `*_brdfandtopo_corrected_*`,
   resampled outputs, and the merged Parquet file. Missing files usually point to an earlier stage error.
3. **Reproduce with `--verbose`.** Extra logging reveals configuration overrides, discovered files, and Ray task lifecycles.
4. **Capture environment details.** Record Python version, OS, and whether Conda or pip installed GDAL/PROJ when filing an issue.
<!-- FILLME:END -->
