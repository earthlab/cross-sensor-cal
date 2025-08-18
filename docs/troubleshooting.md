# Troubleshooting

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Ray `/dev/shm` warning | Shared-memory mount `/dev/shm` is too small for Ray workers | Re-run container with larger `--shm-size` or set `RAY_TMPDIR` to a directory with more space |
| “strict filename parsing” errors | Input file names do not match expected naming pattern | Rename files to conform to the expected pattern or disable strict parsing in configuration |
| “metadata injection” path mismatches | Paths embedded in metadata don't match actual file layout | Verify and correct paths before running metadata injection |
| ENVI header/BSQ mismatches | `.hdr` and `.bsq` files are out of sync or corrupted | Recreate the header or regenerate the BSQ to ensure matching metadata and data files |
| 255 max-value columns in CSVs | Columns stored as unsigned 8-bit integers use 255 as a sentinel | Convert columns to a wider type and replace 255 with a nodata value |
| iRODS transient failures | Temporary network or server issue when communicating with iRODS | Retry the operation after a delay; consider using built-in retry mechanisms |
<!-- FILLME:END -->
