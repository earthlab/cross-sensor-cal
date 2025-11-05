> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
# Troubleshooting (by symptom)

| Symptom | Likely cause | Fix |
|---|---|---|
| No correction JSON written | Stage 3 didn’t run or wrong paths | Re-run Stage 3; verify `..._brdfandtopo_corrected_envi.json` exists |
| Convolution error: wavelengths | Missing/garbled HDR wavelength block | Regenerate ENVI or fix header; re-run Stage 4 |
| QA fails: ENVI reader missing | Helper not installed in env | Install QA helper; run Stage 7 again |
| OOM / slow export | Too many workers or large chunks | Reduce `--max-workers`; use `--chunksize` |
| Parquet magic bytes / skipping invalid parquet | Parquet sidecar truncated, empty, or not parquet | Re-run the pipeline so export rebuilds it automatically; optionally remove the file and run `bin/validate_parquets --soft <dir>` |
| QA issues: “Wavelengths not strictly increasing” | ENVI header wavelengths out of order or missing | Re-export header from HyTools/ENVI; ensure `wavelength` block is numeric and ordered |
| QA issues: “X% negative pixels” | Mask gaps or incorrect brightness/BRDF parameters | Inspect mask rasters, rerun `cscal-qa --full`, adjust brightness/BRDF inputs |
| High RMSE/SAM in QA JSON | Resampled cube misaligned with corrected cube | Confirm sensor SRFs, regenerate `*_convolved_envi` / `*_resampled_<sensor>_envi` products |

See also: [Stages](pipeline/stages.md) • [Outputs](pipeline/outputs.md)
<!-- FILLME:END -->
