> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
# Troubleshooting (by symptom)

| Symptom | Likely cause | Fix |
|---|---|---|
| No correction JSON written | Stage 3 didn’t run or wrong paths | Re-run Stage 3; verify `..._brdfandtopo_corrected_envi.json` exists |
| Convolution error: wavelengths | Missing/garbled HDR wavelength block | Regenerate ENVI or fix header; re-run Stage 4 |
| QA fails: ENVI reader missing | Helper not installed in env | Install QA helper; run Stage 7 again |
| OOM / slow export | Too many workers or large chunks | Reduce `--max-workers`; use `--chunksize` |

See also: [Stages](pipeline/stages.md) • [Outputs](pipeline/outputs.md)
<!-- FILLME:END -->
