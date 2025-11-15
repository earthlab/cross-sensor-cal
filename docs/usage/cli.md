# CLI & Examples

## Full pipeline (one tile)
**Purpose** Run every stage end-to-end for a single flight line.

**Inputs** Base folder, NEON site/month/product, flight line IDs.

**Outputs** Per-flight directory with ENVI exports, corrected/resampled cubes, Parquet tables, merged parquet, QA panel PNG, QA PDF, and QA JSON.

**Run it**
```bash
cscal-pipeline --base-folder out --site-code NIWO --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2 \
  --engine thread  # optional: choose "ray" when the extra dependency is installed
```
**Pitfalls** High worker counts can exhaust `/dev/shm`; rerun with the same arguments to resume safely.

The CLI logs are designed to be affirmative. Messages like:

> 📦 No existing ENVI export detected — creating a new one from source data ...

are normal and indicate that the tool is creating outputs for the first time.
Only `WARNING` or `ERROR` messages require attention.

## Convolve only
**Purpose** Resample corrected ENVI cubes to a target sensor.

**Inputs** Corrected ENVI `.img` files, target sensor code, output folder.

**Outputs** `<prefix>_resampled_<sensor>_envi.img/.hdr` inside per-sensor resample folders.

**Run it**
```bash
cross-sensor-cal convolve --in corrected/*_brdfandtopo_corrected_envi.img --sensor OLI --out convolved/
```
**Pitfalls** Ensure corrected HDR files include valid wavelength metadata.

## Export parquet only
**Purpose** Flatten ENVI products into tidy Parquet files.

**Inputs** One or more ENVI `.img` paths (corrected and/or resampled) and an output directory.

**Outputs** One Parquet per ENVI input plus logs describing chunk sizes.

**Run it**
```bash
cross-sensor-cal export-parquet --in corrected/*.img Convolution_Reflectance_Resample_*/*.img --out parquet/
```
**Pitfalls** Use `--chunksize` when RAM is limited. Existing sidecars are validated on
startup; corrupted files are deleted and regenerated automatically.

## Validate parquet sidecars
**Purpose** Audit `.parquet` sidecars on disk.

**Run it**
```bash
bin/validate_parquets path/to/flightline_dir
bin/validate_parquets --soft path/to/flightline_dir
```
**Behavior** Default mode checks every file with `pyarrow.parquet.read_schema` and exits 1
if any are invalid. `--soft` logs the same issues but exits 0 so callers can treat the
report as a warning.

## Merge + QA
**Purpose** Consolidate product Parquet tables and rebuild QA panels.

**Inputs** Set of Parquet files, target merged filename, optional QA output directory.

**Outputs** `<prefix>_merged_pixel_extraction.parquet`, `<prefix>_qa.png`, `<prefix>_qa.pdf`, and `<prefix>_qa.json`.

**Run it**
```bash
cross-sensor-cal merge-duckdb --in parquet/*.parquet --out merged/demo_merged_pixel_extraction.parquet
cross-sensor-cal qa-panel --merged merged/demo_merged_pixel_extraction.parquet --out qa/
```
**Pitfalls** `merge-duckdb` validates each input once and skips corrupt files with warnings;
it only fails if none of the candidates are usable. Use `--merge-memory-limit`,
`--merge-threads`, or `--merge-row-group-size` if you need to further constrain memory
pressure during the streaming merge. If QA fails, check [Troubleshooting](../troubleshooting.md).

## Minimal Python equivalent
```python
from cross_sensor_cal import go_forth_and_multiply

go_forth_and_multiply(
    base_folder="out",
    site_code="NIWO",
    year_month="2023-08",
    product_code="DP1.30006.001",
    flight_lines=["NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"],
    max_workers=2,
)
```

## QA

The pipeline saves:

- `<prefix>_qa.png` – a compact single-page QA panel.
- `<prefix>_qa.pdf` – a multi-page QA report (ENVI overview, topo/BRDF diagnostics, and additional QA metrics).
- `<prefix>_qa.json` – machine-readable QA metrics, including header, mask, convolution error, and brightness coefficients when applied.
