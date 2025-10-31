# CLI & Examples

## Full pipeline (one tile)
**Purpose** Run every stage end-to-end for a single flight line.

**Inputs** Base folder, NEON site/month/product, flight line IDs.

**Outputs** Per-flight directory with ENVI exports, corrected/convolved cubes, Parquet tables, merged parquet, QA panel.

**Run it**
```bash
cscal-pipeline --base-folder out --site-code NIWO --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2
```
**Pitfalls** High worker counts can exhaust `/dev/shm`; rerun with the same arguments to resume safely.

## Convolve only
**Purpose** Resample corrected ENVI cubes to a target sensor.

**Inputs** Corrected ENVI `.img` files, target sensor code, output folder.

**Outputs** `_brdfandtopo_corrected_envi_<sensor>.img/.hdr` for each input cube.

**Run it**
```bash
cross-sensor-cal convolve --in corrected/*_brdfandtopo_corrected_envi.img --sensor OLI --out convolved/
```
**Pitfalls** Ensure corrected HDR files include valid wavelength metadata.

## Export parquet only
**Purpose** Flatten ENVI products into tidy Parquet files.

**Inputs** One or more ENVI `.img` paths (corrected and/or convolved) and an output directory.

**Outputs** One Parquet per ENVI input plus logs describing chunk sizes.

**Run it**
```bash
cross-sensor-cal export-parquet --in corrected/*.img convolved/*.img --out parquet/
```
**Pitfalls** Use `--chunksize` when RAM is limited.

## Merge + QA
**Purpose** Consolidate product Parquet tables and rebuild QA panels.

**Inputs** Set of Parquet files, target merged filename, optional QA output directory.

**Outputs** `<prefix>_merged_pixel_extraction.parquet` and `<prefix>_qa.png`.

**Run it**
```bash
cross-sensor-cal merge-duckdb --in parquet/*.parquet --out merged/demo_merged_pixel_extraction.parquet
cross-sensor-cal qa-panel --merged merged/demo_merged_pixel_extraction.parquet --out qa/
```
**Pitfalls** `merge-duckdb` expects consistent schemas; if QA fails, check [Troubleshooting](../troubleshooting.md).

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
