# CLI & Examples

## Merge (per flightline or batch)
```bash
# Batch: all flightlines under a root
python -m bin.merge_duckdb --data-root /path/to/output --flightline-glob "NEON_*"

# Single flightline directory (debug)
python -m bin.merge_duckdb --flightline-dir /path/to/.../NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance
```
**Defaults**

- Output name: `<prefix>_merged_pixel_extraction.parquet`
- QA panel: enabled (produces `<prefix>_qa.png`)

**Flags**

- `--out-name` override output parquet name (optional)
- `--no-qa` skip panel rendering
- `--original-glob`, `--corrected-glob`, `--resampled-glob` to customize discovery
