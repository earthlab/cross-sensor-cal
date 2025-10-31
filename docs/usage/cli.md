# CLI & Examples

Cross-Sensor Calibration installs several console scripts that cover download, the end-to-end pipeline, QA generation, and
post-processing. Each command logs with a `[cscal-*]` prefix so you can correlate console output with artifacts written to disk.

## Console script overview

| Command | Purpose | Typical usage |
| --- | --- | --- |
| `cscal-download` | Fetch NEON AOP HDF5 flight lines and ancillary files | Prefetch data to a workstation before running the full pipeline |
| `cscal-pipeline` | Run download → ENVI export → correction → cross-sensor convolution → Parquet merge | Primary entry point for production runs |
| `cscal-qa` | Rebuild QA panels for processed flight lines | Spot-check results after editing outputs or merging tables |
| `cscal-qa-dashboard` | Launch an interactive dashboard to browse QA artifacts | Review many flight lines quickly |
| `cscal-qa-metrics` | Summarize QA metrics across flight lines in CSV form | Track trends or regressions over time |
| `cscal-recover-raw` | Restore original HDF5 cubes from staged products | Recover deleted `.h5` files without redownloading |
| `python -m bin.merge_duckdb` | Merge per-stage Parquet tables and emit QA panel | One-off merges, debugging, or custom folder layouts |

## Downloading data only

Use `cscal-download` when you want to stage NEON HDF5 files ahead of time. The command is idempotent—reruns detect existing files
and skip the transfer.

```bash
cscal-download \
  --base-folder data/NIWO_2023-08 \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance
```

Add `--max-workers` to parallelize downloads or `--no-ancillary` to skip optional ancillary products.

## Full pipeline run

`cscal-pipeline` orchestrates every stage and writes outputs to a per-flightline directory. Use the same arguments as the download
command plus any overrides you need (e.g., `--config` to point at a custom YAML file).

```bash
cscal-pipeline \
  --base-folder output_demo \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines \
    NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance \
    NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2 \
  --verbose
```

Key behaviors:

- **Restart safe:** existing stages log `✅ ... (skipping)` and are left untouched.
- **Structured outputs:** artifacts live under `<base>/<flight_stem>/` with canonical names for ENVI, Parquet, JSON, and QA files.
- **Configuration aware:** pass `--config path/to/settings.yaml` to customize sensor lists, resampling options, or QA thresholds.

## QA regeneration and dashboards

After editing Parquet tables or tweaking configuration, regenerate QA panels without rerunning heavy stages:

```bash
# Quick (default) QA regeneration with JSON sidecars
cscal-qa --base-folder output_demo --out-dir qa-panels

# Full sampling pass if you need dense metrics
cscal-qa --base-folder output_demo --full --n-sample 120000
```

Point the dashboard at the same base folder to explore results interactively:

```bash
cscal-qa-dashboard --base-folder output_demo
```

Use `cscal-qa-metrics --base-folder output_demo --out metrics.csv` to summarize per-flightline metrics for monitoring.

## Recovering raw cubes

If the original `.h5` files were deleted but the ENVI exports remain, `cscal-recover-raw` rebuilds them from staged intermediates:

```bash
cscal-recover-raw --flightline-dir output_demo/NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance
```

The command validates that all required ENVI pieces exist before restoring the raw cube to `<flight_stem>.h5` at the base folder.

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
