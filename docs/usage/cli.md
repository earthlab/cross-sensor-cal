# Command-Line Interface (CLI)

The CLI provides restart-safe entry points for downloading NEON flightlines, running the full pipeline, generating QA outputs, and merging Parquet artefacts.

---

## `cscal-download`

Purpose: Download NEON HDF5 flightlines into a workspace (`cross_sensor_cal.cli:download_main`).

**Usage:**

```bash
cscal-download
```

**Key options:**
- `site` (positional) and `--year-month` (required) identify the NEON flightlines.
- `--flight` (repeatable) lists flightline IDs.
- `--product` sets the NEON product code (default `DP1.30006.001`).
- `--output` controls the destination directory (default `data`).

**Outputs:** HDF5 files saved under `<output>/<site>/` for later `cscal-pipeline` runs.

---

## `cscal-pipeline`

Purpose: Run the full cross-sensor pipeline (`cross_sensor_cal.cli.pipeline_cli:main`).

**Usage:**

```bash
cscal-pipeline
```

**Key options:**
- Required: `--base-folder`, `--site-code`, `--year-month`, `--product-code`, `--flight-lines`.
- Resampling: `--resample-method` (`convolution`, `legacy`, or `resample`).
- Performance: `--engine` (`thread`, `process`, `ray`), `--max-workers` (defaults to 8), `--parquet-chunk-size`.
- Merge tuning: `--merge-memory-limit`, `--merge-threads`, `--merge-row-group-size`, `--merge-temp-directory`.
- Radiometry: `--brightness-offset` for ENVI export.

**Outputs:** Everything listed in [Outputs](../pipeline/outputs.md), including `_merged_pixel_extraction.parquet` and `_qa.png`.

---

## `cscal-qa`

Purpose: Re-render QA panels/metrics for existing flightline folders (`cross_sensor_cal.cli.qa_cli:main`).

**Usage:**

```bash
cscal-qa
```

**Key options:**
- `--base-folder` (required) points to the workspace containing flightline directories.
- Sampling: `--quick` (25k deterministic sample) vs `--full` (use configured size), `--n-sample` to override pixel count.
- `--rgb-bands` to override RGB band selection.
- `--save-json` to toggle writing `<flight_id>_qa.json`.
- `--out-dir` to copy PNG/JSON outputs elsewhere after generation.

**Outputs:** QA PNG/JSON (and PDF if rendered) following the naming in [Outputs](../pipeline/outputs.md).

---

## `cscal-recover-raw`

Purpose: Backfill raw ENVI exports when corrected products already exist (`cross_sensor_cal.cli.recover_cli:main`).

**Usage:**

```bash
cscal-recover-raw
```

**Key options:**
- `--base-folder` (required) where HDF5 and flightline folders live.
- `--brightness-offset` to pass through to `stage_export_envi_from_h5`.

**Outputs:** Restores `<flight_id>_envi.(img|hdr)` so later stages can proceed or be revalidated.

---

## `cscal-qa-dashboard`

Purpose: Aggregate QA metrics across flightlines and build an overview plot (`cross_sensor_cal.qa_dashboard:main`).

**Usage:**

```bash
cscal-qa-dashboard
```

**Key options:**
- `--base-folder` (required) root containing flightline outputs.
- `--out-parquet` to choose the aggregated summary parquet path.
- `--out-png` to choose the dashboard PNG path.

**Outputs:** `qa_dashboard_summary.parquet` and `qa_dashboard_summary.png` (default locations inside the base folder) summarizing `<flight_id>_qa_metrics.parquet` tables.

---

## `csc-merge-duckdb`

Purpose: Merge per-product Parquet tables into a master parquet and optional QA panel (`cross_sensor_cal.merge_duckdb:main`).

**Usage:**

```bash
csc-merge-duckdb
```

**Key options:**
- `--data-root` (required) and `--flightline-glob` to locate flightline folders.
- `--out-name` to override the default `<prefix>_merged_pixel_extraction.parquet`.
- Input globs: `--original-glob`, `--corrected-glob`, `--resampled-glob`.
- QA and format controls: `--no-qa` to skip PNG, `--write-feather` to emit Feather alongside Parquet.
- Merge tuning: `--merge-memory-limit`, `--merge-threads`, `--merge-row-group-size`, `--merge-temp-directory`.

**Outputs:** Merged parquet (and optional QA PNG/JSON) matching the naming patterns in [Outputs](../pipeline/outputs.md).

---
