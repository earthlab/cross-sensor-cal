# SpectralBridge — Translating reflectance across sensors and scales

**SpectralBridge (formerly cross-sensor-cal)** is a modular Python-based tool that adjusts fine-resolution (few centimeters to ~ 5 meters) “pure” spectra from airborne imaging spectroscopy (IS) and uncrewed aerial system (UAS) multispectral imagery to match the spectral configurations of moderate-resolution satellite sensors (over 30 meters).

![[docs/img/pipeline.png](https://github.com/earthlab/spectralbridge/blob/main/docs/EL_workflow_diagram.png)](https://github.com/earthlab/spectralbridge/blob/main/docs/EL_workflow_diagram_updatedQA.png)

## Environment setup

Option A (conda, recommended for GDAL/rasterio users):

```bash
conda env create -f environment.yaml
conda activate spectralbridge
```

Option B (pip, lightweight):

```bash
python -m venv spectralbridge-env
source spectralbridge-env/bin/activate  # or Windows equivalent
pip install .
```

We test on Python 3.10 in GitHub Actions (Linux x86_64) and periodically validate
Python 3.11 builds on the same platform. Other operating systems may work, but Linux
is the documented baseline.

## Quickstart (CLI)

Run the full SpectralBridge pipeline on one or more NEON flight lines:

```bash
spectralbridge-pipeline \
  --base-folder output_demo \
  --site-code NIWO \
  --year-month 2023-08 \
  --product-code DP1.30006.001 \
  --flight-lines NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance \
                 NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
  --max-workers 2
```

Legacy `cscal-*` commands remain available for now and forward to the same implementation.

This will:

- Download each NEON `.h5` flight line (with a live download progress bar).
- Convert the cube to ENVI using canonical `<flight_stem>_envi.img/.hdr` names.
- Compute and apply BRDF + topographic correction.
- Convolve to multiple sensor bandpasses (Landsat TM/ETM+/OLI, OLI2, Micasense, etc.).
- Export reflectance products and per-sensor tables to ENVI + Parquet.

Output structure:

```text
output_demo/
    <flight_stem>.h5                       # raw NEON flightline (safe to delete later)
    <flight_stem>/                         # all derived products for that line
        <flight_stem>_envi.img/.hdr/.parquet
        <flight_stem>_brdfandtopo_corrected_envi.img/.hdr/.json/.parquet
        <flight_stem>_landsat_tm_envi.img/.hdr/.parquet
        ...
        <flight_stem>_merged_pixel_extraction.parquet
                                        # Master pixel table (original + corrected + resampled)
        <flight_stem>_qa.png             # QA summary figure (auto-generated after merge)
```

QA panels are emitted automatically at the end of the merge stage. You can
re-generate them on demand with:

```bash
spectralbridge-qa --base-folder output_demo
```

That command re-renders `<flight_stem>_qa.png` inside each per-flightline folder.

### Parallel execution from the CLI

- `--max-workers N` (defaults to `2`) bounds parallelism.
- `--engine {thread,process,ray}` selects the parallel backend. Ray is only
  loaded when explicitly requested and requires the optional dependency.
- Each worker processes one flight line in isolation inside its own subdirectory.
- Logs from each worker are prefixed with the flight line ID for readability.
- Memory warning: each hyperspectral cube can consume tens of GB in memory, so avoid
  setting `--max-workers` higher than your hardware can support.

## Quickstart (Python API)

> As of v2.2 the pipeline automatically downloads NEON HDF5 cubes, streams live progress
> bars, and writes every derived product into a dedicated per-flightline folder.

Install the base package (threads/process execution; Ray remains optional):

```bash
pip install spectralbridge
```

> Need Ray-backed execution? Install the optional extra instead:
> `pip install "spectralbridge[full]"`. The default engine uses threads so
> Ray is not required for typical workflows.

> Upgrading from older versions? Imports under ``cross_sensor_cal`` continue to
> work for now, but new examples use the ``spectralbridge`` namespace.

### Quickstart Example

```python
from spectralbridge.pipelines.pipeline import go_forth_and_multiply
from pathlib import Path

go_forth_and_multiply(
    base_folder=Path("output_fresh"),
    site_code="NIWO",
    year_month="2023-08",
    product_code="DP1.30006.001",
    flight_lines=[
        "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance",
        "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance",
    ],
    max_workers=2,  # Run flightlines in parallel
)
```

This automatically:

- Downloads the required NEON HDF5 flightlines with live progress bars.
- Converts each cube to ENVI with per-tile progress updates.
- Builds BRDF + topo correction JSON.
- Applies corrections, performs cross-sensor convolution, and exports Parquet tables.
- Automatically clears memory between major steps to prevent Ray OOM crashes.
- Writes every derived product into a per-flightline subfolder while leaving the raw
  `.h5` next to it for easy cleanup.

### Example Output Layout

```text
output_fresh/
├── NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance.h5
├── NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance/
│   ├── ..._envi.img/.hdr/.parquet
│   ├── ..._brdfandtopo_corrected_envi.img/.hdr/.json/.parquet
│   ├── ..._landsat_tm_envi.img/.hdr/.parquet
│   ├── ..._micasense_envi.img/.hdr/.parquet
│   └── NIWO_brdf_model.json
├── NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance.h5
└── NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/
    └── <same pattern>
```

### Reproducibility guarantees

- **Deterministic staging:** Each step writes files with canonical names. Downstream
  steps never guess paths.
- **Checkpointing / idempotence:** Stages skip work if valid outputs already exist
  (`✅ ... already complete ... (skipping)`).
- **Crash-safe restarts:** You can re-run `spectralbridge-pipeline` on the same folder after an
  interruption; it will resume from what's missing.
- **Per-flightline isolation:** Each flight line has its own subdirectory. This allows
  parallel execution and makes it clear which outputs belong together.
- **Ephemeral HDF5:** The original NEON `.h5` stays at the top level and can be deleted
  later if you only want corrected/derived products.
- **QA panels:** After processing, `spectralbridge-qa` generates a multi-panel summary figure per
  flight line, to visually confirm that each step (export, correction, convolution,
  parquet) completed successfully. QA figures are re-generated on every run to reflect
  the current pipeline settings. The spectral panel uses unitless reflectance (0–1) and
  shades VIS/NIR/SWIR regions for readability.

### Parallel Execution

By default the pipeline processes multiple flight lines in sequence. To speed up
workflows, set `max_workers` in `go_forth_and_multiply()` to run several in
parallel. Each worker operates on its own subfolder and logs are prefixed with
the flightline ID:

```
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] 🚀 Processing ...
[NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance] 🚀 Processing ...
```

Feature availability by install type:

| Feature | Base | `[full]` |
|---|---|---|
| Core array ops (NumPy/Scipy) | ✅ | ✅ |
| Raster I/O (Rasterio) | ✅ | ✅ |
| Vector I/O/ops (GeoPandas) | ✅ | ✅ |
| ENVI/HDR (Spectral) | ✅ | ✅ |
| HDF5 (h5py) | ✅ | ✅ |
| Ray engine option | ➖ | ✅ |

Replace `SITE` with a NEON site code and `FLIGHT_LINE` with an actual line identifier.

## Pipeline overview

Cross-Sensor Calibration processes every flight line through a restart-safe
seven-stage flow. Each stage streams a tqdm-style progress bar, logs with a
scoped `[flight_stem]` prefix, and writes artifacts using canonical names from
`get_flight_paths()`:

```mermaid
flowchart LR
    D[Download .h5]
    E[Export ENVI]
    J[Build BRDF+topo JSON]
    C[Correct reflectance]
    R[Resample + Parquet]
    M[Merge parquet tables]
    Q[QA panel]
    D --> E --> J --> C --> R --> M --> Q
```

1. **Download** NEON HDF5 reflectance files (`*_directional_reflectance.h5`)
2. **Export** to ENVI format (`*_envi.img/.hdr`)
3. **Topographic and BRDF Correction** → produces `*_brdfandtopo_corrected_envi.img/.hdr`
4. **Cross-Sensor Convolution** to target sensors (Landsat TM, ETM+, OLI/OLI-2, MicaSense, etc.)
5. **Parquet Export** for all ENVI products. Each rerun validates existing `*.parquet`
   sidecars with `pyarrow.parquet.read_schema`. Valid files are reused; corrupt ones are
   deleted and rebuilt so the stage self-heals bad exports automatically.
6. **Merge Stage (new)** → merges all pixel tables (original, corrected, resampled) into one master parquet per flightline:
   `<prefix>_merged_pixel_extraction.parquet`
   Each row = one pixel, all wavelengths and metadata combined.
7. **QA Panel (restored)** → generates a per-flightline visual summary panel:
   `<prefix>_qa.png`

### Example Usage

```bash
python -m bin.merge_duckdb --data-root /path/to/output --flightline-glob "NEON_*"
```

This produces for each flightline:

```
NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_merged_pixel_extraction.parquet
NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_qa.png
```

## Pipeline Outputs

Each flightline directory will contain:

| Output Type | Example Filename | Description |
|--------------|------------------|--------------|
| Original ENVI | `*_envi.img/.hdr` | Raw NEON export |
| Corrected ENVI | `*_brdfandtopo_corrected_envi.img/.hdr` | BRDF + topography corrected |
| Convolved Products | `*_landsat_tm_envi.img`, etc. | Sensor-matched cubes |
| Parquet Tables | `*_envi.parquet`, `*_landsat_oli_envi.parquet` | Per-product reflectance tables |
| **Merged Master** | `*_merged_pixel_extraction.parquet` | One row per pixel, all wavelengths and metadata combined |
| **QA Panel (PNG)** | `*_qa.png` | Visual summary of all stages |
| QA Metrics (JSON) | `*_qa.json` | Numeric QA measurements |

Helper utilities such as `get_flight_paths(base_folder, flight_stem)` and
`_scoped_log_prefix(prefix)` keep each worker isolated, ensure consistent
filenames, and make the parallel logs readable.

## Running the pipeline

```python
from pathlib import Path
from spectralbridge.pipelines.pipeline import go_forth_and_multiply

go_forth_and_multiply(
    base_folder=Path("output_tester"),
    site_code="NIWO",
    year_month="2023-08",
    product_code="DP1.30006.001",
    flight_lines=[
        "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance",
        "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance",
    ],
    max_workers=4,
)
```

This executes the download → ENVI → BRDF+topo → resample → merge → QA pipeline for every
flight line, streaming progress bars along the way. After the last worker
finishes, the pipeline logs `✅ All requested flightlines processed.`.

### Idempotent / restart-safe

You can safely rerun the same command. The pipeline is stage-aware and
restart-safe:

- If a stage already produced a valid output, that stage logs a `✅ ... (skipping)`
  message and returns immediately.
- If an output is missing or looks corrupted/empty, only that stage is recomputed.
- If you crashed halfway through a long run, you can rerun the same call to resume where
  work is still needed.

A realistic rerun for one flight line now looks like (progress bars omitted for
brevity):

```
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] 🚀 Processing ...
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] 📥 stage_download_h5() found existing .h5 (skipping)
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] 🔎 ENVI export target is ..._envi.img / ..._envi.hdr
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] ✅ ENVI export already complete -> ..._envi.img / ..._envi.hdr (skipping heavy export)
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] ✅ Correction JSON already complete -> ..._brdfandtopo_corrected_envi.json (skipping)
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] ✅ BRDF+topo correction already complete -> ..._brdfandtopo_corrected_envi.img / ..._brdfandtopo_corrected_envi.hdr (skipping)
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] 🎯 Convolving corrected reflectance
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] ✅ Wrote landsat_tm product -> ..._landsat_tm_envi.img / ..._landsat_tm_envi.hdr
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] ✅ Wrote micasense product -> ..._micasense_envi.img / ..._micasense_envi.hdr
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] 📊 Sensor convolution summary | succeeded=['landsat_tm', 'micasense'] skipped=['landsat_etm+', 'landsat_oli', 'landsat_oli2'] failed=[]
[NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance] 🎉 Finished pipeline
```

After all requested flight lines finish, the run concludes with
`✅ All requested flightlines processed.`

### Memory safety

The new pipeline will NOT keep re-loading 20+ GB hyperspectral cubes into memory on every rerun.
The ENVI export step now checks for an existing, valid ENVI pair before doing any heavy work.
If it's already there, it logs "✅ ... skipping heavy export" and moves on.

### Data products

After a successful run you should see, for each flight line:

- `<base_folder>/<flight_stem>.h5` at the workspace root for easy cleanup or
  archival.
- `<base_folder>/<flight_stem>/` containing every derived artifact:
  - `<flight_stem>_envi.img/.hdr/.parquet` (uncorrected ENVI export + summary).
  - `<flight_stem>_brdfandtopo_corrected_envi.img/.hdr/.json/.parquet` (canonical
    corrected cube).
  - `<flight_stem>_<sensor>_envi.img/.hdr/.parquet` for each simulated sensor.
  - `<flight_stem>_merged_pixel_extraction.parquet` (all pixels + wavelengths merged).
  - `<flight_stem>_qa.png` (visual QA panel produced automatically).
  - Support files such as `NIWO_brdf_model.json` generated during correction.

## Troubleshooting Parquet issues

**Symptom**

```text
Parquet magic bytes not found in footer. Either the file is corrupted or this is not a parquet file.
[merge] ⚠️ Skipping invalid parquet <file>: <error>
❌ Issues found: - <file>.parquet → unable to read schema (...)
```

**What it means**

The Parquet sidecar is empty, truncated, or not actually a Parquet file.

**How it recovers**

Re-run the pipeline for that flightline. During Parquet export the pipeline validates
existing files, deletes any that fail `pyarrow.parquet.read_schema`, and rebuilds them
from the ENVI source. `bin/validate_parquets --soft` surfaces the same issues without
aborting the run, and the merge stage skips corrupt sidecars as long as at least one
valid file remains for the prefix.

**Manual recovery (optional)**

```bash
# 1. Optionally, manually remove a known-bad parquet file
rm base_dir/NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance/NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.parquet

# 2. Re-run the pipeline for that flightline; auto-heal will regenerate any missing/invalid parquet
bin/go_forth_and_multiply base_dir/NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance

# 3. Optionally, re-validate in soft mode
bin/validate_parquets --soft base_dir/NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance
```

## Quality Assurance (QA) panels

<!-- TODO: Replace this note with an actual QA panel screenshot when available. -->
<p align="center"><em>QA panel example coming soon.</em></p>

- **Panel A – Raw ENVI RGB:** Confirms the uncorrected export renders with sensible
  color balance and spatial alignment.
- **Panel B – Patch-mean spectrum:** Compares raw vs. BRDF+topo corrected spectra with a
  difference trace to verify the correction stage modified reflectance as expected.
- **Panel C – Corrected NIR preview:** Displays a high-NIR band from the corrected cube to
  quickly spot artifacts such as striping or nodata gaps.
- **Panel D – Sensor thumbnails:** Shows downsampled previews for each convolved sensor
  product to confirm bandstacks were generated.
- **Panel E – Parquet summary:** Lists the Parquet sidecars and file sizes so you can
  confirm tabular exports are present and non-empty.

Generate these summaries at any time with `spectralbridge-qa --base-folder <output_dir>`.

The `_brdfandtopo_corrected_envi` suffix remains the canonical "final"
reflectance for analysis and downstream comparisons; all scientific semantics
are unchanged from previous releases.

### QA dashboard summaries

To review QA performance across many flightlines at once, run:

```bash
spectralbridge-qa-dashboard --base-folder output_fresh
```

This aggregates every `*_qa_metrics.parquet` file, computes per-flightline
statistics, writes `qa_dashboard_summary.parquet`, and renders a companion plot
(`qa_dashboard_summary.png`). Flag rates above 25% are marked with ⚠️ for quick
triage.

### Pipeline stages

Each stage uses `get_flight_paths()` to discover its inputs/outputs and performs
restart-safe validation before doing work. Valid ENVI pairs or JSON artifacts
are reused rather than recomputed, ensuring reruns only fill in missing or
corrupted pieces. `_scoped_log_prefix()` keeps the console readable when several
flightlines run concurrently.

#### Sensor convolution / resampling behavior

- The final stage turns the corrected reflectance cube
  (`*_brdfandtopo_corrected_envi.img/.hdr`) into simulated sensor products
  (e.g. Landsat-style band stacks).
- Each target sensor is attempted independently. Missing/unknown sensor definitions
  are logged with a warning and skipped.
- Each simulated sensor writes an ENVI `.img/.hdr` pair named
  `<flight_stem>_<sensor>_envi.*`. GeoTIFFs are no longer emitted by this stage.
- If a sensor product already exists on disk and validates as an ENVI pair, it is skipped with
  a `✅ ... already complete ... (skipping)` log.
- At the end of the stage, the pipeline logs a summary of which sensors succeeded,
  which were skipped (already done), and which failed.
- The pipeline only raises a runtime error if *all* sensors failed to produce usable
  output for that flight line. Otherwise, partial success is allowed and the
  pipeline continues.

This enforced order prevents earlier bugs where convolution could run on uncorrected data.

### Developer notes

- `process_one_flightline()` is now the canonical per-flightline workflow.
- `go_forth_and_multiply()` orchestrates downloads, per-flightline workers, and
  options like `max_workers`.
- `get_flight_paths()` is the single source of truth for naming and layout of:
  - the `.h5` input,
  - the per-flightline working directory,
  - the uncorrected ENVI export,
  - the correction JSON,
  - the corrected ENVI (`*_brdfandtopo_corrected_envi.*`),
  - the per-sensor convolution outputs and Parquet summaries.

  All pipeline stages call `get_flight_paths()` instead of guessing filenames.
  If file naming changes, update `get_flight_paths()`, not each stage.

- Each stage validates its outputs (non-empty files, parseable JSON, etc.). If outputs are valid,
  that stage logs "✅ ... skipping" and returns immediately.  
  If outputs are missing or corrupted, that stage recomputes them.  
  This is what makes the pipeline resumable after a crash or partial run.

## Install

The recommended setup commands are documented in [Environment setup](#environment-setup).
For a quick pip-based installation use:

```bash
pip install spectralbridge
```

> `spectralbridge[full]` remains available as an alias for teams with
> existing automation but currently resolves to the same dependency set.

## Documentation

Browse the full documentation site at
[earthlab.github.io/SpectralBridge](https://earthlab.github.io/SpectralBridge).
The site is built with MkDocs Material and automatically deployed to GitHub
Pages.

Key entry points:

- [Overview](docs/overview.md)
- [Quickstart](docs/quickstart.md)
- [Stage 01 Raster Processing](docs/stage-01-raster-processing.md)
- [Stage 02 Sorting](docs/stage-02-sorting.md)
- [Stage 03 Pixel Extraction](docs/stage-03-pixel-extraction.md)
- [Stage 04 Spectral Library](docs/stage-04-spectral-library.md)
- [Stage 05 MESMA](docs/stage-05-mesma.md)

## Support Matrix

| Python versions | OS (CI)       | Notes                          |
|-----------------|---------------|--------------------------------|
| 3.10, 3.11      | Linux x86_64  | Tested in GitHub Actions (CI). |
| 3.10, 3.11      | macOS / other | Community supported.           |

## Citation

If you use this software, please cite:

> SpectralBridge (v2.2.0): NEON hyperspectral cross-sensor harmonization pipeline.  
> See `CITATION.cff` in this repository for full author list and metadata.

## License

Distributed under the GPLv3 License. Full citation details are available in
[CITATION.cff](CITATION.cff).

