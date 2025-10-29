# Pipeline reference

Cross-Sensor Calibration orchestrates the same five ordered stages for every
flight line. Each stage consults `get_flight_paths()` to locate its inputs,
per-flightline working directory, and outputs, validates artifacts before
working, and emits emoji-rich logs that make the restart-safe behavior explicit.
All persistent products land inside `<base_folder>/<flight_stem>/` (with the raw
`.h5` kept at the base root) and include ENVI (`.img/.hdr`), JSON metadata, and
Parquet summaries.

## Canonical paths via `get_flight_paths()`

`get_flight_paths(base_folder, flight_stem)` is the authoritative source of
truth for every artifact the pipeline reads or writes. For a flight line with
stem `<flight_stem>` it yields:

- `h5_path` → `<base_folder>/<flight_stem>.h5`
- `work_dir` → `<base_folder>/<flight_stem>/`
- `raw_envi_img` / `raw_envi_hdr` → `<flight_stem>_envi.img` and
  `<flight_stem>_envi.hdr` inside `work_dir`
- `correction_json` → `<flight_stem>_brdfandtopo_corrected_envi.json` in
  `work_dir`
- `corrected_img` / `corrected_hdr` →
  `<flight_stem>_brdfandtopo_corrected_envi.img` and
  `<flight_stem>_brdfandtopo_corrected_envi.hdr`
- `sensor_products` → a dict mapping each sensor (e.g. `landsat_tm`,
  `landsat_etm+`, `landsat_oli`, `landsat_oli2`, `micasense`) to
  `<flight_stem>_<sensor_name>_envi.img` /
  `<flight_stem>_<sensor_name>_envi.hdr`
- `parquet_products` → optional Parquet summaries colocated with their source
  rasters

All stages request their expected paths from this function and refuse to invent
filenames on the fly. If naming ever changes, update `get_flight_paths()` once
rather than editing every stage.

## Stage-by-stage details

Every stage is restart-safe: it skips itself when valid outputs already exist.
Validation requires both sides of an ENVI pair to exist and be non-empty or, for
JSON, the file must parse successfully. When a stage skips, it logs a `✅ ...
(skipping)` message; otherwise it performs work and logs what it produced. Live
tqdm progress bars accompany downloads, ENVI chunk exports, and BRDF+topo
corrections.

### 0. Download NEON HDF5

- **Inputs**
  - NEON site code, product code, and flight stem.
- **Outputs**
  - `<base_folder>/<flight_stem>.h5` (left in the workspace root).
- **Skip criteria**
  - Existing `.h5` file that passes a size/metadata sanity check.
- **Logging**
  - Logs `📥 stage_download_h5()` with a streaming byte counter while downloading.
  - On skip emits `✅ stage_download_h5() found existing .h5 (skipping)`.
- **Failure handling**
  - Raises on HTTP errors or truncated downloads; reruns resume by revalidating the file.

### 1. ENVI export

- **Inputs**
  - NEON directional reflectance cube (`<flight_stem>.h5`).
- **Outputs**
  - `<flight_stem>/<flight_stem>_envi.img`
  - `<flight_stem>/<flight_stem>_envi.hdr`
- **Extras**
  - `<flight_stem>/<flight_stem>_envi.parquet` (summary statistics per tile).
- **Skip criteria**
  - Both ENVI files exist, are non-empty, and pass the internal ENVI validation.
- **Logging**
  - Always logs `🔎 ENVI export target for <flight_stem> is ..._envi.img / ..._envi.hdr` with a chunked progress bar.
  - On skip emits
    `✅ ENVI export already complete for <flight_stem> -> ..._envi.img / ..._envi.hdr (skipping heavy export)`.
  - Otherwise streams progress as tiles are written and logs success.
- **Failure handling**
  - Errors here stop the stage; reruns regenerate if outputs were missing or invalid.

### 2. Build correction JSON

- **Inputs**
  - Uncorrected ENVI pair from stage 1.
- **Output**
  - `<flight_stem>/<flight_stem>_brdfandtopo_corrected_envi.json`
- **Skip criteria**
  - JSON exists and parses.
- **Logging**
  - On skip logs
    `✅ Correction JSON already complete for <flight_stem> -> ..._brdfandtopo_corrected_envi.json (skipping)`.
  - Otherwise logs that it is computing parameters and then writing the JSON.
- **Failure handling**
  - Failures propagate so that reruns recompute the JSON before downstream stages continue.

### 3. BRDF + topographic correction

- **Inputs**
  - Uncorrected ENVI pair.
  - Correction JSON from stage 2.
- **Outputs**
  - `<flight_stem>/<flight_stem>_brdfandtopo_corrected_envi.img`
  - `<flight_stem>/<flight_stem>_brdfandtopo_corrected_envi.hdr`
- **Extras**
  - `<flight_stem>/<flight_stem>_brdfandtopo_corrected_envi.parquet`
- **Skip criteria**
  - Corrected ENVI pair exists, is non-empty, and validates.
- **Logging**
  - On skip logs
    `✅ BRDF+topo correction already complete for <flight_stem> -> ..._brdfandtopo_corrected_envi.img / ..._brdfandtopo_corrected_envi.hdr (skipping)`.
  - When recomputing, streams a chunk progress bar while writing and logs completion.
- **Failure handling**
  - Failures raise immediately because the corrected cube is the canonical science product.
    Reruns recompute just this stage if its outputs were missing or corrupt.

### 4. Sensor convolution / resampling

- **Inputs**
  - Corrected ENVI pair from stage 3. This stage never reads the raw `.h5`.
  - Sensor spectral response library bundled with the project.
- **Outputs**
  - For each known sensor, an ENVI pair following
    `<flight_stem>/<flight_stem>_<sensor_name>_envi.img` /
    `<flight_stem>/<flight_stem>_<sensor_name>_envi.hdr` (e.g.
    `.../NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_tm_envi.img`).
- **Extras**
  - Optional Parquet tables named `<flight_stem>_<sensor_name>_envi.parquet` in
    the same folder.
- **Skip criteria**
  - Individual sensor ENVI pairs that already exist and validate are skipped and reported.
- **Logging**
  - Begins with `🎯 Convolving corrected reflectance for <flight_stem>`.
  - On fresh generation logs
    `✅ Wrote <sensor_name> product for <flight_stem> -> ..._envi.img / ..._envi.hdr`.
  - On skip logs
    `✅ <sensor_name> product already complete for <flight_stem> -> ... (skipping)`.
  - Ends with
    `📊 Sensor convolution summary for <flight_stem> | succeeded=[...] skipped=[...] failed=[...]`
    followed by `🎉 Finished pipeline for <flight_stem>`.
- **Failure handling**
  - Sensors are processed independently. Missing definitions or write failures mark that
    sensor as `failed` but do not abort the stage unless *all* sensors fail and none were
    previously valid. Partial success is acceptable.

## Example run transcript

The restart-safe logs surface the exact work performed. A real rerun for one
flight line now looks like:

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

### Parallel flightline execution

After all downloads succeed, `go_forth_and_multiply()` dispatches each
flightline to a `ThreadPoolExecutor`. The `max_workers` argument controls the
level of concurrency; leave it `None` to process sequentially. `_scoped_log_prefix()`
prepends `[flight_stem]` to every log so parallel runs stay readable, and each
worker writes only within its own `<base>/<flight_stem>/` directory to preserve
idempotence.

When `go_forth_and_multiply(...)` finishes looping over every requested flight
line it logs `✅ All requested flightlines processed.` to confirm site-level
completion.

## Rerun guidance

Call `go_forth_and_multiply(...)` with the same parameters to rerun an entire
site-month. The restart-safe checks ensure that valid artifacts are reused,
missing or invalid stages are recomputed, and partial sensor failures do not
stop progress across the rest of the flight lines. Because each sensor is
accounted for independently, you can inspect the summary lists to see exactly
which products succeeded, which were reused, and which need attention.
