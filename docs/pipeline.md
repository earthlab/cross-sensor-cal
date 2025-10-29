# Pipeline reference

Cross-Sensor Calibration orchestrates the same four ordered stages for every
flight line. Each stage consults `get_flightline_products()` to locate its
inputs/outputs and validates artifacts before deciding whether to work or skip.
All persistent products are ENVI cubes (`.img/.hdr`) or JSON metadata.

## Canonical artifacts

For a flight line with stem `<flight_stem>` the pipeline expects the following
artifacts in the base folder:

- `<flight_stem>.h5` – original NEON hyperspectral input.
- `<flight_stem>_envi.img` / `<flight_stem>_envi.hdr` – uncorrected ENVI export.
- `<flight_stem>_brdfandtopo_corrected_envi.json` – parameters for BRDF +
  topographic correction.
- `<flight_stem>_brdfandtopo_corrected_envi.img` /
  `<flight_stem>_brdfandtopo_corrected_envi.hdr` – corrected reflectance cube
  used by all downstream steps.
- `<flight_stem>_<sensor_name>_envi.img` /
  `<flight_stem>_<sensor_name>_envi.hdr` – per-sensor simulated bandstacks.

The helper `get_flightline_products()` is the single source of truth for these
paths. Any change to naming should occur there so all stages stay consistent.

## Stage-by-stage details

Each stage is restart-safe: it skips itself when valid outputs already exist.
Validation requires both sides of an ENVI pair to exist and be non-empty or, for
JSON, the file must parse successfully.

### 1. ENVI export

- **Input**: `<flight_stem>.h5` directional reflectance cube from NEON.
- **Outputs**: `<flight_stem>_envi.img` and `<flight_stem>_envi.hdr`.
- **Skip conditions**: both ENVI files already exist, have size > 0, and pass the
  internal consistency check.
- **Failure/continue behavior**: errors stop the stage because downstream steps
  require the ENVI export. On rerun the stage attempts regeneration if the
  outputs were missing or invalid.
- **Notes**: this is the only time the pipeline reads the large `.h5`; skipping
  here saves significant I/O when rerunning the site-month.

### 2. Correction JSON build

- **Inputs**: uncorrected ENVI export, metadata from the `.h5` file.
- **Output**: `<flight_stem>_brdfandtopo_corrected_envi.json` containing BRDF and
  topographic correction parameters.
- **Skip conditions**: JSON exists and parses successfully.
- **Failure/continue behavior**: failure regenerates on rerun; downstream stages
  do not proceed without this JSON.

### 3. BRDF + topographic correction

- **Inputs**: uncorrected ENVI export and the correction JSON.
- **Outputs**: `<flight_stem>_brdfandtopo_corrected_envi.img` and
  `<flight_stem>_brdfandtopo_corrected_envi.hdr`.
- **Skip conditions**: corrected ENVI pair already exists, is non-empty, and
  validates.
- **Failure/continue behavior**: failure raises immediately because the
  corrected cube is the canonical reflectance product. Reruns regenerate only
  this stage if the corrected pair is missing or invalid.
- **Notes**: the corrected ENVI cube is now the single source of truth for
  downstream analysis and resampling.

### 4. Sensor convolution / resampling

- **Input**: the corrected ENVI pair from stage 3. The stage never reads the raw
  `.h5` or uncorrected ENVI export.
- **Outputs**: for each known sensor definition, an ENVI pair named
  `<flight_stem>_<sensor_name>_envi.img/.hdr`.
- **Skip conditions**: if an individual sensor's ENVI pair exists and validates,
  that sensor is skipped and reported as such.
- **Failure/continue behavior**: sensors are processed independently. Unknown
  sensors log a warning and are skipped. If a sensor fails to generate a valid
  ENVI pair, it is recorded as a failure but does not halt the stage unless all
  sensors fail. The pipeline continues to the next flight line as long as at
  least one sensor succeeded or was previously skipped with valid outputs.
- **Notes**: the stage emits a summary log listing succeeded, skipped, and
  failed sensors.

## Example logs

A typical rerun produces logs similar to the following:

```
12:01:03 | INFO | 🚀 Processing NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance ...
12:01:03 | INFO | 🔎 ENVI export target is ..._envi.img / ..._envi.hdr
12:01:03 | INFO | ✅ ENVI export already complete for ..._envi.img / ..._envi.hdr (skipping heavy export)
12:01:04 | INFO | ✅ Correction JSON already complete for ..._brdfandtopo_corrected_envi.json (skipping)
12:01:05 | INFO | ✅ BRDF+topo correction already complete for ..._brdfandtopo_corrected_envi.img/.hdr (skipping)
12:01:06 | INFO | 🎯 Convolving corrected reflectance for ...
12:01:06 | INFO | ✅ landsat_oli product already complete ... (skipping)
12:01:06 | INFO | ⚠️  Sensor micasense_altum is not defined in the library ... (skipping)
12:01:06 | INFO | 📊 Sensor convolution summary for ... | succeeded=['landsat_oli'] skipped=['landsat_tm'] failed=['sentinel2a']
12:01:06 | INFO | 🎉 Finished pipeline for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance
```

The emoji-driven logs highlight when work is skipped (`✅`), when the pipeline is
convolving sensors (`🎯`), and summarize per-sensor outcomes (`📊`).

## Rerun guidance

Call `go_forth_and_multiply(...)` with the same parameters to rerun an entire
site-month. The restart-safe checks ensure that valid artifacts are reused, only
missing or invalid stages are recomputed, and partial sensor failures do not stop
progress across the rest of the flight lines.
