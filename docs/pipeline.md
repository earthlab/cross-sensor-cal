# Pipeline reference

Cross-Sensor Calibration orchestrates the same four ordered stages for every
flight line. Each stage consults `get_flightline_products()` to locate its
inputs/outputs, validates artifacts before working, and emits emoji-rich logs
that make the restart-safe behavior explicit. All persistent products are ENVI
(`.img/.hdr`) or JSON metadata.

## Canonical filenames via `get_flightline_products()`

`get_flightline_products()` is the authoritative source of truth for every
artifact the pipeline reads or writes. For a flight line with stem
`<flight_stem>` it yields:

- `raw_envi_img` / `raw_envi_hdr` → `<flight_stem>_envi.img` and
  `<flight_stem>_envi.hdr`
- `correction_json` → `<flight_stem>_brdfandtopo_corrected_envi.json`
- `corrected_img` / `corrected_hdr` →
  `<flight_stem>_brdfandtopo_corrected_envi.img` and
  `<flight_stem>_brdfandtopo_corrected_envi.hdr`
- `sensor_products` → a dict mapping each sensor (e.g. `landsat_tm`,
  `landsat_etm+`, `landsat_oli`, `landsat_oli2`, `micasense`) to
  `<flight_stem>_<sensor_name>_envi.img` /
  `<flight_stem>_<sensor_name>_envi.hdr`

All stages request their expected paths from this function and refuse to invent
filenames on the fly. If naming ever changes, update
`get_flightline_products()` once rather than editing every stage.

## Stage-by-stage details

Every stage is restart-safe: it skips itself when valid outputs already exist.
Validation requires both sides of an ENVI pair to exist and be non-empty or, for
JSON, the file must parse successfully. When a stage skips, it logs a `✅ ...
(skipping)` message; otherwise it performs work and logs what it produced.

### 1. ENVI export

- **Inputs**
  - NEON directional reflectance cube (`<flight_stem>.h5`).
- **Outputs**
  - `<flight_stem>_envi.img`
  - `<flight_stem>_envi.hdr`
- **Skip criteria**
  - Both ENVI files exist, are non-empty, and pass the internal ENVI validation.
- **Logging**
  - Always logs `🔎 ENVI export target for <flight_stem> is ..._envi.img / ..._envi.hdr`.
  - On skip emits
    `✅ ENVI export already complete for <flight_stem> -> ..._envi.img / ..._envi.hdr (skipping heavy export)`.
  - Otherwise loads the ~20+ GB cube, exports it, and logs success.
- **Failure handling**
  - Errors here stop the stage; reruns regenerate if outputs were missing or invalid.

### 2. Build correction JSON

- **Inputs**
  - Uncorrected ENVI pair from stage 1.
- **Output**
  - `<flight_stem>_brdfandtopo_corrected_envi.json`
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
  - `<flight_stem>_brdfandtopo_corrected_envi.img`
  - `<flight_stem>_brdfandtopo_corrected_envi.hdr`
- **Skip criteria**
  - Corrected ENVI pair exists, is non-empty, and validates.
- **Logging**
  - On skip logs
    `✅ BRDF+topo correction already complete for <flight_stem> -> ..._brdfandtopo_corrected_envi.img / ..._brdfandtopo_corrected_envi.hdr (skipping)`.
  - When recomputing, logs the correction progress and completion.
- **Failure handling**
  - Failures raise immediately because the corrected cube is the canonical science product.
    Reruns recompute just this stage if its outputs were missing or corrupt.

### 4. Sensor convolution / resampling

- **Inputs**
  - Corrected ENVI pair from stage 3. This stage never reads the raw `.h5`.
  - Sensor spectral response library bundled with the project.
- **Outputs**
  - For each known sensor, an ENVI pair following
    `<flight_stem>_<sensor_name>_envi.img` /
    `<flight_stem>_<sensor_name>_envi.hdr` (e.g.
    `NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_tm_envi.img`).
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
🚀 Processing NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance ...
🔎 ENVI export target for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance is NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.hdr
✅ ENVI export already complete for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.hdr (skipping heavy export)
✅ Correction JSON already complete for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_brdfandtopo_corrected_envi.json (skipping)
✅ BRDF+topo correction already complete for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_brdfandtopo_corrected_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_brdfandtopo_corrected_envi.hdr (skipping)
🎯 Convolving corrected reflectance for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance
✅ Wrote landsat_tm product for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_tm_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_tm_envi.hdr
✅ Wrote landsat_etm+ product for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_etm+_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_etm+_envi.hdr
✅ Wrote landsat_oli product for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_oli_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_oli_envi.hdr
✅ Wrote landsat_oli2 product for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_oli2_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_landsat_oli2_envi.hdr
✅ Wrote micasense product for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance -> NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_micasense_envi.img / NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_micasense_envi.hdr
📊 Sensor convolution summary for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance | succeeded=['landsat_tm', 'landsat_etm+', 'landsat_oli', 'landsat_oli2', 'micasense'] skipped=[] failed=[]
🎉 Finished pipeline for NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance
```

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
