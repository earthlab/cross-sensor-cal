## [Unreleased] – Pipeline refactor for idempotent, ordered execution (October 2025)

- Pipeline is now restart-safe / idempotent: each major stage checks for valid existing outputs
  and skips heavy recompute, logging `✅ ... (skipping)`.
- Introduced canonical output naming via `get_flightline_products()`.
- All per-sensor convolution products are now written as ENVI `.img/.hdr` pairs using the
  pattern `<flight_stem>_<sensor_name>_envi.img/.hdr`.
- Removed `.tif` GeoTIFF outputs from the advertised workflow.
- Convolution now ALWAYS reads the BRDF+topo corrected ENVI cube, never the raw NEON `.h5` directly.
- Added per-sensor success/skip/fail accounting and a final summary line:
  `📊 Sensor convolution summary ... | succeeded=[...] skipped=[...] failed=[...]`
- Pipeline no longer hard-stops if one sensor fails; it finishes the flight line as long as at
  least one sensor succeeded (or had a valid preexisting output).
- Added final site-level completion log: `✅ All requested flightlines processed.`
