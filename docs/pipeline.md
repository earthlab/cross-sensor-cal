# Pipeline overview

Cross-Sensor Calibration processes each flightline through staged outputs to
ensure idempotent reruns and consistent naming. After the BRDF + topographic
correction stage writes `<flight_stem>_brdfandtopo_corrected_envi.img/.hdr`, the
sensor convolution stage derives per-sensor bandstacks exclusively from that
corrected cube.

## Sensor convolution outputs

- Every simulated sensor produces an ENVI cube pair named
  `<flight_stem>_<sensor>_envi.img` and `<flight_stem>_<sensor>_envi.hdr`.
- The pipeline no longer emits GeoTIFF sensor products by default; the ENVI
  cubes are the canonical outputs for downstream workflows.
- Convolution runs only against the corrected ENVI pair
  (`<flight_stem>_brdfandtopo_corrected_envi.img/.hdr`).
- Each sensor is attempted independently so partial success is allowed.
- Prior to any heavy work the pipeline validates whether a sensor's ENVI pair
  already exists and is non-empty; valid pairs are skipped with a log message.
- After processing the stage reports which sensors succeeded, which were
  skipped, and which failed. A runtime error is raised only if **all** sensors
  failed to produce usable ENVI cubes.

These rules align sensor naming with the rest of the pipeline and guarantee that
reruns can detect previously generated products without relying on GeoTIFF
artifacts.
