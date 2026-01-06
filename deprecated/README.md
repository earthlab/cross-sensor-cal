# Deprecated resources

This folder quarantines legacy code, scripts, notebooks, documentation, and tests
that are no longer part of the supported SpectralBridge pipeline. The files are
kept temporarily for reference while teams finish migrating to the ENVI-only,
idempotent workflow implemented in `src/spectralbridge/pipelines/pipeline.py`.
They are **not** imported, executed, or covered by CI.

## Layout
- `code/` – HyTools-era exporters, GeoTIFF writers, superseded resampling logic,
  and historical utilities that have been replaced by the modern pipeline.
- `bin/` – Command line entry points that targeted the pre-package repository
  layout. Use the `cscal-*` console scripts instead.
- `docs/` – Documentation describing workflows that produced GeoTIFF deliverables
  or relied on the old module structure. Current docs live under `docs/` at the
  repository root.
- `notebooks/` – Exploratory notebooks and scratch pads that are not part of the
  reproducible processing path.
- `tests_legacy/` – Tests for the retired workflows. Active tests reside in
  `tests/` and validate the ENVI-only pipeline.

The supported deliverables are ENVI `.img/.hdr` pairs produced by the BRDF+topo
corrected pipeline, plus optional `.parquet` sidecars generated in stage 6. GeoTIFF
exports and HyTools-based conversion stages are no longer maintained.
