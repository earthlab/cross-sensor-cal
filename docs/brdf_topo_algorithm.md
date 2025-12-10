# Streamlined BRDF + Topographic corrections

This document summarises the updated streamlined correction path so it aligns with
HyTools/FlexBRDF behaviour.

## Topographic correction (SCS+C)

* For each band a regression `rho = a*cos(i) + b` is fit over valid pixels to
  recover the C-parameter `C=b/a`.
* The surface correction applies `(cos(theta_s)*cos(beta) + C)/(cos(i)+C)` to
  the reflectance in unitless space before converting back to NEON scaling;
  a small denominator guard prevents extreme ratios when `cos(i)+C` is nearly
  zero.
* This SCS+C path is now the default; a cosine-ratio fallback remains available
  via the `use_scs_c` flag.

### Topographic modes and defaults

* `apply_topo_correct` supports two modes:
  * Legacy cosine-ratio (`use_scs_c=False`).
  * SCS+C (`use_scs_c=True`) matching HyTools/FlexBRDF behaviour.
* The current default is `use_scs_c=True`, which is a change from older
  releases that defaulted to cosine-ratio. Callers that require the legacy
  behaviour should explicitly pass `use_scs_c=False` (or the equivalent CLI
  flag) to disable SCS+C.

## NDVI binning

* NDVI is derived from bands nearest 665 nm and 865 nm after converting to
  unitless reflectance.
* Pixels are assigned to configurable bins (defaults ~0.05–1.0 over 25 bins
  with percentile clipping) used for BRDF fitting and application.
* When coefficients are missing or bin counts mismatch, neutral coefficients
  are broadcast across all bins to avoid dropping pixels. Pixels with NDVI
  outside the bin range are remapped into the first bin to preserve coverage
  when no explicit coefficients are available.

## BRDF fitting and application

* Per-band, per-bin regressions solve `rho = f_iso + f_vol*K_vol + f_geo*K_geo`.
* BRDF normalization uses the FlexBRDF ratio `R_ref/R_pix`, evaluating kernels
  at both pixel geometry and a configurable reference geometry.

## Scaling and thresholds

* Modeling occurs in unitless reflectance; scale factors are applied only at the
  edges. Optional clamps guard obviously invalid reflectance values.
* All masks propagate cube no-data and NDVI bin assignments; non-finite outputs
  resolve to the cube `no_data` value.
