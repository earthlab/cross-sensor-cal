# QA Panels & Metrics

Each processed NEON flight line generates a set of quality assurance (QA) artifacts designed to summarize reflectance distributions, masks, geometry, and harmonization decisions.

This page describes the QA PNG, QA PDF, and QA JSON outputs.

---

## QA PNG

The QA PNG provides a quick visual summary of:

- reflectance distributions for key wavelengths  
- spatial distribution of masks  
- before/after brightness comparison  
- BRDF correction diagnostics  
- wavelength alignment checks  

A typical PNG includes panels for:

1. reflectance histograms  
2. strip-based brightness summaries  
3. cloud/water/snow/invalid pixel frequencies  
4. wavelength metadata  
5. sensor-specific diagnostics  

Use this for fast visual inspection.

---

## QA PDF

The PDF includes:

- all PNG content  
- multi-page extended diagnostics  
- tabulated statistics  
- optional per-band summaries  
- BRDF coefficient tables  
- brightness adjustment coefficients  

This document serves as a flight-line-level audit record.

---

## QA JSON

The JSON file contains machine-readable metrics including:

### Reflectance metrics
- min/max/median for each band  
- proportion of saturated or masked pixels  
- brightness differences across correction stages  

### Mask summary metrics
- percent cloud  
- percent cloud shadow  
- percent snow  
- percent water  
- percent invalid  

### BRDF metrics
- per-band BRDF coefficients  
- reconstruction error statistics  

### Sensor harmonization metrics
- brightness coefficients (per band)  
- spectral alignment checks  
- per-band RMSE for regression-based harmonization (if applicable)  

### Geometry metadata
- solar zenith/azimuth  
- view zenith/azimuth  
- DEM statistics  

---

## Interpreting QA results

Flags to watch for:

- **Reflectance values > 1.5** → possible BRDF or DEM issues  
- **Large brightness shifts** → check convolution stage  
- **High invalid or shadow fraction** → consider masking strategy  
- **Noisy BRDF coefficients** → unstable correction in low-SNR regions  

---

## Next steps

- [Pipeline stages](stages.md)  
- [Outputs & file structure](outputs.md)  
- [Troubleshooting](../troubleshooting.md)
