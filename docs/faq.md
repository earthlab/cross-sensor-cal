# Frequently Asked Questions (FAQ)

---

## What is the purpose of cross-sensor calibration?

To make reflectance from NEON, drones, and satellites comparable by removing geometric artifacts and aligning spectral responses.

---

## Why does NEON use scaled reflectance?

NEON stores reflectance as scaled integers for performance. The export stage automatically converts these into floating-point reflectance values using NEON’s scale factors.

---

## Do I need both topographic and BRDF correction?

Yes, for most ecological analyses.  
Topographic correction removes slope/aspect artifacts; BRDF correction normalizes view/sun geometry.

---

## Why do some bands look noisy?

Low-SNR wavelength regions (especially in the SWIR) are more sensitive to BRDF fitting errors.

---

## What sensors can I harmonize to?

Currently:

- Landsat OLI / OLI-2  
- MicaSense RedEdge (via regression)  

More sensors can be added as SRF tables become available.

---

## Why does the pipeline take so long?

NEON tiles are very large (tens of GB).  
BRDF and ENVI export steps are computationally expensive.

Parallelization helps, but memory constraints matter.

---

## Why do some pixels have reflectance > 1?

BRDF or DEM correction may amplify noise at certain angles or terrain slopes.  
Such pixels are reported in QA metrics and usually masked.

---

## Can I run analyses without touching ENVI files?

Yes—use the Parquet outputs (`*_merged_pixel_extraction.parquet`) for large-scale analysis.

---

## Next steps

- [Troubleshooting](troubleshooting.md)  
- [Pipeline outputs](pipeline/outputs.md)
