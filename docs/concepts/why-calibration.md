# Why cross-sensor calibration?

Understanding vegetation, disturbance, and ecosystem structure often requires integrating information from **multiple sensors** operating at different spatial, temporal, and spectral scales. These include:

- NEON airborne imaging spectroscopy  
- drone multispectral systems (e.g., MicaSense)  
- moderate-resolution satellites like Landsat and Sentinel  

Each sensor “sees” the landscape differently, and these differences can obscure the ecological signals we care about unless we account for them.

---

## The problem: apples and oranges reflectance

Sensors differ in:

- **spectral response** (band centers, widths, shapes)  
- **illumination geometry** (solar zenith, azimuth, atmospheric path)  
- **viewing geometry** (sensor zenith, azimuth)  
- **radiometric scaling and masking conventions**  
- **ground sampling distance and spatial aggregation**  

Even when they image the same location on the same day, their raw reflectance values are *not* directly comparable.

This creates challenges when trying to:

- validate satellite products using NEON  
- relate drone measurements to NEON or Landsat  
- build cross-scale ecological models  
- interpret changes in reflectance through time or across terrain  

---

## Correcting vs. harmonizing

cross-sensor-cal performs two distinct operations:

### 1. Physical corrections  
These aim to reduce variation caused by **illumination and terrain**:

- topographic correction (slope and aspect effects)  
- BRDF correction (view/sun geometry effects)  

The result is a reflectance product that is more comparable across acquisition conditions.

### 2. Sensor harmonization  
This converts corrected hyperspectral data into another sensor’s bandspace by integrating spectra against **published spectral response functions** (e.g., Landsat OLI, MicaSense RedEdge).

Optional brightness adjustments are documented in the QA outputs.

---

## Why NEON as the foundation?

NEON AOP data provide:

- high spectral resolution  
- per-pixel geometry information  
- consistent radiometric processing  
- spatial coverage aligned with ecological research sites  

These properties make NEON a powerful intermediary between plot-scale measurements and satellite observations.

cross-sensor-cal implements a *reproducible stepwise process* to:

1. extract NEON reflectance into ENVI  
2. correct it physically  
3. harmonize it to satellite/drone sensors  
4. output analysis-ready tables and QA documentation  

---

## What still requires care

Even after calibration and harmonization:

- residual BRDF effects can remain  
- atmospheric differences between sensors matter  
- snow, smoke, water, and shadows require attention  
- scale mismatch affects interpretation  
- masks and quality flags differ across platforms  

The pipeline aims to make all assumptions explicit—every major processing step writes a JSON sidecar describing inputs, parameters, and results.

---

## Where to go next

- Learn how each pipeline stage works:  
  [Pipeline overview & stages](../pipeline/stages.md)

- Follow a practical workflow:  
  [NEON → corrected ENVI](../tutorials/neon-to-envi.md)

- Explore validation and metrics:  
  [QA panels & metrics](../pipeline/qa.md)
