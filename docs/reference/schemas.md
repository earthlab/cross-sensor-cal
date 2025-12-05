# JSON Schemas

The pipeline emits several JSON files containing structured metadata and QA metrics. These files allow downstream tools to audit processing decisions and validate outputs.

This page summarizes the purpose and structure of each schema.

---

## 1. ENVI export metadata

Recorded in:

*_export_metadata.json

Contains:

- wavelength array  
- band names  
- reflectance scaling (NEON conventions)  
- mask types and bit fields  
- CRS and affine transform  

---

## 2. Topographic correction metadata

*_topo_metadata.json

Includes:

- slope and aspect statistics  
- solar geometry used  
- mask adjustments  
- correction parameters  

---

## 3. BRDF metadata

*_brdf_metadata.json

Includes:

- BRDF coefficients per wavelength  
- RMSE of BRDF fits  
- flags for unstable fits  
- view/sun geometry summaries  

---

## 4. Convolution metadata (sensor harmonization)

*_convolution_metadata.json

Contains:

- SRF files used  
- wavelength alignment checks  
- brightness correction coefficients  
- bandpass integration statistics  

---

## 5. QA metrics schema

The QA JSON file contains:

- reflectance summary statistics  
- mask percentages  
- wavelength metadata  
- BRDF/brightness coefficients  
- geometry metadata  

Exact fields are defined in the validation section.

---

## Using schemas in downstream workflows

These JSON files support:

- reproducibility  
- debugging  
- statistical evaluation of harmonization quality  
- provenance tracking for scientific analyses  

---

## Next steps

- [Validation metrics](validation.md)  
- [Pipeline outputs](../pipeline/outputs.md)
