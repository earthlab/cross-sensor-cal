# Schemas

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
## Raster band metadata schema

Describes the properties of each raster band after resampling or
convolution.

| Field           | Type    | Units | Description                      |
|-----------------|---------|-------|----------------------------------|
| `band`          | int     | –     | Sequential band number           |
| `wavelength_nm` | float   | nm    | Center wavelength                |
| `fwhm_nm`       | float   | nm    | Full width at half maximum       |
| `unit`          | string  | –     | Measurement units for reflectance|

Example JSON:

```json
[
  {"band": 1, "wavelength_nm": 450.0, "fwhm_nm": 20.0, "unit": "nm"},
  {"band": 2, "wavelength_nm": 550.0, "fwhm_nm": 25.0, "unit": "nm"}
]
```

## Pixel table schema

Each row represents one pixel extracted from a raster scene.

| Column      | Type  | Units | Description                    |
|-------------|-------|-------|--------------------------------|
| `Pixel_ID`  | int   | –     | Unique pixel identifier        |
| `Pixel_Row` | int   | row   | Raster row index (0‑based)     |
| `Pixel_Col` | int   | col   | Raster column index (0‑based)  |
| `B1..Bn`    | float | reflectance | Band reflectance values |

Example table:

| Pixel_ID | Pixel_Row | Pixel_Col | B1   | B2   | B3   |
|---------:|----------:|----------:|-----:|-----:|-----:|
| 1        | 10        | 15        | 0.12 | 0.09 | 0.03 |
| 2        | 11        | 16        | 0.10 | 0.08 | 0.02 |

## Spectral library schema

Spectral libraries store reference spectra for endmembers used during
unmixing. When saved as JSON or Parquet, each record contains:

| Field            | Type          | Description                                       |
|------------------|---------------|---------------------------------------------------|
| `spectrum_id`    | string        | Unique identifier                                 |
| `class_label`    | string        | Endmember class (e.g., soil, vegetation)          |
| `wavelength_nm`  | array<float>  | Wavelength centers                                |
| `reflectance`    | array<float>  | Corresponding reflectance values                  |
| `metadata`       | object        | Optional information (sensor, date, notes, etc.)  |

Example JSON entry:

```json
{
  "spectrum_id": "veg01",
  "class_label": "vegetation",
  "wavelength_nm": [450, 550, 650],
  "reflectance": [0.12, 0.32, 0.45],
  "metadata": {"sensor": "NEON", "acquired": "2020-08-01"}
}
```

## MESMA outputs schema

Results from Multiple Endmember Spectral Mixture Analysis for each
pixel.

| Column          | Type  | Units       | Description                               |
|-----------------|-------|-------------|-------------------------------------------|
| `Pixel_ID`      | int   | –           | Input pixel identifier                     |
| `fraction_*`    | float | fraction    | Fractional abundance per endmember band    |
| `RMSE`          | float | reflectance | Root mean square error of the model fit    |
| `QA`            | int   | –           | Quality flag (0=good, higher=worse)        |

Example table:

| Pixel_ID | fraction_soil | fraction_veg | RMSE | QA |
|---------:|--------------:|-------------:|-----:|---:|
| 1        | 0.40          | 0.60         | 0.01 | 0 |
| 2        | 0.55          | 0.45         | 0.02 | 1 |

Example JSON line for one pixel:

```json
{
  "Pixel_ID": 1,
  "fraction_soil": 0.40,
  "fraction_veg": 0.60,
  "RMSE": 0.01,
  "QA": 0
}
```

Last updated: 2025-08-18
<!-- FILLME:END -->
