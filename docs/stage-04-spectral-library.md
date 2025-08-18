# Stage 04 Spectral Library

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
The spectral library stores every spectrum with its full provenance. Each entry includes:

- `site`: location identifier where you collected the field sample.
- `sensor`: instrument or platform that measured the spectrum.
- `wavelengths`: array of nanometer values shared across spectra.

These fields live alongside the reflectance values in a row-oriented table or NetCDF group. Use them to
filter spectra by site, compare sensors, or align data to the wavelength grid.

### Quality controls

You can clean spectra before analysis:

1. **Outlier filtering** – drop samples that exceed three standard deviations from the mean reflectance
   at any wavelength.
2. **Smoothing** – apply a Savitzky–Golay or moving-average filter to reduce instrument noise while
   preserving absorption features.
3. **Signal-to-noise ratio (SNR)** – flag spectra with low SNR and exclude them from downstream
   modeling.

### Versioning and provenance

Each library release increments a semantic version. A `manifest.json` file lists source datasets,
processing code commits, and software versions so you can reproduce the library or audit its origin.

Last updated: 2025-08-18
<!-- FILLME:END -->
