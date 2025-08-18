# Stage 05 MESMA

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
## Using the spectral library
The spectral library from [Stage 04](stage-04-spectral-library.md) provides the
candidate endmember spectra. You load it as a NumPy array and pass it to the
MESMA routine alongside the target image. MESMA iterates through library
combinations to find the model with the lowest root mean square error (RMSE).

```python
from unmixing.el_mesma import MesmaCore

mesma = MesmaCore()
fractions, residuals = mesma._mesma(image, library)
```

## Endmember selection strategies
- **Exhaustive search** – evaluate all combinations up to a fixed complexity.
- **Class-based** – restrict models to endmembers drawn from predefined
  classes such as vegetation or soil.
- **Random sampling** – sample combinations to reduce runtime for large
  libraries.

## Outputs
- Per-endmember fraction maps showing the proportional contribution of each
  material and a shade fraction.
- A residual raster capturing the difference between observed and reconstructed
  spectra.

## Validation
- Verify that the fractions for each pixel sum to approximately `1.0`.
- Discard models with RMSE above a user-defined threshold to ensure a reliable
  fit.

Last updated: 2025-08-18
<!-- FILLME:END -->
