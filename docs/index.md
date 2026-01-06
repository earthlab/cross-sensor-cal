# Cross-Sensor Calibration

Earth Lab's **cross-sensor-cal** translates reflectance among NEON AOP, uncrewed aerial system (e.g., MicaSense), and Landsat observations to place disparate datasets into a common Landsat-referenced frame. It is designed for notebook-first scientists who need reliable, reproducible harmonization to bridge fine-scale ecological measurements with the long-term continental Landsat record. The workflow is pipeline-based rather than a single correction, and it produces artifacts that can be inspected, reused, and resumed.

## What cross-sensor-cal does
- Normalizes directional reflectance through topographic and BRDF adjustments, then resamples spectra into Landsat and other target bandpasses for direct comparison.
- Bridges across ecological scales by aligning field, UAS, and airborne measurements with the Landsat time series, enabling cross-scale analyses and model transfer.
- Generates harmonized, provenance-rich spectral libraries alongside imagery so downstream modeling can start from analysis-ready data.
- Emits restart-safe outputs (ENVI, Parquet sidecars, merged pixel tables) plus QA panels that document every run.

## Why this approach is trustworthy
- Physics-informed normalization (topographic + BRDF) precedes empirical calibration to keep results grounded in surface reflectance rather than image brightness alone.
- Cross-sensor calibration is referenced to Landsat NBAR, providing an anchored target for spectral translation.
- Every run produces QA PNG/PDF panels and JSON summaries so users can interrogate performance and document decisions.
- The pipeline is deterministic and restartable: intermediate artifacts are written to disk, making runs auditable and recoverable.

## Connection to the scientific literature
This software implements the workflow described in the Remote Sensing of Environment manuscript, *Bridging Scales for Macrosystems Ecology: Harmonizing Western US Plant Functional Types Spectral Data from Drones and NEON Airborne Imagery to Landsat Observations*. It operationalizes that study's approach to aligning UAS and NEON hyperspectral data with Landsat, providing an open, inspectable implementation for reproducible science.

## Who should use this
- Notebook-first scientists analyzing spectral libraries and reflectance products.
- Ecologists and remote sensing researchers needing to connect plot-level or flightline observations to Landsat's long-term record.
- Advanced users and developers integrating the pipeline into larger workflows or extending it to additional sensors.

## Next steps
- [Quickstart](quickstart.md)
- [Jupyter notebook example](usage/notebook-example.md)
- [Tutorials](tutorials/neon-to-envi.md)
- [Reference](reference/configuration.md)
