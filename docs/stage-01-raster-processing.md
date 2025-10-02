# Stage 01 Raster Processing

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
## Purpose
- Convert NEON HDF5 scenes to ENVI rasters.
- Export ancillary layers for geometry and topography.
- Apply polygon and QA masks.
- Run topographic correction followed by BRDF adjustment.

## Inputs and assumptions
- HDF5 reflectance products with NEON-style filenames.
- CRS defined in metadata (`map info` or `coordinate system string`); override if missing.
- Metadata keys used: `Reflectance/Reflectance_Data`, `Ancillary_Imagery/*`, `Logs/Solar_*`,
  `to-sensor_*`.

## Outputs and guarantees
- Reflectance ENVI: `NEON_<domain>_<site>_<product>_<tile>_<date>_<time>_reflectance_envi.img`
  plus `.hdr`.
- Ancillary stack: `*_ancillary_envi.img` with bands Path Length, Sensor Azimuth, Sensor
  Zenith, Solar Azimuth, Solar Zenith, Slope, Aspect.
- Masked rasters append `_masked`.
- Topo+BRDF corrected raster: `*_reflectance_brdfandtopo_corrected_<suffix>.img`.
- Band names preserved from source metadata.

## Configuration keys
- `input_files`: list of ENVI reflectance images.
- `anc_files`: mapping of ancillary band names to `[path, index]`.
- `corrections`: ordered list, e.g., `["topo", "brdf"]`.
- `export.output_dir` and `export.suffix`: where corrected files land.
- `num_cpus`: Ray worker count.
- `bad_bands`, `topo`, `brdf`: sections controlling masking and coefficient options.

## How to run
### CLI
```bash
python src/neon_to_envi.py --images scene.h5 --output_dir out -anc
python src/topo_and_brdf_correction.py config.json
```

### Python
```python
from src.neon_to_envi import neon_to_envi
from src.mask_raster import mask_raster_with_polygons
from src.topo_and_brdf_correction import topo_and_brdf_correction

neon_to_envi(["scene.h5"], "out", anc=True)
mask_raster_with_polygons(envi_file, "polygons.geojson")
topo_and_brdf_correction("config.json")
```

## Performance knobs
- `num_cpus` controls Ray parallelism.
- HyTools `iterate(by="chunk")` respects `chunk_size` to limit memory.

## QA checks
- Confirm reflectance ranges (0–1); ancillary angles 0–180°.
- Plot per-band histograms to catch saturation or negative values.
- Verify masked pixels equal nodata values.

## Troubleshooting
- Filenames must match NEON regex; use `metadata_override` when they do not.
- Inject missing metadata via `metadata_override` or `raster_crs_override`.
- ENVI header shape mismatches (`samples`, `lines`, `bands`) cause HyTools load errors—fix
  headers or regenerate files.

Last updated: 2025-08-18
<!-- FILLME:END -->
