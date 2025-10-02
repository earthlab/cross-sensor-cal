# Datasets

## Overview
This folder stores static geographic datasets used as inputs or examples in
the calibration workflow. The files include vector polygons and spectral
lookup tables that help you test the pipeline without downloading large
archives.

## Prerequisites
- A GIS application or library capable of reading `.gpkg` and `.xlsx` files
- Optional: NEON flightline identifiers for context

## Step-by-step tutorial
1. Inspect a polygon dataset:

```bash
ogrinfo Datasets/niwot_aop_macrosystems_data_2023_12_8_23.gpkg -summary
```

2. Use the polygons when extracting spectra by passing the file path to
   `process_base_folder`:

```python
from cross_sensor_cal.polygon_extraction import control_function_for_extraction

control_function_for_extraction(
    directory="data/NIWO_2023_08",
    polygon_path="Datasets/niwot_aop_macrosystems_data_2023_12_8_23.gpkg",
)
```

## Reference
- `Micasense band centers.xlsx` – band center definitions for MicaSense sensors
- `niwot_aop_macrosystems_data_2023_12_8_23.gpkg` – example AOI polygons
- `niwot_aop_polygons_2023_12_8_23_analysis_ready_half_diam.gpkg` – analysis
  subset of the above polygons

## Next steps
You can add your own datasets to this directory and reference them in the
pipeline configurations.

Last updated: 2025-08-14
