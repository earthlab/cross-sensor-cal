# Outputs

## ENVI export (`*_envi.img/.hdr`)
- **What**: Source H5 as ENVI for fast banded IO.
- **Produced by**: Stage 2
- **Example**: `NEON_D13_NIWO_..._directional_reflectance_envi.img`
- **Next**: [Stages → 3 Correction](stages.md) • [Schemas](../reference/schemas.md) • [Preview](../usage/parquet_preview.md)

## Corrected ENVI (`*_brdfandtopo_corrected_envi.*`)
- **What**: Topo+BRDF-corrected cube with JSON sidecar.
- **Produced by**: Stage 3
- **Example**: `..._brdfandtopo_corrected_envi.img` + `..._brdfandtopo_corrected_envi.json`
- **Next**: [Convolution](stages.md#4-cross-sensor-convolution) • [Schemas](../reference/schemas.md)

## Convolved ENVI (`*_..._envi_<sensor>.*`)
- **What**: Resampled to target sensor bandpasses.
- **Produced by**: Stage 4
- **Example**: `..._brdfandtopo_corrected_envi_OLI.img`
- **Next**: [Parquet export](stages.md#5-parquet-export) • [Schemas](../reference/schemas.md)

## Parquet (per product)
- **What**: Tidy columns for pixels and bands.
- **Produced by**: Stage 5
- **Next**: [Merge](stages.md#6-duckdb-merge) • [Preview](../usage/parquet_preview.md)

## Merged parquet
- **What**: Analysis-ready combined table.
- **Produced by**: Stage 6
- **Example**: `<prefix>_merged_pixel_extraction.parquet`
- **Next**: [QA panel](stages.md#7-qa-panel)
