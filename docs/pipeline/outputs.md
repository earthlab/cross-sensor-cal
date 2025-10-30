# Outputs

| Output | Pattern | Notes |
|---|---|---|
| ENVI (original) | `*_envi.img/.hdr` | 426 bands (NEON wavelengths) |
| ENVI (corrected) | `*_brdfandtopo_corrected_envi.img/.hdr` | BRDF+topo corrected |
| ENVI (convolved) | `*_landsat_tm_envi.img`, `*_landsat_etm+_envi.img`, `*_landsat_oli[_oli2]_envi.img`, `*_micasense*_envi.img` | Sensor-matched |
| Parquet tables | `*_envi.parquet`, `*_brdfandtopo_corrected_envi.parquet`, `*_landsat_oli_envi.parquet`, ... | Per product |
| **Merged master (new)** | **`<prefix>_merged_pixel_extraction.parquet`** | One row per pixel, all wavelengths + metadata |
| **QA Panel (PNG)** | **`<prefix>_qa.png`** | Visual summary of original/corrected/convolved |

> `<prefix>` resolves to the canonical scene prefix (e.g., `NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance`).
