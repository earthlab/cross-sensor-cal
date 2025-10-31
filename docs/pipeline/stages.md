# Pipeline Stages

1. **Download** `*_directional_reflectance.h5`
2. **Export to ENVI** → `*_envi.img/.hdr`
3. **Topographic + BRDF correction** → `*_brdfandtopo_corrected_envi.img/.hdr` (+ JSON)
   **Optional: Brightness correction**  
   Normalize reflectance values across varying illumination using  
   `apply_brightness_correction()` before BRDF correction to stabilize tile mosaics.
4. **Cross-sensor convolution** (TM, ETM+, OLI/OLI-2, MicaSense variants)
5. **Parquet export** for all ENVI products
6. **DuckDB Merge (new)** → **`<prefix>_merged_pixel_extraction.parquet`**
7. **QA Panel (restored)** → **`<prefix>_qa.png`** (triggered post-merge)
