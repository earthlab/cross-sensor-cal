# QA Panel

**What it shows**: pre/post distributions, thumbnails, quick smells (banding, skew, masking).

**Generate during pipeline**: produced automatically by Stage 7.

**Regenerate directly**:
```bash
cross-sensor-cal qa-panel \
  --merged merged/<prefix>_merged_pixel_extraction.parquet \
  --out qa/
```

**Interpretation hints**
- Strong banding → inspect ENVI header wavelengths.
- Skewed histograms → check masks and BRDF parameters.
- Missing panel → ensure merged parquet path is correct.

**Tip**: Click images to enlarge (lightbox enabled).
