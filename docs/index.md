# cross-sensor-cal

End-to-end NEON hyperspectral → ENVI export → BRDF+topo correction → cross-sensor convolution → Parquet export → **DuckDB merge (new)** → **QA panel (restored)**.

!!! success "What’s new"
    - Per-flightline master table written as **`<prefix>_merged_pixel_extraction.parquet`**
    - QA panel **`<prefix>_qa.png`** is emitted **after the merge** during full runs
