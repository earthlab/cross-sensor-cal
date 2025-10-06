# Stage 03 Pixel Extraction

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
## Overview
At this stage you sample pixels from the sorted scenes and write the values to a tabular file.
The table becomes the input to spectral unmixing and other downstream analyses.

## Sampling rules
1. Define a consistent random seed so repeated runs draw the same pixels.
2. Sample within each land‐cover class or tile to avoid geographic bias.
3. Drop any pixel flagged by a quality mask or falling outside the region of interest.

## Handling nodata and masks
- Treat nodata values (`-9999` by default) as missing and skip those records.
- Apply cloud, shadow, and water masks before sampling so invalid pixels never reach the table.
- Keep a boolean `is_masked` column to track which values were rejected.

## Tile vs full scene
- **Tiles** scale better for large mosaics and let you parallelize extraction.
- **Full scenes** are faster when memory allows and ensure contiguous coverage.
Choose the approach that matches your hardware and scene size; the output format is identical.

## Output tables
Each row represents one pixel.
Columns typically include `scene_id`, `tile_id`, `x`, `y`, band values, and `is_masked`.
Write tables as CSV for quick inspection or Parquet for efficient storage.
Keep Parquet outputs in a `full_extracted_pixels` folder that lives alongside the tile folder so the extracted tables sit next to, not inside, the source data.
Partition by scene and tile so you can read subsets without loading the whole dataset.

## Memory tips
- Process one tile at a time and release arrays with `del` to free RAM.
- When writing CSV, stream rows with a generator instead of building a huge DataFrame.
- Prefer Parquet with compression to reduce disk use and load times.

## Quick integrity checks
- Confirm row counts match the number of valid pixels expected per tile.
- Scan for remaining nodata values: `rg -n "-9999" sample.csv`.
- Plot a histogram of one band to detect obvious outliers before moving on.

## Next steps
Continue to [Stage 04](stage-04-spectral-library.md) to build the spectral library from the extracted pixels.

Last updated: 2025-08-18
<!-- FILLME:END -->
