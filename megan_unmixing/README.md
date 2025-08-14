# Megan Unmixing

## Overview
This folder contains an R-based exploration of spectral unmixing methods led by
Megan. It demonstrates how endmember selection affects root mean square error
(RMSE) in mixture models.

## Prerequisites
- R 4.0+
- Packages: `raster`, `rgdal`, and other dependencies referenced in
  `unmixing.Rmd`

## Step-by-step tutorial
1. Open `unmixing.Rmd` in RStudio or another R Markdown editor.
2. Knit the document to produce `unmixing.html` and review the resulting plots
   (`max_rmse_percent_drop.png`, `mean_rmse_percent_drop.png`).

## Reference
- `data/` – sample datasets used in the analysis
- `unmixing.Rmd` – main R Markdown file
- `unmixing.html` – rendered output

## Next steps
Translate promising approaches from this analysis into Python modules within the
`unmixing/` directory.

Last updated: 2025-08-14
