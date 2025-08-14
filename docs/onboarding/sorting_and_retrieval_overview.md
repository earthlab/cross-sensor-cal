# 🔄 Data Pipeline Overview & Sorting Responsibilities

## 🌐 Context

This project processes NEON hyperspectral HDF5 (`.h5`) files through a series of transformations to produce corrected and convolved raster data products. The pipeline runs in a cloud instance without persistent local storage. To preserve data between runs, we interface with CyVerse’s persistent storage system using `gocmd` — a modified command-line tool developed by ESIIL.

## 📊 Data Flow Overview

Data flows spatially by **site** and **flight line**, with outputs organized by these spatial-temporal identifiers:

1. **Download** raw `.h5` files for specific NEON flight lines
2. **Convert** each `.h5` to ENVI format (`_radiance.img`, `_ancillary.img`)
3. **Correct** the ENVI reflectance imagery using TOPO and BRDF models
4. **Convolve** corrected files to simulate different satellite sensors (Landsat, MicaSense, etc.)
5. **Sort** the resulting raster outputs into consistent folders and filenames by spatial location and sensor type
6. **Run** additional analysis scripts that:
   - Pull sorted files *back* from persistent storage
   - Process them in parallel
   - Save results back to persistent storage

## 🧩 Your Task: Sorting & Retrieval

You are responsible for:
- Organizing chaotic output raster files after the convolution/resampling step
- Designing a sorting scheme that groups files by:
  - Site
  - Sensor type
  - Correction state
- Using `gocmd` to:
  - Push sorted files to persistent storage in `/iplant/home/shared/...`
  - Retrieve files as-needed for future analysis
- Ensuring naming conventions and folder structures match what downstream scripts expect
- Creating helper scripts for sorting, syncing, and inspecting files
- Supporting parallel workflows at:
  - The **folder level** (e.g., multiple sites at once)
  - The **file level** (e.g., multiple bands/images within a site)

## 🛠️ Tools & Tips

- Use Python `pathlib`, `os`, `re`, and `shutil` to inspect, rename, and organize files
- Automate transfer with `subprocess.run(["./gocmd", "put", ...])` and `get`
- Use structured logs (`print(f"[INFO] Sorting {file}...")`) to monitor progress
- Review existing code in `csv_extract_for_masked_envi.py` or `convolution_csv_merge.py` for examples

## 📁 Suggested Folder Structure

sorted_files/
└── convolution_resample/
├── Reflectance_Resample_Landsat_5_TM/
├── Reflectance_Resample_Landsat_7_ETM+/
└── ...

## 🚀 Goal

By organizing the raster outputs cleanly and ensuring consistent retrieval patterns, you’ll unlock reproducible analysis across all spatial scales and sensor types. Reach out if you need example naming conventions, `gocmd` helpers, or debugging support.

Welcome aboard! 🛰️
