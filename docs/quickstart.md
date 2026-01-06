# Quickstart

This Quickstart gives you two ways to run the pipeline end-to-end:

1. **CLI path** – run `spectralbridge-pipeline` from a terminal  
2. **Notebook path** – run the pipeline inside Jupyter

Both produce the same ENVI, Parquet, and QA artifacts.

---

## Install

Install from PyPI:

    pip install spectralbridge

For Ray support:

    pip install "spectralbridge[full]"

> Upgrading from older versions? ``cross_sensor_cal`` imports and ``cscal-*``
> commands still work, but new examples use ``spectralbridge`` imports and
> ``spectralbridge-*`` entry points.

---

## 1. CLI path

Use the CLI if you run jobs on your laptop, server, or HPC cluster.

### Run a NEON flight line

Choose an output directory:

    BASE=output_quickstart
    mkdir -p "$BASE"

Run the pipeline:

    spectralbridge-pipeline \
      --base-folder "$BASE" \
      --site-code NIWO \
      --year-month 2023-08 \
      --product-code DP1.30006.001 \
      --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
      --max-workers 2 \
      --engine thread

The first time this runs, it will:

- download NEON HDF5 tiles  
- export ENVI cubes  
- apply topographic + BRDF correction  
- convolve to Landsat-style reflectance  
- write Parquet tables  
- produce QA PNG, PDF, and JSON summaries  

If rerun, completed stages are skipped safely.

### Inspect QA files (example on macOS)

    open $BASE/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/*_qa.png
    open $BASE/NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance/*_qa.pdf

---

## 2. Notebook path (Jupyter)

Use this if you want an interactive, reproducible workflow.

### Run the pipeline from Python

In a notebook cell:

    from spectralbridge import go_forth_and_multiply

    base = "output_quickstart_py"

    go_forth_and_multiply(
        base_folder=base,
        site_code="NIWO",
        year_month="2023-08",
        product_code="DP1.30006.001",
        flight_lines=["NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"],
        max_workers=2,
        engine="thread",
    )

### Preview the merged Parquet table

    import duckdb, os

    fl = "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"
    merged = os.path.join(base, fl, f"{fl}_merged_pixel_extraction.parquet")

    duckdb.query(f"SELECT * FROM '{merged}' LIMIT 5").df()

For a more complete notebook-style walkthrough, see:  
**Usage → Jupyter notebook example**

---

## Next steps

- [Why cross-sensor calibration?](concepts/why-calibration.md)  
- [Tutorials](tutorials/neon-to-envi.md)
- [Pipeline overview & stages](pipeline/stages.md)
- [Working with Parquet outputs](usage/parquet.md)

---
