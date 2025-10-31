# Quickstart

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
Choose your path:

=== "CLI only"
    ```bash
    cscal-pipeline \
      --base-folder output_demo \
      --site-code NIWO \
      --year-month 2023-08 \
      --product-code DP1.30006.001 \
      --flight-lines NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance \
      --max-workers 2
    ```

=== "Python control"
    ```python
    from cross_sensor_cal import go_forth_and_multiply

    go_forth_and_multiply(
        base_folder="output_demo",
        site_code="NIWO",
        year_month="2023-08",
        product_code="DP1.30006.001",
        flight_lines=["NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"],
        max_workers=2,
    )
    ```

=== "HPC / many tiles"
    - Use `--max-workers` conservatively to avoid RAM pressure.
    - Re-run the same command to resume; completed stages are skipped.
    - If `/dev/shm` is small, limit concurrency.
<!-- FILLME:END -->
