# Benchmarks

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
## Reproducible timing experiment design

- Pin package versions and record hardware specifications.
- Use fixed random seeds and keep the system load constant.
- Run each benchmark multiple times, reporting mean and variance.
- Save the exact command line and configuration for future runs.

## Ray cluster knobs

When scaling benchmarks on Ray, adjust:

- `--num-cpus` and `--num-gpus` to control available resources.
- `--object-store-memory` for large in-memory datasets.
- `--temp-dir` to point to fast local storage.
- `--dashboard-port` to monitor cluster status.

## I/O bottleneck tips

- Chunk rasters along the row and column dimensions so each worker reads contiguous blocks.
- Enable compression such as LZW or DEFLATE to reduce disk usage and transfer time.
- Cache intermediate products or use memory-mapped files to avoid repeated reads.
<!-- FILLME:END -->
