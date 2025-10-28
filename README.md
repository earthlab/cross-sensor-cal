# Cross-Sensor Calibration

Cross-Sensor Calibration provides a Python pipeline for processing NEON Airborne Observation Platform hyperspectral flight lines and resampling them to emulate alternate sensors in a reproducible, scriptable workflow.

![Pipeline diagram](docs/img/pipeline.png)

## Quickstart

> As of v2.0, cross-sensor-cal no longer depends on HyTools. All NEON handling is internal and GPL-compatible.

Install the lightweight base package:

```bash
pip install cross-sensor-cal
```

Create a workspace and download a flight line:

```bash
mkdir -p data/SITE
cscal-download SITE --year-month 2021-06 --flight FLIGHT_LINE --output data
```

Run the end-to-end processing pipeline (see `--help` for all options):

```bash
cscal-pipeline --help
```

Programmatic invocation is equally simple:

```python
from pathlib import Path

from cross_sensor_cal.pipelines.pipeline import go_forth_and_multiply

go_forth_and_multiply(
    base_folder=Path("data/NIWO_2023-08"),
    site_code="NIWO",
    year_month="2023-08",
    flight_lines=[
        "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance",
    ],
)
```

If you need the full geospatial/hyperspectral toolchain (Rasterio, GeoPandas, Spectral, Ray, HDF5),
install the optional extras:

```bash
pip install "cross-sensor-cal[full]"
```

Feature availability by install type:

| Feature | Base | `[full]` |
|---|---|---|
| Core array ops (NumPy/Scipy) | ✅ | ✅ |
| Raster I/O (rasterio) | ⚠️ (not included) | ✅ |
| Vector I/O/ops (GeoPandas) | ⚠️ | ✅ |
| ENVI/HDR (spectral) | ⚠️ | ✅ |
| HDF5 (h5py) | ⚠️ | ✅ |
| Ray parallelism | ⚠️ | ✅ |

Replace `SITE` with a NEON site code and `FLIGHT_LINE` with an actual line identifier.

## 🚀 Updated Processing Workflow (October 2025)

The pipeline now runs in four clearly ordered, restart-safe stages:

1. **ENVI Export** — Converts NEON `.h5` reflectance files to ENVI format.  
   Skips automatically if `_envi.img/.hdr` already exist.
2. **Correction JSON** — Builds illumination/topographic correction parameters.  
   Written to `<stem>_brdfandtopo_corrected_envi.json` and reused if valid.
3. **BRDF + Topographic Correction** — Applies physical corrections using the precomputed JSON.  
   Produces `<stem>_brdfandtopo_corrected_envi.img/.hdr`.
4. **Convolution / Resampling** — Generates simulated reflectance for target sensors (Landsat, Sentinel, etc.).  
   Operates only on the corrected ENVI and skips finished sensors.

✅ **Idempotent Execution:**  
If you rerun the same `go_forth_and_multiply()` call, all previously completed steps are detected and skipped automatically. Invalid or incomplete files (e.g., from interrupted runs) are recomputed.

📜 **Logs now show:**
```
✅ ENVI export already complete for NEON_D13_NIWO_DP1_L019-1..., skipping
✅ Correction JSON already complete for NEON_D13_NIWO_DP1_L019-1..., skipping
✅ BRDF+topo correction already complete for NEON_D13_NIWO_DP1_L019-1..., skipping
✅ Landsat 8 OLI convolution already complete, skipping
🎉 Finished pipeline for NEON_D13_NIWO_DP1_L019-1...
```

💡 **Why this matters:**  
The pipeline can now resume safely after partial failures or system restarts without duplicating computation. It guarantees that convolution uses the corrected data products and that every output on disk has passed integrity checks.

## Pipeline Overview

1. **Locate NEON reflectance HDF5 files.** Download or copy the flightline `.h5` files from NEON AOP into your workspace before starting.
2. **Export HDF5 to ENVI (no HyTools).** `neon_to_envi_no_hytools()` opens each file with `NeonCube`, streams spatial tiles, and writes float32 BSQ ENVI rasters plus ancillary angle layers via `EnviWriter`. This stage runs entirely within cross-sensor-cal—no HyTools or Ray runtime dependencies remain. Existing `_envi.img/.hdr` pairs are validated and reused.
3. **Persist correction parameters.** `build_and_write_correction_json()` inspects the flightline geometry and writes `<flightline>_brdfandtopo_corrected_envi.json`. The JSON is regenerated only when missing or invalid.
4. **Apply topographic and BRDF correction.** Tiles from the uncorrected ENVI cube are corrected using slope/aspect rasters and the stored parameters. Optional `brightness_offset` is applied before writing `<flightline>_brdfandtopo_corrected_envi.img/.hdr`, whose header includes spatial metadata and true wavelength/FWHM information. Existing outputs are checked for completeness before skipping.
5. **Convolve to simulated sensors.** `convolve_resample_product()` memmaps the corrected cube, multiplies each tile by sensor-specific spectral response functions from `cross_sensor_cal/data/`, and emits one ENVI product per simulated sensor band set. Finished products are validated and skipped on reruns.

## Data Products

- **`<flightline>_directional_reflectance.img/.hdr`** – Uncorrected directional reflectance exported from the NEON HDF5 (no topographic, BRDF, or spectral convolution applied). Ancillary angle rasters are written alongside for downstream use.
- **`<flightline>_brdf_model.json`** – Intermediate BRDF coefficients referenced by the correction JSON. Regenerated only when inputs change.
- **`<flightline>_brdfandtopo_corrected_envi.json`** – Correction parameters describing illumination geometry, BRDF coefficients, and ancillary stats for the flightline. Reused whenever it remains valid on disk.
- **`<flightline>_brdfandtopo_corrected_envi.img/.hdr`** – Float32 BSQ cube with both topographic and BRDF corrections applied (and any configured brightness offset). The header preserves spatial metadata plus wavelength lists, FWHM values, and wavelength units that power later resampling steps.
- **`<flightline>_resampled_<sensor>.img/.hdr`** – Simulated multispectral products generated by spectrally convolving the corrected cube with sensor SRFs. These rasters inherit map metadata from the corrected cube but represent only the spectral convolution stage (no additional corrections).

## Install

Cross-Sensor Calibration depends on GDAL, PROJ, Ray, h5py, and optional geospatial stacks such as Rasterio and GeoPandas. We recommend the Conda workflow below because it installs the required native libraries automatically. If you prefer a pure `pip` workflow, install system packages for GDAL/PROJ first (e.g., `brew install gdal` on macOS or `apt-get install gdal-bin libgdal-dev proj-bin` on Debian/Ubuntu).

### Conda

```bash
conda create -n cscal python=3.10 gdal proj
conda activate cscal
pip install -e .
```

### uv/pip

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

If GDAL wheels are unavailable for your platform, install it from Conda-forge and then point `pip` at the Conda environment by exporting `CPLUS_INCLUDE_PATH` and `C_INCLUDE_PATH`.

## Documentation

Browse the full documentation site at
[earthlab.github.io/cross-sensor-cal](https://earthlab.github.io/cross-sensor-cal).
The site is built with MkDocs Material and automatically deployed to GitHub
Pages.

Key entry points:

- [Overview](docs/overview.md)
- [Quickstart](docs/quickstart.md)
- [Stage 01 Raster Processing](docs/stage-01-raster-processing.md)
- [Stage 02 Sorting](docs/stage-02-sorting.md)
- [Stage 03 Pixel Extraction](docs/stage-03-pixel-extraction.md)
- [Stage 04 Spectral Library](docs/stage-04-spectral-library.md)
- [Stage 05 MESMA](docs/stage-05-mesma.md)

## Support Matrix

| Python | OS            | GDAL | Ray |
|--------|---------------|------|-----|
| 3.10+  | Linux, macOS  | 3.4+ | 2.0+ |

## How to cite

If you use Cross-Sensor Calibration in your research, please cite the project:

```
Earth Lab Data Innovation Team. (2025). Cross-Sensor Calibration (Version 0.1.0) [Software]. University of Colorado Boulder. https://github.com/earthlab/cross-sensor-cal
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## License and Citation

Distributed under the GPLv3 License. Please cite the project using [CITATION.cff](CITATION.cff).

