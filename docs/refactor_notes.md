# Cross-Sensor-Cal Refactor Notes

## Overview
The 2024 refactor eliminates the runtime dependency on HyTools while retaining required GPL attribution. The codebase now vendors the minimal NEON readers, correction utilities, and ENVI writers so production pipelines operate without external HyTools installations. Remaining GPL notices accompany the adapted modules.

Key updates include new in-repo loaders (`neon_cube.NeonCube`), correction utilities (`corrections`), ENVI export helpers (`envi_writer`), and sensor resampling (`resample`). The orchestration pipeline (`pipelines.pipeline`) was rewritten to consume these modules and expose a streamlined `go_forth_and_multiply` entry point. Associated tests now validate the new Neon cube abstraction.

By folding the HyTools functionality into maintained modules, developers avoid brittle environment pinning, dramatically shrink cold-start time when provisioning workers, and gain direct control over performance-critical code paths. Internal APIs are clearer, dependency graphs simpler, and profiling shows improved memory locality when iterating NEON chunks.

## New Module Structure
| Module | Purpose | Key Classes/Functions |
|:--|:--|:--|
| neon_cube.py | Loads NEON HDF5 reflectance into memory; handles chunk iteration and metadata | `NeonCube` |
| corrections.py | Applies topographic and BRDF correction | `apply_topo_correct`, `apply_brdf_correct` |
| envi_writer.py | Writes corrected ENVI BSQ cubes with headers | `EnviWriter`, `build_envi_header_text` |
| resample.py | Convolves hyperspectral reflectance to other sensors | `resample_chunk_to_sensor` |
| pipeline.py | Coordinates the full processing sequence | `go_forth_and_multiply` |
| tests/test_neon_cube.py | Unit tests NeonCube | `pytest` fixtures |

## Licensing
Portions of `neon_cube.py`, `corrections.py`, and `envi_writer.py` remain derived from HyTools and therefore continue to carry GPLv3 attribution. Each file embeds explicit credit to the HyTools authors and identifies the adapted functions to satisfy the license obligations while hosting the logic internally.

## Developer Workflow
1. Create or locate a NEON HDF5 reflectance file.
2. Run the pipeline:
   ```bash
   python -m cross_sensor_cal.pipelines.pipeline --input neon_dir --output corrected_dir
   ```
3. Optional: run tests
   ```bash
   pytest -v tests/
   ```
4. Optional: resample to a sensor (Sentinel-2, Landsat, etc.)
   ```python
   from cross_sensor_cal.resample import resample_chunk_to_sensor
   ```

## Tests and Validation
`tests/test_neon_cube.py` verifies the NeonCube loader constructs spectral metadata, enforces geometry assumptions, tiles reflectance chunks, and builds ENVI headers. Additional end-to-end validation is planned in `tests/test_pipeline_end_to_end.py` to cover pipeline orchestration and file-system side effects once realistic fixtures are finalised.

## Deprecated Files
The following files still reference HyTools-era logic and require follow-up cleanup or deletion:
- src/cross_sensor_cal/standard_resample.py
- src/cross_sensor_cal/topo_and_brdf_correction.py
- src/cross_sensor_cal/hytools_compat.py
- src/cross_sensor_cal/neon_to_envi.py
- bin/neon_to_envi.py
- tests/test_hytools_preflight.py
- tests/test_hytools_compat.py
- spectral_unmixing_tools_original.py
- BRDF-Topo-HyTools/README.md
- Raster_processing.ipynb
- Untitled.ipynb
- ty_notebooks/Macrosystem_polygon_extractions.ipynb
- Macrosystem_polygon_extractions.ipynb
- BRDF-Topo-HyTools (module contents)

## Files to Review or Remove
- src/cross_sensor_cal/standard_resample.py — imports HyTools resampling (fallback only partly rewritten)
- src/cross_sensor_cal/topo_and_brdf_correction.py — legacy wrapper around HyTools correction workflow
- src/cross_sensor_cal/hytools_compat.py — discovery shims for HyTools imports
- src/cross_sensor_cal/neon_to_envi.py — depends on HyTools compatibility layer
- bin/neon_to_envi.py — CLI invoking HyTools compatibility layer
- tests/test_hytools_preflight.py — ensures HyTools installed
- tests/test_hytools_compat.py — mocks HyTools namespace discovery
- spectral_unmixing_tools_original.py — imports `hytools as ht`
- BRDF-Topo-HyTools/README.md — documents HyTools requirements
- Raster_processing.ipynb — clones HyTools and patches base.py
- Untitled.ipynb — notebook cells referencing hytools
- ty_notebooks/Macrosystem_polygon_extractions.ipynb — patches HyTools base.py
- Macrosystem_polygon_extractions.ipynb — same HyTools patch instructions
