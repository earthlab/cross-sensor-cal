# Debugging ENVI Naming Issues

This document provides guidance for troubleshooting the NEON hyperspectral
pipeline when files fail to move through the BRDF correction and resampling
steps.

## Pipeline overview

1. `download_neon_flight_lines` retrieves raw NEON HDF5 flight lines.
2. `flight_lines_to_envi` converts each HDF5 file to ENVI format.
3. `generate_config_json` scans for `_reflectance_envi` products and writes
   configuration JSON files.
4. `topo_and_brdf_correction` uses those configs to create corrected imagery.
5. Resampling functions translate the corrected imagery into other sensor
   formats.

## Typical failure points

- Missing or unexpected filename suffixes prevent `generate_config_json` from
  locating ENVI outputs.
- If no config files are written, BRDF correction and resampling steps
  silently skip.

## Suffix handling

The pipeline expects ENVI outputs with `_reflectance_envi` in their names.
Alternate HDF5 inputs may omit this suffix in their metadata, producing files
that do not match the strict pattern and are ignored downstream.

## Debugging tips

- After ENVI conversion the pipeline lists every `*envi.hdr` file it created.
  Verify that the expected suffix appears.
- `generate_config_json` reports the paths of each config it writes and warns if
  none were generated.
- `NEONReflectanceConfigFile.find_in_directory` can run with `verbose=True` to
  show which files were ignored and why.

Use these logs and simple file listings to ensure that filenames match the
patterns expected by downstream steps.

