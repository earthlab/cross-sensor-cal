# Documentation Drift Report

## Naming rule
- Merged parquet naming in code: **<prefix>_merged_pixel_extraction.parquet**

## Missing items in docs
- Outputs: —
- Sensors: micasense_to_match_oli_and_oli, micasense_to_match_oli_oli, micasense_to_match_tm_etm+

## Stale items in docs
- Outputs: _brdfandtopo_corrected_envi.parquet, _landsat_etm+_envi.img, _landsat_oli_envi.parquet, _landsat_tm_envi.hdr, _landsat_tm_envi.img, _micasense*_envi.img, _micasense_envi.hdr, _micasense_envi.img, _qa.json
- Sensors: MicaSense, MicaSense,, MicaSense,,, MicaSense,,,, MicaSense,,,,, MicaSense,,,,,, MicaSense,,,,,,, Micasense,, Micasense,,, Micasense,,,, Micasense,,,,, Micasense,,,,,, Micasense,,,,,,, micasense'], micasense'],, micasense'],,, micasense'],,,, micasense'],,,,, micasense'],,,,,, micasense*_envi.img`, micasense_envi.hdr, micasense_envi.hdr,, micasense_envi.hdr,,, micasense_envi.hdr,,,, micasense_envi.hdr,,,,, micasense_envi.hdr,,,,,, micasense_envi.img, micasense_envi.img,, micasense_envi.img,,, micasense_envi.img,,,, micasense_envi.img,,,,, micasense_envi.img,,,,,, micasense_envi.img/.hdr/.parquet, micasense_envi.img/.hdr/.parquet,, micasense_envi.img/.hdr/.parquet,,, micasense_envi.img/.hdr/.parquet,,,, micasense_envi.img/.hdr/.parquet,,,,, micasense_envi.img/.hdr/.parquet,,,,,, micasense_to_match_oli_and_oli,, micasense_to_match_oli_and_oli,,, micasense_to_match_oli_and_oli,,,, micasense_to_match_oli_and_oli,,,,, micasense_to_match_oli_and_oli,,,,,, micasense_to_match_oli_oli,, micasense_to_match_oli_oli,,, micasense_to_match_oli_oli,,,, micasense_to_match_oli_oli,,,,, micasense_to_match_oli_oli,,,,,, micasense_to_match_tm_etm+,,,,,, micasense`)

## Entry points (pyproject)
{
  "spectralbridge-download": "spectralbridge.cli:download_main",
  "spectralbridge-pipeline": "spectralbridge.cli.pipeline_cli:main",
  "spectralbridge-qa": "spectralbridge.cli.qa_cli:main",
  "spectralbridge-recover-raw": "spectralbridge.cli.recover_cli:main",
  "spectralbridge-qa-dashboard": "spectralbridge.qa_dashboard:main",
  "spectralbridge-merge-duckdb": "spectralbridge.merge_duckdb:main"
}

## CLI flags discovered
{
  "topo_and_brdf_correction.py": [
    "--config_file"
  ],
  "merge_duckdb.py": [
    "--corrected-glob",
    "--data-root",
    "--flightline-glob",
    "--no-qa",
    "--original-glob",
    "--out-name",
    "--resampled-glob",
    "--write-feather"
  ],
  "qa_metrics.py": [
    "--base-folder",
    "--flight-stem"
  ],
  "standard_resample.py": [
    "--hdr_path",
    "--json_file",
    "--output_path",
    "--resampling_file_path",
    "--sensor_type"
  ],
  "qa_dashboard.py": [
    "--base-folder",
    "--out-parquet",
    "--out-png"
  ],
  "neon_to_envi.py": [
    "--brightness-offset"
  ],
  "pipeline_cli.py": [
    "--base-folder",
    "--brightness-offset",
    "--flight-lines",
    "--max-workers",
    "--product-code",
    "--resample-method",
    "--site-code",
    "--year-month"
  ],
  "recover_cli.py": [
    "--base-folder",
    "--brightness-offset"
  ],
  "qa_cli.py": [
    "--base-folder",
    "--no-overwrite",
    "--out-dir",
    "--overwrite"
  ],
  "pipeline.py": [
    "--brightness-offset",
    "--no-sync",
    "--polygon_layer_path",
    "--reflectance-offset",
    "--remote-prefix",
    "--resample-method",
    "--verbose"
  ],
  "download.py": [
    "--flight",
    "--output",
    "--product",
    "--year-month"
  ]
}

## Stage markers found in logs
- [spectralbridge-qa] ✅ QA panels written to: {target.resolve()}
- ⏭️ Parquet already present for %s -> %s (skipping)
- ⚠️  QA panel generation failed for %s: %s
- ⚠️ Cannot create Parquet work directory for %s: %s
- ⚠️ Cannot export Parquet for %s because .hdr is missing or empty
- ⚠️ Cannot export Parquet for %s because .img is missing or empty
- ⚠️ Cannot locate work directory for Parquet export: %s
- ⚠️ Failed Parquet export for %s: %s
- ⚠️ Merge failed for {future_map[future]}: {exc}
- ⚠️ QA panel after merge failed for {flightline_dir.name}: {e}
- ✅ BRDF+topo correction already complete for %s -> %s / %s (skipping)
- ✅ BRDF+topo correction already complete for %s, skipping
- ✅ BRDF+topo correction completed for %s -> %s / %s
- ✅ Download complete for %s → %s
- ✅ ENVI export already complete for %s -> %s / %s (skipping heavy export)
- ✅ ENVI export completed for %s -> %s / %s
- ✅ Parquet stage complete for %s
- ✅ Wrote Parquet for %s -> %s
- 🌐 Downloading %s (%s, %s) into %s ...
- 🎉 Finished pipeline for %s
- 🎉 Finished pipeline for %s (parallel worker join)
- 🎯 Convolving corrected reflectance for %s
- 📦 ENVI export not found or invalid for %s, generating from %s
- 📦 Parquet export for %s ...
- 🔎 ENVI export target for %s is %s / %s
- 🖼️  Overwriting QA panel -> %s
- 🖼️  QA panel written → {prefix}_qa.png
- 🖼️  Writing QA panel -> %s
- 🖼️  Wrote QA panel for %s -> %s

TODO: Review sensor labels in docs for formatting issues.
