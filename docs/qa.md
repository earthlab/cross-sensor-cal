# Quality Assurance (QA) panels

The `spectralbridge-qa` command now emits both a PNG panel and a machine-readable
`*_qa.json` file for every flightline. The PNG highlights spectral checks while
the JSON records the underlying metrics so you can track drift over time or feed
it into dashboards.

```bash
# Deterministic quick pass (≤25k sampled pixels per flightline)
spectralbridge-qa --base-folder output_demo --quick

# Exhaustive sampling with custom RGB mapping
spectralbridge-qa --base-folder output_demo --full --n-sample 150000 --rgb-bands 650,550,480
```

Key sections on the panel:

- **RGB quicklook** – automatically picks 660/560/490 nm (override with
  `--rgb-bands`). Red callouts overlay any flagged issues from the metrics.
- **Histograms** – pre vs post correction distributions with shared bins so you
  can judge how BRDF/topo shifts the scene.
- **Δ median vs wavelength** – bandwise medians with IQR ribbon; uses header
  wavelengths or sensor defaults.
- **Convolved scatter** – compares corrected data to any `*_convolved_envi`
  or `*_resampled_<sensor>_envi` outputs with a 1:1 reference line.

Each PNG lives alongside `<prefix>_qa.json`, which mirrors the
`QAMetrics` dataclass (`provenance`, `header`, `mask`, `correction`,
`convolution`, `negatives_pct`, `overbright_pct`, `issues`). When the brightness correction stage
runs, the JSON also lists per-band gain/offsets so the QA team can trace changes
back to illumination harmonisation.

Use `--out-dir` if you want to collect all PNG/JSON pairs into a single folder
for review. Re-running the command overwrites previous outputs, so a second pass
after fixing headers or rerunning corrections always reflects the current state.

## QA Dashboard

The legacy `cscal-qa-dashboard` command still expects `_qa_metrics.parquet`
files. Until the dashboard is updated to read the new JSON schema, keep legacy
parquet artifacts if you rely on that summary view.
