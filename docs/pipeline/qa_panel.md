# QA Panel

The QA panel is generated automatically after the **merge** stage. It always writes a
PNG and, when requested, a machine-readable JSON payload next to the corrected ENVI
products. Files follow the pattern **`<prefix>_qa.png`** and **`<prefix>_qa.json`**.

## What the panel shows

- **RGB quicklook (left column):** Auto-selects bands nearest 660/560/490 nm (or
  uses `--rgb-bands`) and annotates which indices were chosen. When the metrics
  detect issues, red callouts overlay the RGB panel.
- **Histograms (top-right):** Compares raw vs corrected reflectance distributions
  with a fixed binning and annotates the change in median and percent of negative
  pixels.
- **Δ vs λ (middle-right):** Plots the spectral median difference between raw and
  corrected cubes with an interquartile ribbon to highlight wavelengths with the
  largest adjustments.
- **Convolution scatter (bottom-right):** When convolved products exist, charts the
  expected vs observed reflectance with a 1:1 reference line and per-band RMSE/SAM
  annotations.
- **Footer:** Records provenance (flightline ID, UTC timestamp, package version,
  git SHA, and hashes of the inputs used for the panel).

## Quick vs full mode

`cscal-qa` now exposes deterministic sampling modes so you can trade speed for more
detail:

```bash
# Quick (default): ~10k samples, ideal for CI smoke tests
cscal-qa --base-folder output_demo --quick

# Full: up to 100k samples for production QA
cscal-qa --base-folder output_demo --full --n-sample 120000
```

Use `--rgb-bands "R,G,B"` (zero-based indices) to override the auto-selection and
`--no-save-json` when you only need the PNG panel.

## On-image callouts

Red callouts appear when the metrics report issues such as missing wavelengths,
non-monotonic wavelength arrays, low valid-mask coverage, or high fractions of
negative reflectance. These flags are also serialized into the JSON metrics for
downstream automation.

## JSON metrics

Every run writes `<prefix>_qa.json` alongside the PNG (unless `--no-save-json` is
supplied). The JSON follows the `QAMetrics` schema with provenance, header,
correction, convolution, and mask sections. Example snippet:

```json
{
  "provenance": {"flightline_id": "NEON_D13...", "git_sha": "abc123"},
  "header": {"n_bands": 426, "wavelengths_monotonic": true},
  "mask": {"valid_pct": 0.97},
  "correction": {"largest_delta_indices": [210, 211, 212]},
  "negatives_pct": 0.004,
  "issues": []
}
```

Use the JSON for automated regression tests or monitoring dashboards. It captures the
same issues highlighted in the panel callouts and provides hashes for reproducibility.
