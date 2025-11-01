# QA panel

The QA panel now couples a metrics JSON file with the annotated PNG so that
engineering and science teams can track both spectral statistics and the visual
context for every flightline. Each panel contains four sections:

1. **RGB quicklook** – 660/560/490 nm bands (or your overrides) stretched to
   unitless reflectance. Red callouts appear when metrics flag issues such as
   missing header keys or large correction deltas.
2. **Pre vs post histograms** – fixed-bin histograms that show how the BRDF and
   topographic corrections shift the reflectance distribution.
3. **Δ median vs wavelength** – per-band median change with interquartile ribbons
   (requires wavelengths from the ENVI header or sensor defaults).
4. **Convolved scatter** – 1:1 scatter plot comparing corrected cubes to any
   *_convolved_envi products, annotated with RMSE/SAM in the JSON.

## Running the panel generator

Use the QA CLI to regenerate PNG+JSON pairs. Quick mode is deterministic and
caps sampling at ~25k pixels, while full mode uses the supplied sample budget.

```bash
cscal-qa --base-folder workspace/processed --quick
# or a more exhaustive pass
cscal-qa --base-folder workspace/processed --full --n-sample 150000
```

Flags worth remembering:

* `--rgb-bands 650,550,480` to override the RGB targets.
* `--no-save-json` to skip emitting the metrics file (PNG only).

## Metrics JSON

Each PNG is accompanied by `<prefix>_qa.json`. The JSON mirrors the
`QAMetrics` dataclass and captures provenance, header health, mask coverage, and
spectral deltas. Below is a trimmed example:

```json
{
  "provenance": {
    "flightline_id": "NEON_TEST_FLIGHT",
    "created_utc": "2024-04-08T12:00:00Z",
    "package_version": "2.2.0",
    "git_sha": "abc1234",
    "input_hashes": {"NEON_TEST_FLIGHT_envi.img": "..."}
  },
  "header": {
    "n_bands": 426,
    "wavelength_unit": "Nanometers",
    "wavelength_source": "header"
  },
  "mask": {"valid_pct": 99.2},
  "correction": {"delta_median": [-0.01, 0.02, ...]},
  "convolution": [{"sensor": "oli", "sam": 0.004}],
  "negatives_pct": 0.3,
  "issues": []
}
```

When the pipeline runs brightness correction, the `correction` block will also
contain per-band gain/offset diagnostics to support downstream QA reviews.

## Troubleshooting cues

* **Red callout referencing header gaps** – confirm the ENVI header exports
  `wavelength`, `fwhm`, and `band names`.
* **Large Δ median spikes** – revisit correction coefficients or rerun with a
  denser sample (`--full`).
* **High negative percentage** – inspect masks and solar-geometry inputs for the
  BRDF/topo stage.

Re-running `cscal-qa` after a fix overwrites both the PNG and the JSON so you
always have up-to-date metrics.
