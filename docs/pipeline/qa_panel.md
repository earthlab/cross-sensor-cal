# QA panel

The QA panel couples a metrics JSON file with an annotated PNG so that
engineering and science teams can track both spectral statistics and the visual
context for every flightline. See [How to Interpret the Panel](#how-to-interpret-the-panel)
and the [validation reference](../reference/validation.md) for deeper guidance
on each metric.

Additional QA products created from the merged Parquet:

- `<prefix>_merged__BY_SENSOR_vs_NEON_directional_BRDFTopo.png`  
  Sensor-by-sensor scatter panels versus NEON directional BRDFTopo.

- `<prefix>_merged__MS_vs_Landsat_FIXED.png`  
  MicaSense-matched (X) versus Landsat (Y) scatter panels by band.

## Multi-page QA report

In addition to the single PNG QA panel (`<prefix>_qa.png`), the pipeline now
writes a multi-page PDF report (`<prefix>_qa.pdf`) with three pages:

1. **Page 1 – ENVI overview**  
   One row with one panel per ENVI product. This is a quick visual check that
   all ENVI files exist and render correctly.

2. **Page 2 – Topographic & BRDF diagnostics**  
   - Row 1: pre vs post histograms and Δ median vs wavelength for the combined
     topographic + BRDF correction stage.  
   - Row 2: summaries of topographic (slope/aspect) and BRDF geometry
     (solar/sensor angles) derived from the correction JSON.

3. **Page 3 – Remaining QA diagnostics**  
   - Convolution scatter plots (expected vs computed bands).  
  - Header and wavelength integrity summary (flags when sensor defaults are used).
  - Mask coverage, negatives %, and >1.2 reflectance % summary.
   - Issues/warnings, including brightness coefficients that were applied.

---

# QA Panel and Validation Tests

The QA panel is the final diagnostic step of the **Cross-Sensor Calibration pipeline**.  
It provides both a visual and quantitative summary of how well each product behaved through all correction stages (topographic, BRDF, brightness, convolution).  

---

## What the QA Tests Measure

| Test | What It Checks | Why It Matters |
|------|----------------|----------------|
| **Reflectance Range (negatives & >1.2 %)** | Fraction of pixels below 0 or above 1.2 | Reflectance should remain physically bounded. Large negative or >1.2 values indicate poor radiometric scaling or unmasked clouds/shadows. |
| **Header & Wavelength Integrity** | Presence, count, monotonicity, and provenance of `wavelength` values in ENVI headers | Ensures each band is correctly aligned; missing, non-monotonic, or defaulted wavelengths break convolution and spectral analyses. |
| **ΔReflectance (Pre→Post Correction)** | Median and IQR difference in reflectance before and after BRDF/topo correction | Quantifies how much the correction changed the data. Large deltas in flat terrain suggest over-correction; near-zero deltas in complex terrain may suggest under-correction. |
| **Brightness Normalization (if applied)** | Per-band gain and offset used in brightness correction | Tracks whether correction parameters remain within expected limits (e.g., gain ∈ [0.9, 1.1]). Large deviations imply inconsistent illumination normalization. |
| **Convolution Accuracy (per target sensor)** | RMSE and Spectral Angle Mapper (SAM) between expected vs computed bands | Confirms spectral resampling is physically consistent. High RMSE or large SAM (>0.05 radians) indicates wavelength misalignment or incorrect response functions. |
| **Mask Coverage** | % of valid pixels used for metrics | Low valid coverage (<60%) signals missing masks or unfiltered NaNs. |
| **Histogram Shape Consistency** | Visual histogram overlay of pre/post corrections | Skewed or bimodal shapes suggest scene heterogeneity or masking issues. |

---

### Brightness coefficients

When NEON data are convolved to Landsat bands, we optionally apply small
per-band brightness adjustments so that Landsat-like products match a
MicaSense reference.

- Coefficients are stored in `landsat_to_micasense.json` (units: percent).
- The adjustment is multiplicative:

  `L_adj = L_raw * (1 + coeff / 100)`, where negative coefficients darken
  Landsat bands slightly.

- Applied coefficients are recorded in the QA JSON under
  `brightness_coefficients.landsat_to_micasense` and displayed on Page 3 of
  the QA PDF.

This makes it easy to verify when a brightness adjustment was applied and to
audit the exact per-band values.

---

## Why These Tests Are Appropriate

These diagnostics are **physically interpretable** and **sensor-agnostic**:

- **Radiometric realism:** Reflectance outside [0,1.2] is physically implausible and signals calibration drift or shadow contamination.  
- **Spectral continuity:** Monotonic wavelengths ensure that per-band corrections and convolutions follow real sensor band order.  
- **Conservation principle:** ΔReflectance checks whether corrections preserve brightness globally (no systematic over-darkening).  
- **Geometric realism:** Mask coverage and illumination correlation prevent interpreting shadowed or topographically inverted pixels as valid reflectance.  
- **Spectral fidelity:** RMSE and SAM compare corrected spectra to expected bandpasses, verifying physical sensor compatibility.

---

## How to Interpret the Panel

Each panel includes:

1. **Left:** RGB quicklook using auto-selected bands (660, 560, 490 nm).  
   - *Uniform color tone*: good illumination normalization.  
   - *Patchy shadows or gradients*: check DTM alignment or BRDF model.
2. **Top-right:** Pre (gray) vs Post (green) histograms.  
   - *Slight narrowing*: normal (flattening illumination gradients).  
   - *Severe shift left/right*: over- or under-correction.
3. **Middle-right:** ΔReflectance vs Wavelength curve (median ± IQR).  
   - *Smooth near-zero line*: ideal.  
   - *Large band-specific spikes*: band-specific sensor noise or cloud edges.
4. **Bottom-right:** Convolution scatter (expected vs computed).  
   - *Points close to 1:1 line*: good.  
   - *Systematic bias or slope ≠ 1*: check wavelength alignment or FWHM mismatch.
5. **Footer:** Metadata (flightline ID, date, package version, git SHA).

---

## Quantitative Thresholds for “Not Good”

| Metric | Acceptable Range | “Needs Review” | “Problematic” |
|---------|------------------|----------------|----------------|
| Negatives % | < 0.5 % | 0.5–2 % | > 2 % |
| >1.2 reflectance % | < 0.5 % | 0.5–2 % | > 2 % |
| ΔReflectance median | | < 0.02 (normally good) | > 0.05 (over/under-correction) |
| Brightness gain | 0.9–1.1 | 0.85–0.9 / 1.1–1.15 | < 0.85 or > 1.15 |
| Convolution RMSE | < 0.02 | 0.02–0.05 | > 0.05 |
| SAM (radians) | < 0.03 | 0.03–0.05 | > 0.05 |
| Mask coverage | > 80 % | 60–80 % | < 60 % |

---

## Deciding When a Product Fails QA

Mark a product as **Needs Review** when:
- ≥ 2 metrics fall in the “Needs Review” column, **or**
- Any single metric hits the “Problematic” range.

Mark a product as **Fail** when:
- > 10 % of bands exceed thresholds (e.g., ΔReflectance > 0.05),
- Wavelengths are missing or non-monotonic,
- Convolution RMSE > 0.05 **and** SAM > 0.05,
- Mask coverage < 60 %.

All QA results are summarized in the sidecar JSON (`*_qa.json`), enabling programmatic filtering.

---

## Next Steps After QA Flags

| Issue | Likely Cause | Recommended Fix |
|-------|---------------|----------------|
| Many negatives or high reflectance | Mis-scaled input, wrong gain offset | Re-run brightness correction or check calibration constants. |
| Non-monotonic wavelengths | Corrupted or edited header | Re-export ENVI or fix `wavelength` list manually. |
| Large ΔReflectance | Over-aggressive BRDF correction | Adjust BRDF parameters or review illumination mask. |
| High RMSE/SAM | Wrong sensor response curves | Verify target sensor config file. |
| Low mask coverage | Cloud or DTM mask mismatch | Improve masking or fill small gaps before QA. |

---

## Automating QA Review

Each QA JSON includes numeric thresholds.
You can quickly summarize or flag tiles programmatically:

```python
import json, glob
bad = []
for f in glob.glob("*/**/*_qa.json", recursive=True):
    q = json.load(open(f))
    if (
        q["negatives_pct"] > 2.0
        or q.get("overbright_pct", 0) > 2.0
        or q["mask"]["valid_pct"] < 60
    ):
        bad.append(f)
print("Tiles needing review:", bad)

```

