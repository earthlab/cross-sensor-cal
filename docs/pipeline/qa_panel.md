# QA panel

The QA panel now couples a metrics JSON file with the annotated PNG so that
engineering and science teams can track both spectral statistics and the visual
context for every flightline.

---

# QA Panel and Validation Tests

The QA panel is the final diagnostic step of the **Cross-Sensor Calibration pipeline**.  
It provides both a visual and quantitative summary of how well each product behaved through all correction stages (topographic, BRDF, brightness, convolution).  

---

## What the QA Tests Measure

| Test | What It Checks | Why It Matters |
|------|----------------|----------------|
| **Reflectance Range & Negatives %** | Fraction of pixels below 0 or above 1.2 | Reflectance should remain physically bounded. Large negative or >1.2 values indicate poor radiometric scaling or unmasked clouds/shadows. |
| **Header & Wavelength Integrity** | Presence, count, and monotonicity of `wavelength` values in ENVI headers | Ensures each band is correctly aligned; missing or non-monotonic wavelengths break convolution and spectral analyses. |
| **ΔReflectance (Pre→Post Correction)** | Median and IQR difference in reflectance before and after BRDF/topo correction | Quantifies how much the correction changed the data. Large deltas in flat terrain suggest over-correction; near-zero deltas in complex terrain may suggest under-correction. |
| **Brightness Normalization (if applied)** | Per-band gain and offset used in brightness correction | Tracks whether correction parameters remain within expected limits (e.g., gain ∈ [0.9, 1.1]). Large deviations imply inconsistent illumination normalization. |
| **Convolution Accuracy (per target sensor)** | RMSE and Spectral Angle Mapper (SAM) between expected vs computed bands | Confirms spectral resampling is physically consistent. High RMSE or large SAM (>0.05 radians) indicates wavelength misalignment or incorrect response functions. |
| **Mask Coverage** | % of valid pixels used for metrics | Low valid coverage (<60%) signals missing masks or unfiltered NaNs. |
| **Histogram Shape Consistency** | Visual histogram overlay of pre/post corrections | Skewed or bimodal shapes suggest scene heterogeneity or masking issues. |

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
| Reflectance > 1.2 | < 0.5 % | 0.5–2 % | > 2 % |
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
    if q["negatives_pct"] > 2.0 or q["mask"]["valid_pct"] < 60:
        bad.append(f)
print("Tiles needing review:", bad)

```

