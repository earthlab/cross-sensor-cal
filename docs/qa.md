# Quality Assurance (QA) panels

The `cscal-qa` command generates a summary PNG for each processed flight line. It
combines visual checks and lightweight metadata to confirm that every stage of the
pipeline completed successfully.

QA figures are re-generated on every run so they always reflect the current pipeline
settings. The spectral comparison panel converts reflectance to a unitless 0–1 scale and
shades VIS/NIR/SWIR regions to make interpretation easier.

```bash
cscal-qa --base-folder output_demo
# Optional: write all PNGs to a separate location
cscal-qa --base-folder output_demo --out-dir qa-panels
```

<!-- TODO: Replace this note with an actual QA panel screenshot when available. -->
<p align="center"><em>QA panel example coming soon.</em></p>

Each panel validates a specific part of the workflow:

- **Panel A – Raw ENVI RGB:** Verifies that the uncorrected ENVI export opens and renders
  with sensible colors and geospatial orientation.
- **Panel B – Patch-mean spectrum:** Overlays raw vs. BRDF+topo corrected spectra for a
  central patch, highlights VIS/NIR/SWIR wavelength ranges, and plots the difference to
  confirm the correction stage ran.
- **Panel C – Corrected NIR preview:** Displays a high-NIR band from the corrected cube to
  spot striping, nodata gaps, or other artifacts.
- **Panel D – Sensor thumbnails:** Shows downsampled previews of each convolved sensor
  product, confirming the resampling stage produced ENVI bandstacks.
- **Panel E – Parquet summary:** Lists the Parquet sidecars with file sizes so you can check
  that tabular exports were written.

Panels are written to `<flight_stem>_qa.png` inside each flight directory by default. Use
`--out-dir` to aggregate them in a centralized folder for sharing or archival.

## QA Dashboard

You can summarize QA performance across multiple flightlines with a single command:

```bash
cscal-qa-dashboard --base-folder output_fresh
```

This command:

- Aggregates all `*_qa_metrics.parquet` files,
- Computes per-flightline statistics,
- Writes a combined `qa_dashboard_summary.parquet`,
- Generates an overview plot (`qa_dashboard_summary.png`).

Each bar represents the fraction of flagged bands per flightline. Values above `0.25`
(25%) are marked with ⚠️ and may require review.

---

### ✅ After running

Expected artifacts in `output_fresh/`:

- `qa_dashboard_summary.parquet`
- `qa_dashboard_summary.png`

and log output similar to:

```
📊 Aggregated 12 flightlines (2400 rows)
✅ Computed summary for 12 flightlines
💾 Wrote aggregated QA summary → qa_dashboard_summary.parquet
🖼️ Saved dashboard → qa_dashboard_summary.png
```
