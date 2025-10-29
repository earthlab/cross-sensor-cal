# Quality Assurance (QA) panels

The `cscal-qa` command generates a summary PNG for each processed flight line. It
combines visual checks and lightweight metadata to confirm that every stage of the
pipeline completed successfully.

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
  central patch and plots their difference to confirm the correction stage ran.
- **Panel C – Corrected NIR preview:** Displays a high-NIR band from the corrected cube to
  spot striping, nodata gaps, or other artifacts.
- **Panel D – Sensor thumbnails:** Shows downsampled previews of each convolved sensor
  product, confirming the resampling stage produced ENVI bandstacks.
- **Panel E – Parquet summary:** Lists the Parquet sidecars with file sizes so you can check
  that tabular exports were written.

Panels are written to `<flight_stem>_qa.png` inside each flight directory by default. Use
`--out-dir` to aggregate them in a centralized folder for sharing or archival.
