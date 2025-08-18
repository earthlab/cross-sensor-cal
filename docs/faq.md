# FAQ

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
### Which CRS should I use?
Use a projected CRS (e.g., UTM) consistent across inputs; the workflow reprojects mismatched scenes.

### How are missing ancillary inputs handled?
Stages log the omission and skip affected products; the run continues with available data.

### Can I resume a partially completed run?
Yes. Rerun the same stage and existing outputs are detected and skipped.

### How do I add support for a new sensor?
Create a sensor definition with its band metadata and register it in the configuration file.

### Why don't MESMA fractional totals sum to 1?
Residual fractions capture unmodeled materials and numerical error, so sums may differ from one.

### Where can I find logs and manifests?
Each run writes to `logs/` and `manifests/` directories inside the output folder.

### Do I have to run every processing stage?
No. You can run individual stages; each reads from the previous stage's outputs.

### How do I change the working directories?
Set `work_dir` and `output_dir` in the configuration to point to desired locations.

### How should I cite this project?
Reference the `CITATION.cff` file or the DOI listed in the repository.

### Are intermediate files cleaned up automatically?
Temporary products remain unless you enable the cleanup option in the configuration.

Last updated: 2025-08-18
<!-- FILLME:END -->
