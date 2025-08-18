# Extending

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
## Add a sensor

### Overview
This recipe shows how to integrate a new sensor into the cross‑sensor calibration
workflow.

### Prerequisites
- Spectral response function (SRF) curves for the sensor.
- Familiarity with the project's naming conventions.

### Step-by-step tutorial
1. **Add SRFs.** Place the sensor's SRF files in the data directory and register
   them with the SRF loader.
2. **Update resampling/convolution mapping.** Extend the resampling and
   convolution dictionaries so the pipeline knows how to transform the sensor's
   bands.
3. **Update the naming map.** Insert the sensor's canonical name and band
   identifiers into the shared naming map used across modules.
4. **Add a golden test and validation checklist.**
   - Create a golden test case with expected outputs.
   - Document validation steps to confirm the sensor behaves correctly.

### Next steps
Run the full validation suite and submit a pull request for review.

Last updated: 2025-08-18
<!-- FILLME:END -->
