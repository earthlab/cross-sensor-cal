# QA Panel

- Generated automatically after the **merge** stage during full runs.
- Filename: **`<prefix>_qa.png`** in the flightline folder.
- If troubleshooting: enable verbose logs and ensure ENVI headers parse for band indices/wavelengths.

!!! tip "Common fixes"
    - Ensure corrected/convolved ENVI headers have either numeric `wavelength` or `band names`.
    - Our parser falls back to sensor-default RGB if only `bands = N` exists.
