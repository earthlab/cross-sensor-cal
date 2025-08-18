# Validation

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
The validation procedure ensures quality across the pipeline.

### Stage 01 – Raster Processing
- [ ] Confirm input rasters exist and are readable.
- [ ] Check projection and resolution match expected values.
- [ ] Verify no bands contain all nodata values.
- [ ] Ensure output rasters write successfully.

```python
import rasterio, numpy as np
with rasterio.open("image.tif") as src:
    data = src.read()
    assert not np.isnan(data).all(axis=(1, 2))
```

### Stage 02 – Sorting
- [ ] Confirm filenames follow `YYYYMMDD_sensor.tif` pattern.
- [ ] Check chronological ordering after sorting.
- [ ] Verify number of files per date matches expected counts.

```python
import pandas as pd, glob
files = sorted(glob.glob("sorted/*.tif"))
dates = pd.to_datetime([f.split("_")[0] for f in files])
assert dates.is_monotonic_increasing
```

### Stage 03 – Pixel Extraction
- [ ] Ensure sample coordinates fall within raster bounds.
- [ ] Validate pixel value ranges for each band.
- [ ] Cross-check sample count with original list.

```python
import numpy as np, pandas as pd
pixels = pd.read_csv("pixels.csv")
assert ((pixels['x']>=0) & (pixels['y']>=0)).all()
assert pixels.drop(columns=['x','y']).apply(np.isfinite).all().all()
```

### Stage 04 – Spectral Library
- [ ] Verify spectra length equals number of bands.
- [ ] Check for duplicated materials or IDs.
- [ ] Inspect outlier reflectance values.

```python
import pandas as pd
lib = pd.read_csv("library.csv")
lib.groupby("material").size().pipe(print)
assert (lib.filter(like="band") <= 1).all().all()
```

### Stage 05 – MESMA
- [ ] Confirm endmember sets sum to ≤1.
- [ ] Review residual errors per pixel.
- [ ] Flag negative abundance values.

```python
import pandas as pd
abund = pd.read_csv("mesma_output.csv")
assert (abund.filter(like="EM").sum(axis=1) <= 1.01).all()
assert (abund.filter(like="EM") >= 0).all().all()
```

<!-- FILLME:END -->
