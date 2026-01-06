# Extending

> **When do I need this?** When adding a new target sensor or swapping readers/writers; follow the extension points listed here.

## Purpose
Guide contributions that add sensors to Stage 5 or new exporters feeding [Outputs](../pipeline/outputs.md).

## Inputs
- Bandpass definitions (CSV/JSON) for the new sensor
- Implementation classes under `spectralbridge` to register
- Tests covering the new workflow

## Outputs
Updated convolution products and schemas consumed by [Parquet export & merge](../pipeline/stages.md#parquet-extraction-merging).

## Run it
```bash
pytest tests/convolution/test_new_sensor.py
```

```python
from spectralbridge.convolution import registry

print(registry.available_sensors())
```

## Pitfalls
- Forgetting to update schemas will break Stage 6 merges.
- Ship lightbox-friendly QA thumbnails when adding new visualization layers.
- Document new sensors in [Pipeline Stages](../pipeline/stages.md#sensor-harmonization) and [Troubleshooting](../troubleshooting.md).
