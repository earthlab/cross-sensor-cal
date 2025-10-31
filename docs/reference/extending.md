# Extending

> **When do I need this?** When adding a new target sensor or swapping readers/writers; follow the extension points listed here.

## Purpose
Guide contributions that add sensors to Stage 4 or new exporters feeding [Outputs](../pipeline/outputs.md).

## Inputs
- Bandpass definitions (CSV/JSON) for the new sensor
- Implementation classes under `cross_sensor_cal` to register
- Tests covering the new workflow

## Outputs
Updated convolution products and schemas consumed by [Parquet export](../pipeline/stages.md#5-parquet-export) and [Merge](../pipeline/stages.md#6-duckdb-merge).

## Run it
```bash
pytest tests/convolution/test_new_sensor.py
```

```python
from cross_sensor_cal.convolution import registry

print(registry.available_sensors())
```

## Pitfalls
- Forgetting to update schemas will break Stage 6 merges.
- Ship lightbox-friendly QA thumbnails when adding new visualization layers.
- Document new sensors in [Pipeline Stages](../pipeline/stages.md#4-cross-sensor-convolution) and [Troubleshooting](../troubleshooting.md).
