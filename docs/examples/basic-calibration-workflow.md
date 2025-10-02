# Synthetic calibration workflow

This example mirrors the functionality demonstrated in
[`examples/basic_calibration_workflow.py`](../../examples/basic_calibration_workflow.py).
It uses small NumPy arrays to showcase the calibration utilities without
requiring large NEON data downloads.

```bash
python examples/basic_calibration_workflow.py
```

The script:

1. Creates two synthetic hyperspectral cubes with different mean and variance.
2. Applies a simple calibration routine that aligns statistics between the
   cubes.
3. Prints summary statistics so you can verify the adjustment.

In real workflows you would replace the synthetic arrays with data produced by
`cross_sensor_cal.neon_to_envi` and follow the stage-by-stage instructions in the
main documentation.
