import subprocess
import sys
from pathlib import Path

import numpy as np

from examples.basic_calibration_workflow import demo_calibrate


def test_demo_calibrate_aligns_statistics():
    reference = np.array([0.0, 1.0, -1.0])
    target = np.array([10.0, 12.0, 8.0])
    calibrated = demo_calibrate(reference, target)

    assert np.isclose(calibrated.mean(), reference.mean())
    assert np.isclose(calibrated.std(), reference.std())


def test_example_script_runs(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "basic_calibration_workflow.py"
    )
    result = subprocess.run(
        [sys.executable, str(script)], check=True, capture_output=True, text=True
    )
    assert "calibrated mean/std" in result.stdout
