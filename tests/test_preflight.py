from __future__ import annotations

from pathlib import Path

import pytest

from cross_sensor_cal.validations.preflight import PreflightError, validate_inputs


def test_validate_inputs_missing(tmp_path: Path) -> None:
    missing_input = tmp_path / "nope.h5"
    with pytest.raises(PreflightError):
        validate_inputs(missing_input, {}, required_keys=["dem"])
