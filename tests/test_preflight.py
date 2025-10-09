from pathlib import Path

import pytest

from src.validations.preflight import PreflightError, validate_inputs


def test_validate_inputs_missing(tmp_path: Path):
    with pytest.raises(PreflightError):
        validate_inputs(tmp_path / "missing.h5", {}, required_keys=["dem"])
