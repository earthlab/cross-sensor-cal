import json
from pathlib import Path

import numpy as np
import pytest

from cross_sensor_cal.corrections import apply_brdf_correct, fit_and_save_brdf_model


class _FakeCube:
    def __init__(self, data: np.ndarray, scale_factor: float = 1.0) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.scale_factor = float(scale_factor)
        self.lines, self.columns, self.bands = self.data.shape
        self.mask_no_data = np.ones((self.lines, self.columns), dtype=bool)
        self.no_data = -9999.0
        self.base_key = "fake"

    def get_ancillary(self, name: str, radians: bool = True) -> np.ndarray:
        shape = (self.lines, self.columns)
        if name in {"solar_zn", "sensor_zn"}:
            return np.full(shape, 0.1, dtype=np.float32)
        if name in {"solar_az", "sensor_az", "slope", "aspect"}:
            return np.full(shape, 0.0, dtype=np.float32)
        raise KeyError(name)


def _neutral_coefficients(path: Path, bands: int) -> Path:
    payload = {
        "iso": [1.0 for _ in range(bands)],
        "vol": [0.0 for _ in range(bands)],
        "geo": [0.0 for _ in range(bands)],
        "volume_kernel": "RossThick",
        "geom_kernel": "LiSparseReciprocal",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_brdf_fit_scale_invariant(tmp_path: Path) -> None:
    unitless = np.full((5, 5, 3), 0.3, dtype=np.float32)
    scaled = unitless / 1e-4

    cube_unitless = _FakeCube(unitless, scale_factor=1.0)
    cube_scaled = _FakeCube(scaled, scale_factor=1e-4)

    coeff_unitless = fit_and_save_brdf_model(cube_unitless, tmp_path / "unitless")
    coeff_scaled = fit_and_save_brdf_model(cube_scaled, tmp_path / "scaled")

    model_unitless = json.loads(coeff_unitless.read_text())
    model_scaled = json.loads(coeff_scaled.read_text())

    for key in ("iso", "vol", "geo"):
        assert np.allclose(model_unitless[key], model_scaled[key], atol=1e-3)


def test_correction_respects_raw_scale(tmp_path: Path) -> None:
    unitless = np.full((4, 4, 2), 0.25, dtype=np.float32)
    scaled = unitless / 1e-4
    cube = _FakeCube(scaled, scale_factor=1e-4)

    coeff_path = _neutral_coefficients(tmp_path / "coeff.json", cube.bands)
    corrected = apply_brdf_correct(
        cube,
        cube.data,
        0,
        cube.lines,
        0,
        cube.columns,
        coeff_path=coeff_path,
    )

    assert np.allclose(corrected, cube.data, atol=1e-3)


def test_correction_preserves_shape_and_dtype(tmp_path: Path) -> None:
    scaled = np.full((2, 3, 4), 0.45 / 1e-4, dtype=np.float32)
    cube = _FakeCube(scaled, scale_factor=1e-4)

    coeff_path = _neutral_coefficients(tmp_path / "coeff_dtype.json", cube.bands)
    corrected = apply_brdf_correct(
        cube,
        cube.data,
        0,
        cube.lines,
        0,
        cube.columns,
        coeff_path=coeff_path,
    )

    assert corrected.shape == cube.data.shape
    assert corrected.dtype == np.float32


def test_outliers_masked_from_fit(tmp_path: Path) -> None:
    unitless = np.full((3, 3, 1), 0.2, dtype=np.float32)
    unitless[0, 0, 0] = 1.5  # beyond valid range and should be excluded
    scaled = unitless / 1e-4
    cube = _FakeCube(scaled, scale_factor=1e-4)

    coeff_path = fit_and_save_brdf_model(cube, tmp_path / "outlier")
    model = json.loads(coeff_path.read_text())

    valid_mean = float(np.mean(unitless[unitless < 1.0]))
    assert model["iso"][0] == pytest.approx(valid_mean, rel=0.6)
    assert model["iso"][0] < 0.5
    assert abs(model["vol"][0]) < 0.2
    assert abs(model["geo"][0]) < 0.2
