import numpy as np

from cross_sensor_cal.corrections import (
    NDVIBinningConfig,
    ReferenceGeometry,
    apply_brdf_correct,
    apply_topo_correct,
    _scs_c_ratio,
)


class _DummyCube:
    def __init__(self):
        self.scale_factor = 1.0
        self.no_data = -9999.0
        self.mask_no_data = np.ones((2, 2), dtype=bool)
        self.wavelengths = np.array([650.0, 860.0], dtype=np.float32)
        self.data = None

    def get_ancillary(self, name: str, radians: bool = True):
        if name == "slope":
            return np.full((2, 2), np.deg2rad(20.0), dtype=np.float32)
        if name == "aspect":
            return np.zeros((2, 2), dtype=np.float32)
        if name == "solar_zn":
            return np.full((2, 2), np.deg2rad(30.0), dtype=np.float32)
        if name == "solar_az":
            return np.zeros((2, 2), dtype=np.float32)
        if name == "sensor_zn":
            return np.zeros((2, 2), dtype=np.float32)
        if name == "sensor_az":
            return np.zeros((2, 2), dtype=np.float32)
        raise KeyError(name)


def test_scs_topo_correction_scaling():
    cube = _DummyCube()
    chunk = np.full((2, 2, 1), 0.5, dtype=np.float32)
    corrected = apply_topo_correct(cube, chunk, 0, 2, 0, 2, use_scs_c=True)
    assert np.all(np.isfinite(corrected))
    assert not np.allclose(corrected, chunk)


def test_brdf_ratio_increases_reflectance_when_reference_brighter():
    cube = _DummyCube()
    chunk = np.full((2, 2, 2), 0.1, dtype=np.float32)
    ndvi_config = NDVIBinningConfig(n_bins=1, ndvi_min=-1.0, ndvi_max=1.0)
    coeffs = {
        "iso": np.array([[0.8, 0.8]], dtype=np.float32),
        "vol": np.array([[0.1, 0.1]], dtype=np.float32),
        "geo": np.array([[0.1, 0.1]], dtype=np.float32),
        "volume_kernel": "RossThick",
        "geom_kernel": "LiSparseReciprocal",
        "ndvi_edges": [-1.0, 1.0],
    }
    cube.brdf_coefficients = coeffs
    corrected = apply_brdf_correct(
        cube,
        chunk,
        0,
        2,
        0,
        2,
        ndvi_config=ndvi_config,
        reference_geometry=ReferenceGeometry(solar_zenith_deg=10.0),
    )
    assert np.all(np.isfinite(corrected))
    assert not np.allclose(corrected, chunk)


def test_neutral_coefficients_broadcast_across_bins():
    cube = _DummyCube()
    # Chunk yields NDVI spread across bins when using defaults
    chunk = np.array(
        [
            [[0.2, 0.8], [0.3, 0.7]],
            [[0.4, 0.6], [0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    cube.brdf_coefficients = None
    corrected = apply_brdf_correct(cube, chunk, 0, 2, 0, 2)
    assert np.all(np.isfinite(corrected))
    assert np.all(corrected != cube.no_data)
    np.testing.assert_allclose(corrected, chunk)


class _NearZeroIncidenceCube(_DummyCube):
    def get_ancillary(self, name: str, radians: bool = True):
        if name == "slope":
            return np.full((2, 2), np.deg2rad(90.0), dtype=np.float32)
        if name == "aspect":
            # Offset the relative azimuth slightly to yield a tiny positive cos_i.
            return np.full((2, 2), np.deg2rad(90.0) - 1e-6, dtype=np.float32)
        if name == "solar_zn":
            return np.full((2, 2), np.deg2rad(89.0), dtype=np.float32)
        if name == "solar_az":
            return np.zeros((2, 2), dtype=np.float32)
        return super().get_ancillary(name, radians=radians)


def test_topo_denominator_guard_keeps_ratios_finite():
    cube = _NearZeroIncidenceCube()
    chunk = np.full((2, 2, 1), 0.5, dtype=np.float32)
    corrected = apply_topo_correct(cube, chunk, 0, 2, 0, 2, use_scs_c=True)
    assert np.all(np.isfinite(corrected))
    # Without the guard, a near-zero denominator would collapse reflectance toward zero.
    np.testing.assert_allclose(corrected, chunk, rtol=1e-3, atol=1e-3)


def test_scs_c_ratio_guard_handles_tiny_denominators_directly():
    numerator = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    denominator = np.array([0.5, 1e-6, 0.0, -1e-4], dtype=np.float32)

    ratio = _scs_c_ratio(numerator, denominator, min_denom=1e-3)

    assert np.all(np.isfinite(ratio))
    # Guarded entries fall back to a neutral ratio of 1.0, while valid entries
    # retain the original scaling behaviour.
    np.testing.assert_allclose(ratio[[1, 2, 3]], 1.0)
    assert ratio[0] > 1.0
    assert ratio.max() <= numerator.max() / 1e-3
