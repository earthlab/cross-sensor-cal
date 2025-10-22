import pytest
np = pytest.importorskip("numpy")

from cross_sensor_cal.convolution_resample import (
    _apply_convolution_with_renorm,
    _build_W_from_gaussians,
    _ensure_nm_and_sort,
    _nearest_or_linear_sample,
)


def test_flat_spectrum_preserves_reflectance():
    wl_nm = np.linspace(400, 2500, 50, dtype=float)
    cube = np.full((3, 4, wl_nm.size), 0.2, dtype=np.float32)
    centers = np.array([450.0, 650.0, 850.0], dtype=float)
    fwhm = np.full(centers.size, 30.0, dtype=float)

    weights = _build_W_from_gaussians(wl_nm, centers, fwhm)
    resampled = _apply_convolution_with_renorm(cube, weights)

    assert resampled.shape == (3, 4, centers.size)
    assert np.allclose(resampled, 0.2, atol=1e-6)


def test_masked_wavelengths_renormalise():
    wl_nm = np.linspace(400, 2400, 60, dtype=float)
    cube = np.full((2, 2, wl_nm.size), 0.5, dtype=np.float32)
    cube[..., 30:] = np.nan  # mask half the wavelengths

    centers = np.array([500.0, 700.0], dtype=float)
    fwhm = np.full(centers.size, 40.0, dtype=float)

    weights = _build_W_from_gaussians(wl_nm, centers, fwhm)
    resampled = _apply_convolution_with_renorm(cube, weights)

    assert np.nanmax(np.abs(resampled - 0.5)) <= 0.005  # within 1%


def test_unit_conversion_and_sorting_aligns_cube():
    wavelengths = np.array([1.0, 0.4, 0.7], dtype=float)
    cube = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)

    wl_nm, cube_sorted, order = _ensure_nm_and_sort(wavelengths, cube)

    assert np.allclose(wl_nm, [400.0, 700.0, 1000.0])
    assert cube_sorted.shape[-1] == wavelengths.size
    assert np.allclose(cube_sorted[0, 0], [0.2, 0.3, 0.1])
    assert np.array_equal(order, np.array([1, 2, 0]))


def test_linear_sampling_matches_interpolation():
    wl_nm = np.array([400.0, 500.0, 600.0], dtype=float)
    cube = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)

    result = _nearest_or_linear_sample(cube, wl_nm, [550.0], mode="linear")
    assert result.shape == (1, 1, 1)
    assert np.isclose(result[0, 0, 0], 0.5)

    nearest = _nearest_or_linear_sample(cube, wl_nm, [540.0], mode="nearest")
    assert np.isclose(nearest[0, 0, 0], 0.4)
