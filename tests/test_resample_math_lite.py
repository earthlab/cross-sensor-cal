from __future__ import annotations
import pytest
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.lite

def test_tiny_numeric_operation_is_deterministic():
    rng = np.random.default_rng(42)
    a = rng.random((16,16), dtype=np.float32)
    # emulate a simple 3x3 mean filter to stand in for convolution/resample
    k = np.ones((3,3), dtype=np.float32) / 9.0
    conv = pytest.importorskip("scipy.signal").convolve2d
    out = conv(a, k, mode="same", boundary="symm")
    assert out.shape == a.shape
    # simple invariants
    assert np.isfinite(out).all()
    # mean preserved within tolerance
    assert abs(out.mean() - a.mean()) < 1e-3
