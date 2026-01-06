from __future__ import annotations

import pytest
from pathlib import Path
from spectralbridge.file_sort_adapter import scan_and_categorize
from tests.utils_builders import make_tiny_envi, fake_neon_name

pytestmark = pytest.mark.lite

def test_scan_and_categorize_with_tiny_envi(tmp_path: Path):
    base = tmp_path / "tree"
    base.mkdir()
    img, hdr = make_tiny_envi(base, fake_neon_name(suffix="_reflectance_envi.img"))
    out = scan_and_categorize(base)
    # Expect ENVI and HDR entries
    assert "ENVI" in out and any(p.name.endswith(".img") for p in out["ENVI"])
    assert "HDR" in out and any(p.name.endswith(".hdr") for p in out["HDR"])
