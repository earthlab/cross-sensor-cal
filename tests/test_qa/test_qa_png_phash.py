from __future__ import annotations

from pathlib import Path

import pytest

try:  # pragma: no cover - optional dependency
    from PIL import Image
    import imagehash
except Exception:  # pragma: no cover - optional dependency
    imagehash = None  # type: ignore[assignment]

from cross_sensor_cal.qa_plots import render_flightline_panel


@pytest.mark.skipif(imagehash is None, reason="imagehash not installed")
def test_panel_phash_matches_baseline(qa_fixture_dir: Path) -> None:
    png_path, _ = render_flightline_panel(qa_fixture_dir, quick=True)
    phash = imagehash.phash(Image.open(png_path))
    expected = imagehash.hex_to_hash("3838c6c7c7e384c6")
    assert phash - expected <= 8
