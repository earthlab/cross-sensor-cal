"""Optional perceptual hash smoke test for QA PNG layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

imagehash = pytest.importorskip("imagehash")
PIL = pytest.importorskip("PIL.Image")

from cross_sensor_cal.qa_plots import render_flightline_panel

from .test_qa_metrics_smoke import _write_envi_pair


def test_panel_phash(tmp_path: Path) -> None:
    flight_dir = tmp_path / "flight_test"
    flight_dir.mkdir()

    wavelengths = [450, 550, 650, 750, 850]
    raw_cube = np.full((5, 16, 16), 0.4, dtype=np.float32)
    corrected_cube = raw_cube * 0.95

    _write_envi_pair(flight_dir / "flight_test_envi", raw_cube, wavelengths)
    _write_envi_pair(
        flight_dir / "flight_test_brdfandtopo_corrected_envi",
        corrected_cube,
        wavelengths,
    )

    png_path, _ = render_flightline_panel(flight_dir, quick=True)

    phash = imagehash.phash(PIL.open(png_path))
    baseline = imagehash.hex_to_hash("3b09c400c4c0c0c0")
    distance = phash - baseline
    assert distance <= 8
