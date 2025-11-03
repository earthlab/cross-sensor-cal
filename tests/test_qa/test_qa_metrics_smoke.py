from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cross_sensor_cal.qa_plots import render_flightline_panel


def test_render_panel_writes_png_and_json(qa_fixture_dir: Path) -> None:
    png_path, metrics = render_flightline_panel(qa_fixture_dir, quick=True)
    json_path = png_path.with_suffix(".json")

    assert png_path.exists()
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    assert data["provenance"]["flightline_id"] == qa_fixture_dir.name
    assert data["mask"]["valid_pct"] >= 0
    assert data["negatives_pct"] >= 0
    assert data["overbright_pct"] >= 0
    assert isinstance(metrics["header"]["n_bands"], int)
    assert len(data["correction"]["delta_median"]) == data["header"]["n_bands"]
    assert all(isinstance(idx, int) for idx in data["correction"]["largest_delta_indices"])
    assert "wavelength_source" in data["header"]


def test_metrics_arrays_are_serialisable(qa_fixture_dir: Path) -> None:
    _, metrics = render_flightline_panel(qa_fixture_dir, quick=True)
    correction = metrics["correction"]
    delta = np.array(correction["delta_median"], dtype=float)
    assert np.isfinite(delta).all()
    assert set(metrics.keys()) == {
        "provenance",
        "header",
        "mask",
        "correction",
        "convolution",
        "negatives_pct",
        "overbright_pct",
        "issues",
    }
