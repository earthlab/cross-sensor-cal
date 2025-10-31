"""Smoke tests for QA metrics JSON generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cross_sensor_cal.qa_plots import render_flightline_panel


def _write_envi_pair(base_path: Path, data: np.ndarray, wavelengths: list[float]) -> None:
    data = np.asarray(data, dtype=np.float32)
    img_path = base_path.with_suffix(".img")
    hdr_path = base_path.with_suffix(".hdr")
    data.tofile(img_path)
    hdr_lines = [
        "ENVI",
        f"samples = {data.shape[2]}",
        f"lines = {data.shape[1]}",
        f"bands = {data.shape[0]}",
        "data type = 4",
        "interleave = bsq",
        "byte order = 0",
        "wavelength = {" + ", ".join(f"{w}" for w in wavelengths) + "}",
        "",
    ]
    hdr_path.write_text("\n".join(hdr_lines), encoding="utf-8")


def test_render_panel_writes_metrics(tmp_path: Path) -> None:
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

    png_path, metrics = render_flightline_panel(flight_dir, quick=True)

    assert png_path.exists()
    json_path = png_path.with_suffix(".json")
    assert json_path.exists()

    payload = json.loads(json_path.read_text())
    assert metrics.keys() == payload.keys()

    header = payload["header"]
    assert header["n_bands"] > 0
    assert header["n_wavelengths_finite"] == len(wavelengths)
    assert header["wavelength_source"] in {"header", "sensor_default", "absent"}

    mask = payload["mask"]
    assert mask["n_total"] > 0
    assert mask["n_valid"] > 0
    assert mask["valid_pct"] > 0

    assert payload["negatives_pct"] >= 0

    if header["n_wavelengths_finite"]:
        monotonic = header["wavelengths_monotonic"]
        assert monotonic in (True, False, None)
        if monotonic:
            assert header["first_nm"] < header["last_nm"]

