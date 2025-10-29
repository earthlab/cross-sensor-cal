"""Tests for QA plot generation helpers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

from cross_sensor_cal.qa_plots import summarize_flightline_outputs


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


def test_summarize_flightline_outputs_overwrites_png(tmp_path) -> None:
    base_folder = tmp_path
    flight_stem = "flight_test"
    work_dir = base_folder / flight_stem
    work_dir.mkdir()

    raw_data = np.full((5, 8, 8), 5000, dtype=np.float32)
    corr_data = raw_data * 0.8
    wavelengths = [400, 500, 600, 700, 800]

    _write_envi_pair(work_dir / f"{flight_stem}_envi", raw_data, wavelengths)
    _write_envi_pair(
        work_dir / f"{flight_stem}_brdfandtopo_corrected_envi",
        corr_data,
        wavelengths,
    )

    out_png = work_dir / f"{flight_stem}_qa.png"

    fig1 = summarize_flightline_outputs(base_folder, flight_stem, out_png=out_png)
    plt.close(fig1)
    mtime_1 = out_png.stat().st_mtime_ns

    time.sleep(0.01)

    fig2 = summarize_flightline_outputs(base_folder, flight_stem, out_png=out_png)
    plt.close(fig2)
    mtime_2 = out_png.stat().st_mtime_ns

    assert mtime_2 > mtime_1
