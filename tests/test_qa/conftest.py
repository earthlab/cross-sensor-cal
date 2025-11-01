from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_envi_pair(base_path: Path, data: np.ndarray, wavelengths: list[float]) -> None:
    data = np.asarray(data, dtype=np.float32)
    img_path = base_path.with_suffix(".img")
    hdr_path = base_path.with_suffix(".hdr")
    data.tofile(img_path)
    header_lines = [
        "ENVI",
        f"samples = {data.shape[2]}",
        f"lines = {data.shape[1]}",
        f"bands = {data.shape[0]}",
        "data type = 4",
        "interleave = bsq",
        "byte order = 0",
        "wavelength units = Nanometers",
        "fwhm = {" + ", ".join("10" for _ in wavelengths) + "}",
        "wavelength = {" + ", ".join(f"{w}" for w in wavelengths) + "}",
    ]
    hdr_path.write_text("\n".join(header_lines), encoding="utf-8")


@pytest.fixture()
def qa_fixture_dir(tmp_path: Path) -> Path:
    stem = "NEON_TEST_FLIGHT"
    flight_dir = tmp_path / stem
    flight_dir.mkdir()
    rng = np.random.default_rng(42)
    base = 0.2 + rng.random((4, 12, 10)).astype(np.float32) * 0.05
    corrected = base * 0.92 + 0.01
    convolved = corrected * 0.9
    wavelengths = [490.0, 560.0, 660.0, 820.0]
    _write_envi_pair(flight_dir / f"{stem}_envi", base, wavelengths)
    _write_envi_pair(
        flight_dir / f"{stem}_brdfandtopo_corrected_envi",
        corrected,
        wavelengths,
    )
    _write_envi_pair(
        flight_dir / f"{stem}_oli_convolved_envi",
        convolved,
        wavelengths,
    )
    return flight_dir
