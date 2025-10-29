from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("rasterio")

import pyarrow.parquet as pq

from cross_sensor_cal.file_types import (
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceResampledENVIFile,
)
from cross_sensor_cal.parquet_export import ensure_parquet_for_envi, parquet_exists_and_valid
from cross_sensor_cal.pipelines.pipeline import _export_parquet_stage


def _write_envi_pair(img_path: Path, data: np.ndarray, wavelengths: list[float]) -> Path:
    img_path = Path(img_path)
    hdr_path = img_path.with_suffix(".hdr")

    data.astype(np.float32).tofile(img_path)

    bands, lines, samples = data.shape
    header_lines = [
        "ENVI",
        f"samples = {samples}",
        f"lines = {lines}",
        f"bands = {bands}",
        "interleave = bsq",
        "data type = 4",
        "byte order = 0",
        "wavelength units = Nanometers",
        "wavelength = {" + ", ".join(f"{w:0.1f}" for w in wavelengths) + "}",
        "map info = {UTM, 1, 1, 500000, 4420000, 1, 1}",
    ]

    hdr_path.write_text("\n".join(header_lines), encoding="utf-8")
    return hdr_path


def test_ensure_parquet_writes_file(tmp_path: Path):
    data = np.arange(12, dtype=np.float32).reshape(3, 2, 2) / 100.0
    wavelengths = [400.0, 500.0, 600.0]
    img_path = tmp_path / "NEON_D01_TEST_DP1.30006.001_20200101_brdfandtopo_corrected_envi.img"
    hdr_path = _write_envi_pair(img_path, data, wavelengths)

    logger = logging.getLogger("parquet-test")
    logger.setLevel(logging.INFO)

    ensure_parquet_for_envi(img_path, hdr_path, logger, chunk_size=4)

    parquet_path = img_path.with_suffix(".parquet")
    assert parquet_exists_and_valid(parquet_path)

    table = pq.read_table(parquet_path)
    assert set(table.column_names) == {"x", "y", "band", "wavelength_nm", "reflectance"}
    assert table.num_rows == data.size

    expected_reflectance = data.reshape(3, -1).T.reshape(-1)
    np.testing.assert_allclose(table.column("reflectance").to_numpy(), expected_reflectance)


def test_ensure_parquet_skips_if_present(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    data = np.ones((2, 2, 2), dtype=np.float32)
    wavelengths = [400.0, 450.0]
    img_path = tmp_path / "NEON_D01_TEST_DP1.30006.001_20200101_brdfandtopo_corrected_envi.img"
    hdr_path = _write_envi_pair(img_path, data, wavelengths)

    logger = logging.getLogger("parquet-test-skip")
    logger.setLevel(logging.INFO)

    ensure_parquet_for_envi(img_path, hdr_path, logger, chunk_size=2)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        ensure_parquet_for_envi(img_path, hdr_path, logger, chunk_size=2)

    assert any("Skipped Parquet export" in message for message in caplog.messages)


def test_export_parquet_stage_processes_corrected_and_resampled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corrected = NEONReflectanceBRDFCorrectedENVIFile.from_components(
        domain="D01",
        site="TEST",
        date="20200101",
        suffix="envi",
        folder=tmp_path,
    )
    corrected.path.touch()
    corrected.hdr_path().touch()

    resampled = NEONReflectanceResampledENVIFile.from_components(
        domain="D01",
        site="TEST",
        date="20200101",
        sensor="Landsat_8_OLI",
        suffix="envi",
        folder=tmp_path,
    )
    resampled.path.touch()
    resampled.hdr_path().touch()

    flight_stem = corrected.path.stem.replace("_brdfandtopo_corrected_envi", "")

    calls: list[tuple[Path, Path]] = []

    def _fake_ensure(img: Path, hdr: Path, _logger, *, chunk_size: int = 2048):
        calls.append((Path(img), Path(hdr)))
        return Path(img).with_suffix(".parquet")

    monkeypatch.setattr(
        "cross_sensor_cal.pipelines.pipeline.ensure_parquet_for_envi",
        _fake_ensure,
    )

    _export_parquet_stage(base_folder=tmp_path, flight_stem=flight_stem)

    assert corrected.path in {img for img, _ in calls}
    assert resampled.path in {img for img, _ in calls}
