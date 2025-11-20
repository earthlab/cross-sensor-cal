from __future__ import annotations

import json
import logging
import io
from pathlib import Path
from typing import Iterable
import sys
import types

import numpy as np

import pytest

if "h5py" not in sys.modules:  # pragma: no cover - dependency shim for unit tests
    fake_h5py = types.ModuleType("h5py")

    class _FakeFile:
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("h5py is not installed; real HDF5 IO should be skipped in tests")

        def __enter__(self) -> "_FakeFile":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    fake_h5py.File = _FakeFile
    fake_h5py.Group = type("Group", (), {})
    sys.modules["h5py"] = fake_h5py

if "matplotlib" not in sys.modules:  # pragma: no cover - dependency shim for unit tests
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")

    def _noop(*_: object, **__: object) -> None:
        return None

    class _FakeFigure:
        def __getattr__(self, _name: str) -> object:
            return _noop

    class _FakeAxes:
        def __getattr__(self, _name: str) -> object:
            return _noop

    def _subplots(*_: object, **__: object) -> tuple[_FakeFigure, _FakeAxes]:
        return _FakeFigure(), _FakeAxes()

    fake_pyplot.figure = lambda *a, **k: _FakeFigure()
    fake_pyplot.subplots = _subplots
    fake_pyplot.close = _noop
    fake_pyplot.plot = _noop
    fake_pyplot.imshow = _noop
    fake_pyplot.title = _noop
    fake_pyplot.savefig = _noop
    fake_matplotlib.use = _noop
    fake_matplotlib_figure = types.ModuleType("matplotlib.figure")
    fake_matplotlib_figure.Figure = _FakeFigure
    sys.modules["matplotlib.figure"] = fake_matplotlib_figure
    sys.modules["matplotlib"] = fake_matplotlib
    sys.modules["matplotlib.pyplot"] = fake_pyplot

if "shapely" not in sys.modules:  # pragma: no cover - dependency shim for unit tests
    fake_shapely = types.ModuleType("shapely")
    fake_geometry = types.ModuleType("shapely.geometry")

    def _fake_box(*_: object, **__: object) -> None:
        return None

    fake_geometry.box = _fake_box
    sys.modules["shapely"] = fake_shapely
    sys.modules["shapely.geometry"] = fake_geometry

from cross_sensor_cal.pipelines.pipeline import go_forth_and_multiply
from cross_sensor_cal.pipelines import pipeline


def _write_nonempty(path: Path, data: bytes = b"xx") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _snapshot_stats(paths: Iterable[Path]) -> dict[Path, tuple[int, int]]:
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def test_pipeline_idempotence_skip_behavior(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "workspace"
    base.mkdir()

    flight_stem = "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance"

    h5_path = _write_nonempty(base / f"{flight_stem}.h5")

    work_dir = base / flight_stem
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_img = _write_nonempty(work_dir / f"{flight_stem}_envi.img")
    raw_hdr = _write_nonempty(work_dir / f"{flight_stem}_envi.hdr")
    corrected_img = _write_nonempty(
        work_dir / f"{flight_stem}_brdfandtopo_corrected_envi.img"
    )
    corrected_hdr = _write_nonempty(
        work_dir / f"{flight_stem}_brdfandtopo_corrected_envi.hdr"
    )
    correction_json = work_dir / f"{flight_stem}_brdfandtopo_corrected_envi.json"
    correction_json.write_text(json.dumps({"ok": True}))

    sensors = [
        "landsat_tm",
        "landsat_etm+",
        "landsat_oli",
        "landsat_oli2",
        "micasense",
        "micasense_to_match_tm_etm+",
        "micasense_to_match_oli_oli2",
    ]
    sensor_pairs = []
    for sensor in sensors:
        sensor_img = _write_nonempty(work_dir / f"{flight_stem}_{sensor}_envi.img")
        sensor_hdr = _write_nonempty(work_dir / f"{flight_stem}_{sensor}_envi.hdr")
        sensor_pairs.extend([sensor_img, sensor_hdr])

    preexisting_paths = [
        h5_path,
        raw_img,
        raw_hdr,
        corrected_img,
        corrected_hdr,
        correction_json,
        *sensor_pairs,
    ]

    before_stats = _snapshot_stats(preexisting_paths)

    def _fail(*_: object, **__: object) -> None:  # pragma: no cover - guard against regressions
        raise AssertionError("heavy stage should not be invoked when outputs exist")

    monkeypatch.setattr("cross_sensor_cal.pipelines.pipeline.neon_to_envi_no_hytools", _fail)
    monkeypatch.setattr("cross_sensor_cal.pipelines.pipeline.build_correction_parameters_dict", _fail)
    monkeypatch.setattr("cross_sensor_cal.pipelines.pipeline.apply_brdf_topo_core", _fail)
    monkeypatch.setattr("cross_sensor_cal.pipelines.pipeline.resample_to_sensor_bands", _fail)
    monkeypatch.setattr("cross_sensor_cal.pipelines.pipeline.write_resampled_envi_cube", _fail)

    caplog.set_level(logging.INFO, logger="cross_sensor_cal.pipelines.pipeline")

    log_capture = io.StringIO()
    logger = logging.getLogger("cross_sensor_cal.pipelines.pipeline")
    handler = logging.StreamHandler(log_capture)
    logger.addHandler(handler)
    try:
        go_forth_and_multiply(
            base_folder=base,
            site_code="NIWO",
            year_month="202308",
            flight_lines=[flight_stem],
            resample_method="convolution",
        )
    finally:
        logger.removeHandler(handler)

    captured = capsys.readouterr()
    text = caplog.text + captured.out + captured.err + log_capture.getvalue()
    assert "skipping heavy export" in text
    assert "Correction JSON already complete" in text
    assert "BRDF+topo correction already complete" in text
    for sensor in sensors:
        assert f"{sensor} product already complete" in text

    after_stats = _snapshot_stats(preexisting_paths)
    assert before_stats == after_stats


def test_convolution_writes_undarkened_envi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = lines = 2
    bands = 2

    corrected_hdr = tmp_path / "flight_brdfandtopo_corrected_envi.hdr"
    header = {
        "samples": samples,
        "lines": lines,
        "bands": bands,
        "interleave": "bsq",
        "data type": 4,
        "byte order": 0,
        "wavelength": [1.0, 2.0],
    }
    corrected_hdr.write_text(pipeline._build_resample_header_text(header))

    corrected_img = corrected_hdr.with_suffix(".img")
    data = np.arange(bands * lines * samples, dtype=np.float32).reshape(bands, lines, samples)
    mm = np.memmap(corrected_img, dtype="float32", mode="w+", shape=data.shape)
    mm[:] = data
    mm.flush()
    del mm

    def _fake_resample(tile_yxb: np.ndarray, *_: object, **__: object) -> np.ndarray:
        band_count = 3
        out = np.empty((*tile_yxb.shape[:2], band_count), dtype=np.float32)
        for idx in range(band_count):
            out[:, :, idx] = tile_yxb[:, :, 0] + idx
        return out

    monkeypatch.setattr(pipeline, "resample_chunk_to_sensor", _fake_resample)

    sensor_srf = {str(idx * 100.0): np.ones(bands, dtype=np.float32) for idx in range(1, 4)}
    out_stem = tmp_path / "flight_landsat_tm_envi"

    brightness_map = pipeline.convolve_resample_product(
        corrected_hdr_path=corrected_hdr,
        sensor_srf=sensor_srf,
        out_stem_resampled=out_stem,
        tile_y=lines,
        tile_x=samples,
        interactive_mode=False,
        sensor_name="landsat_tm",
    )

    undarkened_img = tmp_path / "flight_landsat_tm_undarkened_envi.img"
    undarkened_hdr = tmp_path / "flight_landsat_tm_undarkened_envi.hdr"
    darkened_img = out_stem.with_suffix(".img")
    darkened_hdr = out_stem.with_suffix(".hdr")

    assert undarkened_img.exists()
    assert undarkened_hdr.exists()
    assert darkened_img.exists()
    assert darkened_hdr.exists()

    undarkened_cube = np.memmap(undarkened_img, dtype="float32", mode="r", shape=(3, lines, samples))
    darkened_cube = np.memmap(darkened_img, dtype="float32", mode="r", shape=(3, lines, samples))
    assert not np.array_equal(undarkened_cube, darkened_cube)
    del undarkened_cube
    del darkened_cube

    parsed_header = pipeline._parse_envi_header(darkened_hdr)
    assert "brightness_coefficients" in parsed_header
    assert brightness_map
