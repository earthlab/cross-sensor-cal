from __future__ import annotations

import inspect
import logging
from pathlib import Path
import sys
import types

import pytest

if "h5py" not in sys.modules:  # pragma: no cover - dependency shim for unit tests
    fake_h5py = types.ModuleType("h5py")

    class _FakeFile:
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("h5py is not installed; parquet stage tests should stub IO")

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

from cross_sensor_cal.pipelines import pipeline as pipeline_module

ENSURE_PARQUET = getattr(pipeline_module, "ensure_parquet_for_envi", None)
if ENSURE_PARQUET is None:
    ENSURE_PARQUET = getattr(pipeline_module, "_export_parquet_stage", None)

pytestmark = pytest.mark.skipif(ENSURE_PARQUET is None, reason="parquet stage not implemented yet")


def _write_nonempty(path: Path, data: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _invoke_helper(img: Path, hdr: Path, logger: logging.Logger) -> None:
    signature = inspect.signature(ENSURE_PARQUET)
    args = []
    kwargs = {}
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if name in {"img_path", "envi_img_path", "img", "envi_img"}:
                args.append(img)
            elif name in {"hdr_path", "envi_hdr_path", "hdr", "envi_hdr"}:
                args.append(hdr)
            elif name in {"directory", "base_path", "base_folder", "folder"}:
                args.append(img.parent)
            elif name == "logger":
                args.append(logger)
            elif name in {"overwrite", "force"}:
                args.append(False)
            elif param.default is inspect._empty:
                kwargs[name] = None
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            if name in {"img_path", "envi_img_path", "img", "envi_img"}:
                kwargs[name] = img
            elif name in {"hdr_path", "envi_hdr_path", "hdr", "envi_hdr"}:
                kwargs[name] = hdr
            elif name == "logger":
                kwargs[name] = logger
            elif name in {"overwrite", "force"}:
                kwargs[name] = False
            elif param.default is inspect._empty:
                kwargs[name] = None
    ENSURE_PARQUET(*args, **kwargs)


def test_parquet_export_creates_parquet_if_missing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    img = _write_nonempty(tmp_path / "scene_brdfandtopo_corrected_envi.img")
    hdr = _write_nonempty(tmp_path / "scene_brdfandtopo_corrected_envi.hdr", b"hdr")
    parquet = img.with_suffix(".parquet")
    if parquet.exists():
        parquet.unlink()

    _invoke_helper(img, hdr, logging.getLogger("test.parquet"))

    assert parquet.exists()
    assert parquet.stat().st_size > 0
    assert "✅" in caplog.text or "parquet" in caplog.text.lower()


def test_parquet_export_skips_if_parquet_exists(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    img = _write_nonempty(tmp_path / "scene_brdfandtopo_corrected_envi.img")
    hdr = _write_nonempty(tmp_path / "scene_brdfandtopo_corrected_envi.hdr", b"hdr")
    parquet = _write_nonempty(img.with_suffix(".parquet"))

    before = (parquet.stat().st_size, parquet.stat().st_mtime_ns)

    _invoke_helper(img, hdr, logging.getLogger("test.parquet"))

    after = (parquet.stat().st_size, parquet.stat().st_mtime_ns)
    assert after == before
    text = caplog.text.lower()
    assert "skip" in text or "already" in text or "⏭️" in caplog.text
