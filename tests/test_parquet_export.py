from __future__ import annotations

from pathlib import Path
import sys
import types

import pandas as pd

if "h5py" not in sys.modules:  # pragma: no cover - dependency shim for unit tests
    fake_h5py = types.ModuleType("h5py")

    class _FakeFile:
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("h5py is not installed; parquet export tests should stub IO")

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

from cross_sensor_cal.pipelines.pipeline import _export_parquet_stage
from cross_sensor_cal.parquet_export import ensure_parquet_for_envi


class DummyLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg: str, *args) -> None:
        self.infos.append(msg % args if args else msg)

    def warning(self, msg: str, *args) -> None:
        self.warnings.append(msg % args if args else msg)


def test_export_parquet_stage_creates_sidecars_for_all_envi(tmp_path: Path, monkeypatch) -> None:
    flight_stem = "NEON_D13_SITE_20230815_directional_reflectance"
    work_dir = tmp_path / flight_stem
    work_dir.mkdir(parents=True, exist_ok=True)

    corrected_img = work_dir / "NEON_D13_SITE_20230815_brdfandtopo_corrected_envi.img"
    corrected_hdr = work_dir / "NEON_D13_SITE_20230815_brdfandtopo_corrected_envi.hdr"
    corrected_img.write_bytes(b"xx")
    corrected_hdr.write_bytes(b"hdr")

    oli_img = work_dir / "NEON_D13_SITE_20230815_directional_reflectance_landsat_oli_envi.img"
    oli_hdr = work_dir / "NEON_D13_SITE_20230815_directional_reflectance_landsat_oli_envi.hdr"
    oli_img.write_bytes(b"yy")
    oli_hdr.write_bytes(b"hdr")

    mask_img = work_dir / "NEON_D13_SITE_20230815_directional_reflectance_cloud_mask_envi.img"
    mask_hdr = work_dir / "NEON_D13_SITE_20230815_directional_reflectance_cloud_mask_envi.hdr"
    mask_img.write_bytes(b"zz")
    mask_hdr.write_bytes(b"hdr")

    calls: list[str] = []

    def fake_ensure(img_path: Path, logger) -> None:
        parquet_path = img_path.with_suffix(".parquet")
        parquet_path.write_bytes(b"pq")
        calls.append(img_path.name)

    import cross_sensor_cal.parquet_export as px

    monkeypatch.setattr(px, "ensure_parquet_for_envi", fake_ensure)

    logger = DummyLogger()
    _export_parquet_stage(
        base_folder=tmp_path,
        product_code="DP1.30006.001",
        flight_stem=flight_stem,
        logger=logger,
    )

    assert (work_dir / "NEON_D13_SITE_20230815_brdfandtopo_corrected_envi.parquet").exists()
    assert (
        work_dir
        / "NEON_D13_SITE_20230815_directional_reflectance_landsat_oli_envi.parquet"
    ).exists()
    assert not (
        work_dir
        / "NEON_D13_SITE_20230815_directional_reflectance_cloud_mask_envi.parquet"
    ).exists()

    assert corrected_img.name in calls
    assert oli_img.name in calls
    assert mask_img.name not in calls


def test_ensure_parquet_for_envi_creates_and_skips_valid(tmp_path: Path, monkeypatch) -> None:
    class DummyLogger2(DummyLogger):
        pass

    img = tmp_path / "f1_envi.img"
    hdr = tmp_path / "f1_envi.hdr"
    img.write_bytes(b"xx")
    hdr.write_bytes(b"hdr")

    import cross_sensor_cal.parquet_export as px

    build_calls = {"count": 0}

    def fake_build(envi_img: Path, envi_hdr: Path, parquet_path: Path, chunk_size: int = 2048) -> None:
        build_calls["count"] += 1
        df = pd.DataFrame(
            {
                "pixel_id": [1],
                "row": [0],
                "col": [0],
                "lon": [0.0],
                "lat": [0.0],
            }
        )
        df.to_parquet(parquet_path)

    monkeypatch.setattr(px, "build_parquet_from_envi", fake_build)

    logger1 = DummyLogger2()
    parquet_path = ensure_parquet_for_envi(img, logger1)
    assert parquet_path is not None and parquet_path.exists()
    assert build_calls["count"] == 1
    assert any("Wrote Parquet" in msg for msg in logger1.infos)

    def fail_build(*_args, **_kwargs) -> None:
        raise AssertionError("should not run because parquet exists already")

    monkeypatch.setattr(px, "build_parquet_from_envi", fail_build)

    logger2 = DummyLogger2()
    result_path = ensure_parquet_for_envi(img, logger2)
    assert result_path == parquet_path
    assert any("Parquet already present" in msg for msg in logger2.infos)


def test_ensure_parquet_for_envi_regenerates_invalid(tmp_path: Path, monkeypatch) -> None:
    class DummyLogger2(DummyLogger):
        pass

    img = tmp_path / "f_invalid_envi.img"
    hdr = tmp_path / "f_invalid_envi.hdr"
    parquet_path = tmp_path / "f_invalid_envi.parquet"
    img.write_bytes(b"xx")
    hdr.write_bytes(b"hdr")
    parquet_path.write_text("not a parquet", encoding="utf-8")

    import cross_sensor_cal.parquet_export as px

    calls = {"count": 0}

    def fake_build(envi_img: Path, envi_hdr: Path, out_path: Path, chunk_size: int = 2048) -> None:
        calls["count"] += 1
        df = pd.DataFrame(
            {
                "pixel_id": [1],
                "row": [0],
                "col": [0],
                "lon": [0.0],
                "lat": [0.0],
            }
        )
        df.to_parquet(out_path)

    monkeypatch.setattr(px, "build_parquet_from_envi", fake_build)

    logger = DummyLogger2()
    result_path = ensure_parquet_for_envi(img, logger)
    assert result_path == parquet_path
    assert calls["count"] == 1
    assert any("invalid" in msg for msg in logger.warnings)
