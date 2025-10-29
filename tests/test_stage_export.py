import sys
import types
from pathlib import Path

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

    fake_matplotlib.use = _noop
    fake_pyplot.figure = lambda *a, **k: _FakeFigure()
    fake_pyplot.subplots = _subplots
    fake_pyplot.close = _noop
    fake_pyplot.plot = _noop
    fake_pyplot.imshow = _noop
    fake_pyplot.title = _noop
    fake_pyplot.savefig = _noop
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

from cross_sensor_cal.pipelines.pipeline import stage_export_envi_from_h5


def _write_nonempty(path: Path, data: bytes = b"xx") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_stage_export_envi_targets_raw_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "workspace"
    base.mkdir()

    flight_stem = "NEON_D13_TEST_DP1_L001-1_20230101_directional_reflectance"
    h5_path = _write_nonempty(base / f"{flight_stem}.h5")

    created: dict[str, Path] = {}

    def _fake_export(images, output_dir, **_):
        assert images == [str(h5_path)]
        out_dir = Path(output_dir)
        created["output_dir"] = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_nonempty(out_dir / f"{flight_stem}_envi.img")
        _write_nonempty(out_dir / f"{flight_stem}_envi.hdr")

    monkeypatch.setattr(
        "cross_sensor_cal.pipelines.pipeline.neon_to_envi_no_hytools",
        _fake_export,
    )

    raw_img, raw_hdr = stage_export_envi_from_h5(
        base_folder=base,
        product_code="DP1.30006.001",
        flight_stem=flight_stem,
        brightness_offset=None,
        parallel_mode=False,
    )

    assert "output_dir" in created
    assert raw_img == created["output_dir"] / f"{flight_stem}_envi.img"
    assert raw_hdr == created["output_dir"] / f"{flight_stem}_envi.hdr"
    assert raw_img.parent == base / flight_stem
    assert "landsat" not in raw_img.name.lower()
    assert "micasense" not in raw_img.name.lower()


def test_stage_export_raises_when_corrected_without_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "workspace"
    base.mkdir()

    flight_stem = "NEON_D13_TEST_DP1_L001-1_20230101_directional_reflectance"
    _write_nonempty(base / f"{flight_stem}.h5")

    work_dir = base / flight_stem
    work_dir.mkdir()
    _write_nonempty(work_dir / f"{flight_stem}_brdfandtopo_corrected_envi.img")

    def _fail_export(*_args, **_kwargs):  # pragma: no cover - should never run
        raise AssertionError("Raw export should not run when corrected exists without raw")

    monkeypatch.setattr(
        "cross_sensor_cal.pipelines.pipeline.neon_to_envi_no_hytools",
        _fail_export,
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        stage_export_envi_from_h5(
            base_folder=base,
            product_code="DP1.30006.001",
            flight_stem=flight_stem,
            recover_missing_raw=False,
        )

    assert "Raw ENVI missing" in str(excinfo.value)


def test_stage_export_recovers_missing_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "workspace"
    base.mkdir()

    flight_stem = "NEON_D13_TEST_DP1_L001-1_20230101_directional_reflectance"
    h5_path = _write_nonempty(base / f"{flight_stem}.h5")

    work_dir = base / flight_stem
    work_dir.mkdir()
    corrected_img = _write_nonempty(
        work_dir / f"{flight_stem}_brdfandtopo_corrected_envi.img"
    )

    created: dict[str, Path] = {}

    def _fake_export(images, output_dir, brightness_offset=None, **_):
        assert images == [str(h5_path)]
        created["output_dir"] = Path(output_dir)
        created["brightness_offset"] = brightness_offset
        _write_nonempty(created["output_dir"] / f"{flight_stem}_envi.img")
        _write_nonempty(created["output_dir"] / f"{flight_stem}_envi.hdr")

    monkeypatch.setattr(
        "cross_sensor_cal.pipelines.pipeline.neon_to_envi_no_hytools",
        _fake_export,
    )

    raw_img, raw_hdr = stage_export_envi_from_h5(
        base_folder=base,
        product_code="DP1.30006.001",
        flight_stem=flight_stem,
        brightness_offset=0.42,
        parallel_mode=False,
    )

    assert raw_img.exists()
    assert raw_hdr.exists()
    assert raw_img == work_dir / f"{flight_stem}_envi.img"
    assert raw_hdr == work_dir / f"{flight_stem}_envi.hdr"
    assert created["output_dir"] == work_dir
    assert created["brightness_offset"] == 0.42
    assert corrected_img.exists()
