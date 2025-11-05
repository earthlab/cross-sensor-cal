from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pyarrow as pa
import pyarrow.parquet as pq

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
    fake_backends = types.ModuleType("matplotlib.backends")
    fake_backend_pdf = types.ModuleType("matplotlib.backends.backend_pdf")
    fake_axes = types.ModuleType("matplotlib.axes")
    fake_figure = types.ModuleType("matplotlib.figure")

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
    sys.modules["matplotlib.backends"] = fake_backends
    sys.modules["matplotlib.backends.backend_pdf"] = fake_backend_pdf
    sys.modules["matplotlib.axes"] = fake_axes
    sys.modules["matplotlib.figure"] = fake_figure
    fake_matplotlib.use = _noop

    class _FakePdfPages:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def savefig(self, *_: object, **__: object) -> None:
            return None

    fake_backend_pdf.PdfPages = _FakePdfPages

    class _FakeAxesClass:
        pass

    fake_axes.Axes = _FakeAxesClass

    class _FakeFigureClass:
        pass

    fake_figure.Figure = _FakeFigureClass

if "shapely" not in sys.modules:  # pragma: no cover - dependency shim for unit tests
    fake_shapely = types.ModuleType("shapely")
    fake_geometry = types.ModuleType("shapely.geometry")

    def _fake_box(*_: object, **__: object) -> None:
        return None

    fake_geometry.box = _fake_box
    sys.modules["shapely"] = fake_shapely
    sys.modules["shapely.geometry"] = fake_geometry

if "numpy" not in sys.modules:  # pragma: no cover - dependency shim for unit tests
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = type("ndarray", (), {})
    fake_numpy.nan = float("nan")

    def _array(values, *_, **__):
        return values

    fake_numpy.array = _array
    sys.modules["numpy"] = fake_numpy

if "duckdb" not in sys.modules:  # pragma: no cover - dependency shim for unit tests
    fake_duckdb = types.ModuleType("duckdb")

    class _FakeConnection:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def execute(self, *_: object, **__: object):
            return self

        def fetchall(self):
            return []

        def close(self):
            return None

    def connect(*_: object, **__: object) -> _FakeConnection:
        return _FakeConnection()

    fake_duckdb.connect = connect
    sys.modules["duckdb"] = fake_duckdb

if "pandas" not in sys.modules:  # pragma: no cover - dependency shim for unit tests
    fake_pandas = types.ModuleType("pandas")

    class _FakeDataFrame(dict):
        def to_parquet(self, path):
            from pathlib import Path

            Path(path).write_text("{}", encoding="utf-8")

    def DataFrame(*args, **kwargs):
        return _FakeDataFrame()

    fake_pandas.DataFrame = DataFrame
    sys.modules["pandas"] = fake_pandas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _stub_module(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if name not in sys.modules:
        sys.modules[name] = module


_stub_module("cross_sensor_cal")
sys.modules["cross_sensor_cal"].__path__ = [str(SRC_ROOT / "cross_sensor_cal")]  # type: ignore[attr-defined]
_stub_module("cross_sensor_cal.pipelines")
sys.modules["cross_sensor_cal.pipelines"].__path__ = []  # type: ignore[attr-defined]
_stub_module("cross_sensor_cal.utils", get_package_data_path=lambda *a, **k: Path("data"))
sys.modules["cross_sensor_cal.utils"].__path__ = []  # type: ignore[attr-defined]
_stub_module("cross_sensor_cal.utils.naming", get_flight_paths=lambda *a, **k: {}, get_flightline_products=lambda base, code, stem: {"work_dir": Path(base) / stem})
_stub_module("cross_sensor_cal.brdf_topo", apply_brdf_topo_core=lambda *a, **k: None, build_correction_parameters_dict=lambda *a, **k: {})
_stub_module("cross_sensor_cal.brightness_config", load_brightness_coefficients=lambda *a, **k: {})
_stub_module("cross_sensor_cal.paths", normalize_brdf_model_path=lambda *a, **k: Path("model.json"))
_stub_module("cross_sensor_cal.qa_plots", render_flightline_panel=lambda *a, **k: None)
_stub_module("cross_sensor_cal.resample", resample_chunk_to_sensor=lambda *a, **k: None)
_stub_module(
    "cross_sensor_cal.sensor_panel_plots",
    make_micasense_vs_landsat_panels=lambda *a, **k: None,
    make_sensor_vs_neon_panels=lambda *a, **k: None,
)
_stub_module("cross_sensor_cal.utils_checks", is_valid_json=lambda *_: True)
_stub_module("cross_sensor_cal.envi_download", download_neon_file=lambda *a, **k: Path("dummy"))
_stub_module("cross_sensor_cal.file_sort", generate_file_move_list=lambda *a, **k: [])
_stub_module("cross_sensor_cal.mask_raster", mask_raster_with_polygons=lambda *a, **k: None)
_stub_module("cross_sensor_cal.merge_duckdb", merge_flightline=lambda *a, **k: Path("merged.parquet"))
_stub_module("cross_sensor_cal.neon_to_envi", neon_to_envi_no_hytools=lambda *a, **k: None)
_stub_module("cross_sensor_cal.polygon_extraction", control_function_for_extraction=lambda *a, **k: None)


class _FakeTileProgressReporter:
    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None


_stub_module("cross_sensor_cal.progress_utils", TileProgressReporter=_FakeTileProgressReporter)
_stub_module("cross_sensor_cal.standard_resample", translate_to_other_sensors=lambda *a, **k: None)


class _FakeFileType:
    def __init__(self, *args, **kwargs) -> None:
        pass


_stub_module(
    "cross_sensor_cal.file_types",
    NEONReflectanceBRDFCorrectedENVIFile=_FakeFileType,
    NEONReflectanceENVIFile=_FakeFileType,
    NEONReflectanceResampledENVIFile=_FakeFileType,
)

pipeline_path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cross_sensor_cal"
    / "pipelines"
    / "pipeline.py"
)
pipeline_spec = importlib.util.spec_from_file_location(
    "cross_sensor_cal.pipelines.pipeline", pipeline_path
)
assert pipeline_spec and pipeline_spec.loader
pipeline_module = importlib.util.module_from_spec(pipeline_spec)
pipeline_spec.loader.exec_module(pipeline_module)
sys.modules.setdefault("cross_sensor_cal.pipelines.pipeline", pipeline_module)

parquet_export_path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cross_sensor_cal"
    / "parquet_export.py"
)
parquet_export_spec = importlib.util.spec_from_file_location(
    "cross_sensor_cal.parquet_export", parquet_export_path
)
assert parquet_export_spec and parquet_export_spec.loader
parquet_export_module = importlib.util.module_from_spec(parquet_export_spec)
parquet_export_spec.loader.exec_module(parquet_export_module)
sys.modules.setdefault("cross_sensor_cal.parquet_export", parquet_export_module)

_export_parquet_stage = pipeline_module._export_parquet_stage
ensure_parquet_for_envi = parquet_export_module.ensure_parquet_for_envi


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
        _write_minimal_parquet(parquet_path)

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
        _write_minimal_parquet(out_path)

    monkeypatch.setattr(px, "build_parquet_from_envi", fake_build)

    logger = DummyLogger2()
    result_path = ensure_parquet_for_envi(img, logger)
    assert result_path == parquet_path
    assert calls["count"] == 1
    assert any("invalid" in msg for msg in logger.warnings)
def _write_minimal_parquet(path: Path) -> None:
    table = pa.table(
        {
            "pixel_id": [1],
            "row": [0],
            "col": [0],
            "lon": [0.0],
            "lat": [0.0],
        }
    )
    pq.write_table(table, path)


