from __future__ import annotations

import json
import logging
import os
import sys
import types
from importlib import import_module
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for candidate in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
os.environ.setdefault("GDAL_NUM_THREADS", "1")
os.environ.setdefault("CPL_DEBUG", "OFF")
os.environ.setdefault("PROJ_NETWORK", "OFF")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

for name in ("ray", "osgeo", "fiona", "rasterio"):
    logging.getLogger(name).setLevel(logging.ERROR)


MODE = os.getenv("CSCAL_TEST_MODE", "unit").lower()


def require_mode(expected: str):
    return pytest.mark.skipif(MODE != expected, reason=f"CSCAL_TEST_MODE!='{expected}'")


if "pyarrow" not in sys.modules:  # pragma: no cover - testing fallback
    fake_pa = types.ModuleType("pyarrow")
    fake_parquet = types.ModuleType("pyarrow.parquet")

    class _FakeSchema:
        def __init__(self, names):
            self.names = list(names)

    class _FakeTable(dict):
        @property
        def column_names(self):
            return list(self.keys())

        def to_pandas(self):  # pragma: no cover - minimal placeholder
            raise RuntimeError("pandas is required to convert fake tables to DataFrame")

    def _ensure_iterable(values):
        return list(values)

    def table(mapping):
        return _FakeTable({key: _ensure_iterable(values) for key, values in mapping.items()})

    def write_table(table_obj, path):
        path = Path(path)
        data = {key: list(value) for key, value in dict(table_obj).items()}
        payload = {"columns": list(data.keys())}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def read_table(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return _FakeTable({name: [] for name in data.get("columns", [])})

    def read_schema(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        raw = path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("unable to read schema (corrupted fake parquet)") from exc
        names = data.get("columns")
        if not isinstance(names, list):
            raise ValueError("unable to read schema (missing columns list)")
        return _FakeSchema(names)

    class _FakeParquetFile:
        def __init__(self, path):
            self._path = Path(path)
            self._schema = read_schema(self._path)

        @property
        def schema(self):
            return self._schema

    fake_pa.__version__ = "0.0.0"
    fake_pa.table = table
    fake_pa.Table = _FakeTable
    fake_pa.parquet = fake_parquet
    fake_parquet.write_table = write_table
    fake_parquet.read_table = read_table
    fake_parquet.read_schema = read_schema
    fake_parquet.ParquetFile = _FakeParquetFile

    sys.modules["pyarrow"] = fake_pa
    sys.modules["pyarrow.parquet"] = fake_parquet


if "hytools_compat" not in sys.modules:  # pragma: no cover - simple re-export helper
    sys.modules["hytools_compat"] = import_module("cross_sensor_cal.hytools_compat")


if "matplotlib" not in sys.modules:  # pragma: no cover - provide lightweight stand-in
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")

    def _noop(*args, **kwargs):
        return None

    fake_pyplot.figure = _noop
    fake_pyplot.subplots = lambda *a, **k: (None, None)
    fake_pyplot.imshow = _noop
    fake_pyplot.close = _noop

    fake_matplotlib.pyplot = fake_pyplot

    sys.modules["matplotlib"] = fake_matplotlib
    sys.modules["matplotlib.pyplot"] = fake_pyplot


