from __future__ import annotations

import json
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


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

    fake_pa.table = table
    fake_pa.Table = _FakeTable
    fake_parquet.write_table = write_table
    fake_parquet.read_table = read_table
    fake_parquet.read_schema = read_schema

    sys.modules["pyarrow"] = fake_pa
    sys.modules["pyarrow.parquet"] = fake_parquet

