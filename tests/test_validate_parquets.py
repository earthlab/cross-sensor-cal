from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _load_validator():
    script_path = Path(__file__).resolve().parents[1] / "bin" / "validate_parquets"
    loader = SourceFileLoader("validate_parquets", str(script_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_validate_parquets_hard_and_soft(tmp_path: Path, capsys) -> None:
    validator = _load_validator()

    good = tmp_path / "good.parquet"
    table = pa.table(
        {
            "lon": [0.0],
            "lat": [0.0],
            "stage_b001_wl0500nm": [0.1],
        }
    )
    pq.write_table(table, good)

    exit_code = validator.main([str(tmp_path)])
    output = capsys.readouterr()
    assert exit_code == 0
    assert "✅" in output.out

    bad = tmp_path / "bad.parquet"
    bad.write_text("not parquet", encoding="utf-8")

    exit_code = validator.main([str(tmp_path)])
    output = capsys.readouterr()
    assert exit_code == 1
    assert "❌ Issues found:" in output.out
    assert "bad.parquet" in output.out

    exit_code = validator.main(["--soft", str(tmp_path)])
    output = capsys.readouterr()
    assert exit_code == 0
    assert "❌ Issues found:" in output.out
