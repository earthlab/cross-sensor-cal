from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_validator():
    script_path = Path(__file__).resolve().parents[1] / "bin" / "validate_parquets"
    spec = importlib.util.spec_from_file_location("validate_parquets", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_parquets_hard_and_soft(tmp_path: Path, capsys) -> None:
    validator = _load_validator()

    good = tmp_path / "good.parquet"
    df = pd.DataFrame(
        {
            "lon": [0.0],
            "lat": [0.0],
            "stage_b001_wl0500nm": [0.1],
        }
    )
    df.to_parquet(good)

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
