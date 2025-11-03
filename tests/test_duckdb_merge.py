from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import duckdb  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment-specific skip
    pytest.skip("duckdb is required for merge tests", allow_module_level=True)

from cross_sensor_cal.merge_duckdb import merge_flightline


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_duckdb_merge_smoke(tmp_path: Path) -> None:
    flight_dir = tmp_path / "NEON_TEST_FLIGHT"
    flight_dir.mkdir()

    wavelengths = np.arange(1, 427)
    pixel_ids = ["pix0", "pix1", "pix2"]

    # Long layout (original)
    long_rows = []
    for idx, pid in enumerate(pixel_ids):
        for wl in wavelengths:
            long_rows.append(
                {
                    "pixel_id": pid,
                    "wavelength_nm": float(wl),
                    "reflectance": (wl + idx) / 1000.0,
                    "site": "TEST",
                    "domain": "D00",
                    "flightline": "FLIGHT",
                    "row": idx,
                    "col": idx + 10,
                }
            )
    long_df = pd.DataFrame(long_rows)
    _write_parquet(long_df, flight_dir / "orig" / "test_original_table.parquet")

    # Wide layout (corrected)
    wide_records = []
    for idx, pid in enumerate(pixel_ids):
        record = {
            "pixel_id": pid,
            "site": "TEST",
            "domain": "D00",
            "flightline": "FLIGHT",
            "row": idx,
            "col": idx + 10,
        }
        for band_idx, wl in enumerate(wavelengths, 1):
            record[f"corr_b{band_idx:03d}_wl{wl:04d}nm"] = (wl + idx) / 2000.0
        wide_records.append(record)
    wide_df = pd.DataFrame(wide_records)
    _write_parquet(wide_df, flight_dir / "corr" / "test_corrected_table.parquet")

    # Long layout with micrometer wavelengths (resampled)
    resamp_records = []
    resamp_wavelengths = np.arange(500, 520)
    for idx, pid in enumerate(pixel_ids):
        record = {
            "pixel_id": pid,
            "site": "TEST",
            "domain": "D00",
            "flightline": "FLIGHT",
        }
        for band_idx, wl in enumerate(resamp_wavelengths, 1):
            record[f"resamp_b{band_idx:03d}_wl{wl:04d}nm"] = (wl + idx) / 3000.0
        resamp_records.append(record)
    resamp_df = pd.DataFrame(resamp_records)
    _write_parquet(resamp_df, flight_dir / "resamp" / "test_resampled_table.parquet")

    output_path = merge_flightline(flight_dir, emit_qa_panel=False)
    merged = pd.read_parquet(output_path)

    assert merged["pixel_id"].is_unique

    orig_cols = [c for c in merged.columns if c.startswith("orig_wl")]
    corr_cols = [c for c in merged.columns if c.startswith("corr_") and "_wl" in c]
    resamp_cols = [c for c in merged.columns if c.startswith("resamp_") and "_wl" in c]

    assert len(orig_cols) == 426
    assert len(corr_cols) == 426
    assert len(resamp_cols) > 0

    for meta in ("site", "domain", "flightline", "row", "col"):
        assert meta in merged.columns
        assert merged[meta].notna().all()


def test_master_parquet_naming(tmp_path):
    fl = tmp_path / "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"
    fl.mkdir()
    (fl / "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance_envi.img").write_bytes(b"")

    # Seed minimal parquet inputs so the merge can succeed while focusing the
    # assertion on the derived output naming.
    orig_df = pd.DataFrame(
        {
            "pixel_id": ["p0"],
            "wavelength_nm": [500.0],
            "reflectance": [0.1],
        }
    )
    _write_parquet(orig_df, fl / "orig" / "dummy_original.parquet")

    corr_df = pd.DataFrame(
        {
            "pixel_id": ["p0"],
            "corr_b001_wl0500nm": [0.2],
        }
    )
    _write_parquet(corr_df, fl / "corr" / "dummy_corrected.parquet")

    resamp_df = pd.DataFrame(
        {
            "pixel_id": ["p0"],
            "resamp_b001_wl0500nm": [0.3],
        }
    )
    _write_parquet(resamp_df, fl / "resamp" / "dummy_resampled.parquet")

    out = merge_flightline(
        fl,
        out_name=None,
        original_glob="orig/*.parquet",
        corrected_glob="corr/*.parquet",
        resampled_glob="resamp/*.parquet",
        emit_qa_panel=False,
    )
    assert (
        out.name
        == "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance_merged_pixel_extraction.parquet"
    )


def _seed_minimal_tables(flight_dir: Path) -> None:
    prefix = flight_dir.name
    orig_df = pd.DataFrame(
        {
            "pixel_id": ["p0"],
            "wavelength_nm": [500.0],
            "reflectance": [0.1],
        }
    )
    _write_parquet(orig_df, flight_dir / f"{prefix}_envi.parquet")

    corr_df = pd.DataFrame(
        {
            "pixel_id": ["p0"],
            "corr_wl0500": [0.2],
        }
    )
    _write_parquet(
        corr_df,
        flight_dir / f"{prefix}_brdfandtopo_corrected_envi.parquet",
    )

    resamp_df = pd.DataFrame(
        {
            "pixel_id": ["p0"],
            "resamp_wl0500": [0.3],
        }
    )
    _write_parquet(
        resamp_df,
        flight_dir / f"{prefix}_resampled_Landsat_8_OLI_envi.parquet",
    )


def test_merge_skips_invalid_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    flight_dir = tmp_path / "NEON_TEST_FLIGHT_INVALID"
    flight_dir.mkdir()
    (flight_dir / "NEON_TEST_FLIGHT_INVALID_envi.img").write_bytes(b"")

    _seed_minimal_tables(flight_dir)

    bad = flight_dir / f"{flight_dir.name}_extra_envi.parquet"
    bad.write_text("not a parquet file", encoding="utf-8")

    out = merge_flightline(flight_dir, emit_qa_panel=False)

    captured = capsys.readouterr()
    assert f"Skipping invalid parquet {bad.name}" in captured.out
    assert out.exists()


def test_merge_errors_when_category_empty_after_skip(tmp_path: Path) -> None:
    flight_dir = tmp_path / "NEON_TEST_FLIGHT_ALL_INVALID"
    flight_dir.mkdir()
    (flight_dir / "NEON_TEST_FLIGHT_ALL_INVALID_envi.img").write_bytes(b"")

    broken = flight_dir / f"{flight_dir.name}_brdfandtopo_corrected_envi.parquet"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("still not parquet", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as excinfo:
        merge_flightline(flight_dir, emit_qa_panel=False)

    assert "Remove or recreate the invalid files" in str(excinfo.value)
