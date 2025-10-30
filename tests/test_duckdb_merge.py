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
        for wl in wavelengths:
            record[f"wl{wl:04d}"] = (wl + idx) / 2000.0
        wide_records.append(record)
    wide_df = pd.DataFrame(wide_records)
    _write_parquet(wide_df, flight_dir / "corr" / "test_corrected_table.parquet")

    # Long layout with micrometer wavelengths (resampled)
    resamp_rows = []
    resamp_wavelengths = np.arange(500, 520)
    for idx, pid in enumerate(pixel_ids):
        for wl in resamp_wavelengths:
            resamp_rows.append(
                {
                    "pixel_id": pid,
                    "wavelength_um": wl / 1000.0,
                    "reflectance": (wl + idx) / 3000.0,
                    "site": "TEST",
                    "domain": "D00",
                    "flightline": "FLIGHT",
                }
            )
    resamp_df = pd.DataFrame(resamp_rows)
    _write_parquet(resamp_df, flight_dir / "resamp" / "test_resampled_table.parquet")

    output_path = merge_flightline(flight_dir, emit_qa_panel=False)
    merged = pd.read_parquet(output_path)

    assert merged["pixel_id"].is_unique

    orig_cols = [c for c in merged.columns if c.startswith("orig_wl")]
    corr_cols = [c for c in merged.columns if c.startswith("corr_wl")]
    resamp_cols = [c for c in merged.columns if c.startswith("resamp_wl")]

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
            "wl0500": [0.2],
        }
    )
    _write_parquet(corr_df, fl / "corr" / "dummy_corrected.parquet")

    resamp_df = pd.DataFrame(
        {
            "pixel_id": ["p0"],
            "wavelength_um": [0.5],
            "reflectance": [0.3],
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
