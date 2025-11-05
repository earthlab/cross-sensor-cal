from __future__ import annotations

from pathlib import Path

import pytest

try:
    import duckdb  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment-specific skip
    pytest.skip("duckdb is required for merge tests", allow_module_level=True)

from cross_sensor_cal.merge_duckdb import merge_flightline


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _write_parquet(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("at least one row is required to seed parquet fixtures")

    columns = list(rows[0].keys())
    base_keys = set(columns)
    for row in rows:
        if set(row.keys()) != base_keys:
            raise ValueError("all rows must define the same columns for parquet fixtures")
    placeholders = "(" + ", ".join(["?"] * len(columns)) + ")"
    values_clause = ", ".join([placeholders] * len(rows))
    flat_params: list[object] = []
    for row in rows:
        flat_params.extend(row[col] for col in columns)

    column_sql = ", ".join(_quote_identifier(col) for col in columns)

    with duckdb.connect() as con:
        con.execute(
            f"""
            CREATE TABLE temp_parquet_fixture AS
            SELECT *
            FROM (VALUES {values_clause}) AS t({column_sql})
            """,
            flat_params,
        )
        con.execute("COPY temp_parquet_fixture TO ? (FORMAT PARQUET)", [str(path)])


def test_duckdb_merge_smoke(tmp_path: Path) -> None:
    flight_dir = tmp_path / "NEON_TEST_FLIGHT"
    flight_dir.mkdir()

    wavelengths = range(1, 427)
    pixel_ids = ["pix0", "pix1", "pix2"]

    # Long layout (original)
    long_rows: list[dict[str, object]] = []
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
    _write_parquet(long_rows, flight_dir / "orig" / "test_original_table.parquet")

    # Wide layout (corrected)
    wide_records: list[dict[str, object]] = []
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
    _write_parquet(wide_records, flight_dir / "corr" / "test_corrected_table.parquet")

    # Long layout with micrometer wavelengths (resampled)
    resamp_records: list[dict[str, object]] = []
    resamp_wavelengths = range(500, 520)
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
    _write_parquet(resamp_records, flight_dir / "resamp" / "test_resampled_table.parquet")

    output_path = merge_flightline(flight_dir, emit_qa_panel=False)
    with duckdb.connect() as con:
        rel = con.from_parquet(str(output_path))
        columns = rel.columns

        pixel_stats = con.execute(
            "SELECT COUNT(*) = COUNT(DISTINCT pixel_id) FROM read_parquet(?)",
            [str(output_path)],
        ).fetchone()
        assert pixel_stats and pixel_stats[0]

        orig_cols = [c for c in columns if c.startswith("orig_wl")]
        corr_cols = [c for c in columns if c.startswith("corr_") and "_wl" in c]
        resamp_cols = [c for c in columns if c.startswith("resamp_") and "_wl" in c]

        assert len(orig_cols) == 426
        assert len(corr_cols) == 426
        assert len(resamp_cols) > 0

        for meta in ("site", "domain", "flightline", "row", "col"):
            assert meta in columns
            missing = con.execute(
                f"SELECT COUNT(*) FROM read_parquet(?) WHERE {meta} IS NULL",
                [str(output_path)],
            ).fetchone()[0]
            assert missing == 0


def test_master_parquet_naming(tmp_path):
    fl = tmp_path / "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance"
    fl.mkdir()
    (fl / "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance_envi.img").write_bytes(b"")

    # Seed minimal parquet inputs so the merge can succeed while focusing the
    # assertion on the derived output naming.
    _write_parquet(
        [
            {
                "pixel_id": "p0",
                "wavelength_nm": 500.0,
                "reflectance": 0.1,
            }
        ],
        fl / "orig" / "dummy_original.parquet",
    )

    _write_parquet(
        [
            {
                "pixel_id": "p0",
                "corr_b001_wl0500nm": 0.2,
            }
        ],
        fl / "corr" / "dummy_corrected.parquet",
    )

    _write_parquet(
        [
            {
                "pixel_id": "p0",
                "resamp_b001_wl0500nm": 0.3,
            }
        ],
        fl / "resamp" / "dummy_resampled.parquet",
    )

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
    _write_parquet(
        [
            {
                "pixel_id": "p0",
                "wavelength_nm": 500.0,
                "reflectance": 0.1,
            }
        ],
        flight_dir / f"{prefix}_envi.parquet",
    )

    _write_parquet(
        [
            {
                "pixel_id": "p0",
                "corr_wl0500": 0.2,
            }
        ],
        flight_dir / f"{prefix}_brdfandtopo_corrected_envi.parquet",
    )

    _write_parquet(
        [
            {
                "pixel_id": "p0",
                "resamp_wl0500": 0.3,
            }
        ],
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
