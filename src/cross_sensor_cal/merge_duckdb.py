"""
cross_sensor_cal.merge_duckdb
-----------------------------

DuckDB-based merge stage that unifies all parquet tables for a NEON flightline.

Each run creates one master parquet per flightline:
    <scene_prefix>_merged_pixel_extraction.parquet

The merged table contains:
  • all pixel-level metadata (row, col, x, y, lat, lon, etc.)
  • 426 original wavelengths (columns prefixed 'orig_wl')
  • 426 corrected wavelengths (columns prefixed 'corr_wl')
  • resampled wavelengths for each target sensor (columns prefixed 'resamp_wl')

After writing the merged parquet, the stage automatically triggers the QA panel
builder (see `qa_plots.render_flightline_panel`) to produce:
    <scene_prefix>_qa.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import duckdb


META_CANDIDATES: Sequence[str] = (
    "pixel_id",
    "site",
    "domain",
    "flightline",
    "date",
    "utm_zone",
    "epsg",
    "x",
    "y",
    "lat",
    "lon",
    "row",
    "col",
    "mask",
    "qa",
)


def _derive_prefix(flightline_dir: Path) -> str:
    """
    Return the canonical scene/file prefix for this flightline (same as other artifacts).
    Prefers any "*_envi.img" in the folder; strips the trailing "_envi" from the stem.
    Falls back to the folder name.
    """

    flightline_dir = Path(flightline_dir)
    prefix = flightline_dir.name
    candidates = sorted(flightline_dir.glob("*_envi.img"))
    if candidates:
        stem = candidates[0].stem
        prefix = stem[:-5] if stem.endswith("_envi") else stem
    return prefix


@dataclass(frozen=True)
class MergeInputs:
    flightline_dir: Path
    original_glob: str = "**/*original*.parquet"
    corrected_glob: str = "**/*corrected*.parquet"
    resampled_glob: str = "**/*resampl*.parquet"

    def discover(self) -> Dict[str, List[str]]:
        def _glob(pattern: str) -> List[str]:
            return [str(p) for p in self.flightline_dir.glob(pattern)]

        return {
            "orig": _glob(self.original_glob),
            "corr": _glob(self.corrected_glob),
            "resamp": _glob(self.resampled_glob),
        }


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> List[str]:
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE UPPER(table_schema) = 'MAIN'
          AND UPPER(table_name) = UPPER(?);
        """,
        [table],
    ).fetchall()
    return [r[0] for r in rows]


def _quote_identifier(identifier: str) -> str:
    """Return a DuckDB-safe identifier."""

    return '"' + identifier.replace('"', '""') + '"'


def _list_spectral_columns(columns: Iterable[str]) -> List[str]:
    spectral: List[str] = []
    for col in columns:
        if col.isdigit():
            spectral.append(col)
        elif col.startswith("wl") and col[2:].isdigit():
            spectral.append(col)
    return sorted(
        spectral,
        key=lambda name: int("".join(ch for ch in name if ch.isdigit()) or 0),
    )


def _quote_path(path: str) -> str:
    return path.replace("'", "''")


def _register_union(
    con: duckdb.DuckDBPyConnection,
    name: str,
    paths: List[str],
    meta_candidates: Sequence[str],
) -> None:
    escaped_name = _quote_identifier(f"{name}_raw")
    if not paths:
        # Create an empty view with expected metadata so downstream SQL succeeds.
        select_cols = ["NULL::VARCHAR AS pixel_id"]
        for col in meta_candidates:
            if col == "pixel_id":
                continue
            select_cols.append(f"NULL AS {_quote_identifier(col)}")
        con.execute(
            f"CREATE OR REPLACE VIEW {_quote_identifier(name)} AS SELECT {', '.join(select_cols)} WHERE 0=1"
        )
        return

    if len(paths) == 1:
        sql = (
            f"CREATE OR REPLACE VIEW {escaped_name} AS "
            f"SELECT * FROM read_parquet('{_quote_path(paths[0])}')"
        )
    else:
        path_list = ", ".join(f"'{_quote_path(p)}'" for p in paths)
        sql = (
            f"CREATE OR REPLACE VIEW {escaped_name} AS "
            f"SELECT * FROM read_parquet([{path_list}])"
        )
    con.execute(sql)

    raw_columns = _table_columns(con, f"{name}_raw")
    raw_column_set = set(raw_columns)

    pixel_expr_parts: List[str] = []
    if "pixel_id" in raw_column_set:
        pixel_expr_parts.append("CAST(pixel_id AS VARCHAR)")
    if {"row", "col"}.issubset(raw_column_set):
        pixel_expr_parts.append(
            "CONCAT('r', CAST(row AS VARCHAR), '_c', CAST(col AS VARCHAR))"
        )
    if {"x", "y"}.issubset(raw_column_set):
        pixel_expr_parts.append(
            "CONCAT('x', CAST(ROUND(x, 3) AS VARCHAR), '_y', CAST(ROUND(y, 3) AS VARCHAR))"
        )
    if not pixel_expr_parts:
        pixel_expr_parts.append("'pixel_' || ROW_NUMBER() OVER ()")
    pixel_expr = "COALESCE(" + ", ".join(pixel_expr_parts) + ")"

    wavelength_cases: List[str] = []
    if "wavelength_nm" in raw_column_set:
        wavelength_cases.append(
            "WHEN wavelength_nm IS NOT NULL THEN CAST(ROUND(wavelength_nm) AS INTEGER)"
        )
    if "wavelength_um" in raw_column_set:
        wavelength_cases.append(
            "WHEN wavelength_um IS NOT NULL THEN CAST(ROUND(wavelength_um * 1000) AS INTEGER)"
        )
    wl_expr = (
        "CASE " + " ".join(wavelength_cases) + " ELSE NULL END"
        if wavelength_cases
        else "NULL"
    )

    spectral_cols = _list_spectral_columns(raw_columns)

    select_parts: List[str] = [f"{pixel_expr} AS pixel_id"]
    for col in meta_candidates:
        if col == "pixel_id":
            continue
        if col in raw_column_set:
            select_parts.append(_quote_identifier(col))
        else:
            select_parts.append(f"NULL AS {_quote_identifier(col)}")

    select_parts.append(f"{wl_expr} AS wl_nm_int")

    if "reflectance" in raw_column_set:
        select_parts.append("reflectance")
    else:
        select_parts.append("NULL AS reflectance")

    for col in spectral_cols:
        select_parts.append(_quote_identifier(col))

    con.execute(
        f"CREATE OR REPLACE VIEW {_quote_identifier(name + '_aug')} AS SELECT "
        + ", ".join(select_parts)
        + f" FROM {escaped_name}"
    )

    wl_values = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT wl_nm_int FROM {_quote_identifier(name + '_aug')} "
            "WHERE wl_nm_int IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]

    if wl_values:
        meta_selects = ["pixel_id"]
        for col in meta_candidates:
            if col == "pixel_id":
                continue
            meta_selects.append(
                f"ANY_VALUE({_quote_identifier(col)}) AS {_quote_identifier(col)}"
            )
        spectral_selects = [
            f"MAX(CASE WHEN wl_nm_int = {val} THEN reflectance END) AS {_quote_identifier(f'wl{val}') }"
            for val in wl_values
        ]
        con.execute(
            f"CREATE OR REPLACE VIEW {_quote_identifier(name + '_wide')} AS SELECT "
            + ", ".join(meta_selects + spectral_selects)
            + f" FROM {_quote_identifier(name + '_aug')} GROUP BY pixel_id"
        )
    else:
        aug_columns = _table_columns(con, f"{name}_aug")
        keep_cols = [
            col
            for col in aug_columns
            if col not in {"wl_nm_int", "reflectance"}
        ]
        select_keep = ", ".join(_quote_identifier(col) for col in keep_cols)
        con.execute(
            f"CREATE OR REPLACE VIEW {_quote_identifier(name + '_wide')} AS SELECT "
            + select_keep
            + f" FROM {_quote_identifier(name + '_aug')}"
        )

    wide_columns = _table_columns(con, f"{name}_wide")
    spectral_final = _list_spectral_columns(wide_columns)
    select_final: List[str] = []
    for col in wide_columns:
        if col in spectral_final:
            nm = col[2:] if col.startswith("wl") else col
            alias = _quote_identifier(f"{name}_wl{nm}")
            select_final.append(f"{_quote_identifier(col)} AS {alias}")
        else:
            select_final.append(_quote_identifier(col))

    for col in meta_candidates:
        if col not in wide_columns and col != "pixel_id":
            select_final.append(f"NULL AS {_quote_identifier(col)}")

    con.execute(
        f"CREATE OR REPLACE VIEW {_quote_identifier(name)} AS SELECT "
        + ", ".join(select_final)
        + f" FROM {_quote_identifier(name + '_wide')}"
    )


def merge_flightline(
    flightline_dir: Path,
    out_name: str | None = None,
    original_glob: str = "**/*original*.parquet",
    corrected_glob: str = "**/*corrected*.parquet",
    resampled_glob: str = "**/*resampl*.parquet",
    write_feather: bool = False,
    emit_qa_panel: bool = True,
) -> Path:
    """
    Merge all pixel-level parquet tables for one flightline.

    Parameters
    ----------
    flightline_dir : Path
        Directory containing the flightline's parquet outputs.
    out_name : str, optional
        Custom name for the merged parquet. If None, defaults to:
        <prefix>_merged_pixel_extraction.parquet
    original_glob : str, optional
        Glob used to locate original reflectance parquet tables.
    corrected_glob : str, optional
        Glob used to locate corrected reflectance parquet tables.
    resampled_glob : str, optional
        Glob used to locate resampled sensor parquet tables.
    write_feather : bool, optional
        If True, writes a Feather copy of the merged table alongside the parquet.
    emit_qa_panel : bool, default True
        If True, renders the standard QA panel (<prefix>_qa.png) after merging.

    Returns
    -------
    Path
        Path to the merged parquet file.
    """

    flightline_dir = Path(flightline_dir)
    prefix = _derive_prefix(flightline_dir)
    if out_name is None:
        out_name = f"{prefix}_merged_pixel_extraction.parquet"
    out_parquet = flightline_dir / out_name

    inputs = MergeInputs(
        flightline_dir,
        original_glob,
        corrected_glob,
        resampled_glob,
    ).discover()

    if not any(inputs.values()):
        raise FileNotFoundError(f"No parquet inputs found in {flightline_dir}")

    tmp_dir = flightline_dir / ".duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={os.cpu_count() or 4}")
        try:
            con.execute("PRAGMA memory_limit='auto'")
        except duckdb.Error as exc:  # pragma: no cover - parser differences across DuckDB versions
            # Older DuckDB releases require a numeric memory limit. Fall back to the
            # default setting if "auto" is rejected rather than failing the merge.
            if "memory limit" not in str(exc).lower():
                raise
        con.execute(
            f"PRAGMA temp_directory='{_quote_path(str(tmp_dir.resolve()))}'"
        )

        _register_union(con, "orig", inputs["orig"], META_CANDIDATES)
        _register_union(con, "corr", inputs["corr"], META_CANDIDATES)
        _register_union(con, "resamp", inputs["resamp"], META_CANDIDATES)

        orig_cols = set(_table_columns(con, "orig"))
        corr_cols = set(_table_columns(con, "corr"))
        resamp_cols = set(_table_columns(con, "resamp"))

        present_meta = [
            col
            for col in META_CANDIDATES
            if col in orig_cols or col in corr_cols or col in resamp_cols
        ]

        meta_selects: List[str] = ["p.pixel_id AS pixel_id"]
        for col in present_meta:
            if col == "pixel_id":
                continue
            ident = _quote_identifier(col)
            meta_selects.append(
                f"COALESCE(o.{ident}, c.{ident}, r.{ident}) AS {ident}"
            )

        def _spectral_select(table_alias: str, columns: Iterable[str]) -> List[str]:
            selects: List[str] = []
            for col in columns:
                ident = _quote_identifier(col)
                selects.append(f"{table_alias}.{ident} AS {ident}")
            return selects

        orig_spectral = [c for c in orig_cols if c.startswith("orig_wl")]
        corr_spectral = [c for c in corr_cols if c.startswith("corr_wl")]
        resamp_spectral = [c for c in resamp_cols if c.startswith("resamp_wl")]

        select_clause = meta_selects + _spectral_select("o", orig_spectral)
        select_clause += _spectral_select("c", corr_spectral)
        select_clause += _spectral_select("r", resamp_spectral)

        con.execute(
            """
            CREATE OR REPLACE VIEW all_pixels AS (
                SELECT pixel_id FROM orig
                UNION
                SELECT pixel_id FROM corr
                UNION
                SELECT pixel_id FROM resamp
            )
            """
        )

        con.execute(
            "CREATE OR REPLACE TABLE merged AS "
            "SELECT "
            + ", ".join(select_clause)
            + " FROM all_pixels p "
            "LEFT JOIN orig o ON p.pixel_id = o.pixel_id "
            "LEFT JOIN corr c ON p.pixel_id = c.pixel_id "
            "LEFT JOIN resamp r ON p.pixel_id = r.pixel_id"
        )

        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        con.execute(
            f"COPY merged TO '{_quote_path(str(out_parquet))}' (FORMAT PARQUET, CODEC ZSTD)"
        )

        if write_feather:
            out_feather = out_parquet.with_suffix(".feather")
            con.execute(
                f"COPY merged TO '{_quote_path(str(out_feather))}' (FORMAT ARROW)"
            )
    finally:
        con.close()

    if emit_qa_panel:
        try:
            from cross_sensor_cal.qa_plots import render_flightline_panel

            render_flightline_panel(flightline_dir, prefix)
            print(f"🖼️  QA panel written → {prefix}_qa.png")
        except Exception as e:  # pragma: no cover - QA best effort
            print(f"⚠️ QA panel after merge failed for {flightline_dir.name}: {e}")

    return out_parquet


def merge_all_flightlines(
    data_root: Path,
    *,
    flightline_glob: str = "NEON_*",
    **kwargs,
) -> List[Path]:
    data_root = Path(data_root)
    flightline_dirs = sorted(
        [p for p in data_root.glob(flightline_glob) if p.is_dir()]
    )

    if not flightline_dirs:
        return []

    results: List[Path] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        future_map = {
            pool.submit(merge_flightline, flight_dir, **kwargs): flight_dir
            for flight_dir in flightline_dirs
        }
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - worker failure logging
                print(f"⚠️ Merge failed for {future_map[future]}: {exc}")

    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge original/corrected/resampled Parquet tables with DuckDB.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--flightline-glob", type=str, default="NEON_*")
    parser.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="Override output parquet name (default: <prefix>_merged_pixel_extraction.parquet)",
    )
    parser.add_argument("--original-glob", type=str, default="**/*original*.parquet")
    parser.add_argument("--corrected-glob", type=str, default="**/*corrected*.parquet")
    parser.add_argument("--resampled-glob", type=str, default="**/*resampl*.parquet")
    parser.add_argument("--write-feather", action="store_true")
    parser.add_argument("--no-qa", action="store_true", help="Do not render QA panel after merge")
    args = parser.parse_args(argv)

    merge_all_flightlines(
        data_root=args.data_root,
        flightline_glob=args.flightline_glob,
        out_name=args.out_name,
        original_glob=args.original_glob,
        corrected_glob=args.corrected_glob,
        resampled_glob=args.resampled_glob,
        write_feather=args.write_feather,
        emit_qa_panel=(not args.no_qa),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
