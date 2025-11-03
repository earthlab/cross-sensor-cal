"""cross_sensor_cal.merge_duckdb
================================

DuckDB-backed merge utilities for the cross-sensor-cal pipeline.

The public entry point :func:`merge_flightline` stitches every ENVI-derived
Parquet table for a single flightline into one master artifact named
``<scene_prefix>_merged_pixel_extraction.parquet``. The merged parquet includes
pixel metadata, the full set of original/corrected wavelengths, and any
resampled sensor bands.

Unless ``qa=False`` is passed, completing the merge automatically invokes
``qa_plots.render_flightline_panel`` to emit ``<scene_prefix>_qa.png`` in the
same directory. This mirrors the default behaviour of
``python -m bin.merge_duckdb`` as documented in the CLI.

Typical usage::

    from cross_sensor_cal import merge_duckdb

    merge_duckdb.merge_flightline(
        Path("/path/to/flightline"),
        out_name="custom_master.parquet",  # optional override
        qa=True,
    )

The module intentionally minimises assumptions about directory layout. The
default glob patterns match the canonical pipeline exports but can be overridden
to support custom workflows.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb


def _consolidate_parquet_dir_to_file(
    con: duckdb.DuckDBPyConnection, dir_path: Path, out_file: Path
):
    dir_path = Path(dir_path)
    if dir_path.is_dir():
        parts = sorted(glob.glob(str(dir_path / "*.parquet")))
        if parts:
            tmp = out_file.with_suffix(".tmp.parquet")
            con.execute(
                f"""
                COPY (SELECT * FROM read_parquet('{dir_path}/*.parquet'))
                TO '{str(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            tmp.replace(out_file)
        shutil.rmtree(dir_path, ignore_errors=True)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None


# Files we must exclude from any input scan to avoid schema collisions
EXCLUDE_PATTERNS = {
    "_merged_pixel_extraction.parquet",
    "_qa_metrics.parquet",
}


def _exclude_parquets(paths):
    keep = []
    for p in paths:
        name = p.name if hasattr(p, "name") else str(p)
        if any(ex in name for ex in EXCLUDE_PATTERNS):
            continue
        keep.append(Path(p))
    return keep


def _filter_valid_parquets(paths: Iterable[Path]) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    valid: List[Path] = []
    skipped: List[Tuple[Path, str]] = []
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - environment expectation
        raise RuntimeError(
            "pyarrow is required to validate parquet inputs before merge"
        ) from exc

    for path in paths:
        try:
            pq.ParquetFile(path.as_posix())
        except Exception as exc:  # pragma: no cover - corruption varies by test data
            skipped.append((path, str(exc)))
        else:
            valid.append(path)
    return valid, skipped


@contextmanager
def _progress(desc: str):
    bar = None
    try:
        if tqdm:
            bar = tqdm(total=1, desc=desc, leave=False, disable=not sys.stderr.isatty())
        yield
        if bar:
            bar.update(1)
    finally:
        if bar:
            bar.close()


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
    flightline_dir = Path(flightline_dir)
    # Prefer *_envi.img to deduce prefix; else stem of first *.h5; else folder name
    imgs = sorted(flightline_dir.glob("*_envi.img"))
    if imgs:
        stem = imgs[0].stem
        return stem[:-5] if stem.endswith("_envi") else stem
    h5s = sorted(flightline_dir.glob("*_directional_reflectance.h5"))
    if h5s:
        return h5s[0].stem
    return flightline_dir.name


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
        if "_wl" in col:
            spectral.append(col)
        elif col.startswith("wl") and col[2:].isdigit():
            spectral.append(col)
        elif col.isdigit():
            spectral.append(col)

    def _sort_key(name: str) -> int:
        match = re.search(r"wl(\d+)", name)
        if match:
            return int(match.group(1))
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0

    seen: set[str] = set()
    ordered: List[str] = []
    for col in sorted(spectral, key=_sort_key):
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def _quote_path(path: str) -> str:
    return path.replace("'", "''")


def _register_union(
    con: duckdb.DuckDBPyConnection,
    name: str,
    paths: Iterable[Path],
    meta_candidates: Sequence[str],
) -> None:
    escaped_name = _quote_identifier(f"{name}_raw")
    path_list = sorted(Path(p) for p in paths)

    if path_list:
        valid_paths, skipped = _filter_valid_parquets(path_list)
        for bad_path, message in skipped:
            print(
                f"[merge] ⚠️ Skipping invalid parquet {bad_path.name}: {message}\n"
                f"        Delete or regenerate this file before re-running the merge."
            )
        if not valid_paths and skipped:
            raise FileNotFoundError(
                "No valid parquet files remain for "
                f"'{name}'. Remove or recreate the invalid files: "
                + ", ".join(bad.name for bad, _ in skipped)
            )
        path_list = valid_paths

    if not path_list:
        con.execute(
            f"CREATE OR REPLACE VIEW {escaped_name} AS SELECT * FROM (SELECT 1) WHERE 1=0"
        )
    elif len(path_list) == 1:
        p = path_list[0]
        con.execute(
            "CREATE OR REPLACE VIEW "
            + escaped_name
            + " AS SELECT * FROM read_parquet('"
            + _quote_path(p.as_posix())
            + "', union_by_name = TRUE)"
        )
    else:
        files_array = ", ".join(
            "'" + _quote_path(p.as_posix()) + "'" for p in path_list
        )
        sql = (
            "CREATE OR REPLACE VIEW "
            + escaped_name
            + " AS SELECT * FROM read_parquet(["
            + files_array
            + "], union_by_name = TRUE)"
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
            if "_wl" in col:
                select_final.append(_quote_identifier(col))
            else:
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

    flightline_dir = Path(flightline_dir).resolve()
    prefix = _derive_prefix(flightline_dir)
    if out_name is None:
        out_name = f"{prefix}_merged_pixel_extraction.parquet"
    out_parquet = (flightline_dir / out_name).resolve()

    print(f"[merge] Start flightline={flightline_dir.name} prefix={prefix}")
    print(f"[merge] Output parquet → {out_parquet}")

    def _discover_inputs() -> Dict[str, List[Path]]:
        # When default globs are supplied, favour precise patterns with exclusions.
        if (
            original_glob == "**/*original*.parquet"
            and corrected_glob == "**/*corrected*.parquet"
            and resampled_glob == "**/*resampl*.parquet"
        ):
            inputs: Dict[str, List[Path]] = {"orig": [], "corr": [], "resamp": []}

            # originals (long format from raw ENVI)
            orig = sorted(flightline_dir.glob("*_envi.parquet"))
            orig = [p for p in orig if "brdfandtopo_corrected" not in p.name]
            inputs["orig"] = _exclude_parquets(orig)

            # corrected (long format from corrected ENVI)
            corr = sorted(
                flightline_dir.glob("*_brdfandtopo_corrected_envi.parquet")
            )
            inputs["corr"] = _exclude_parquets(corr)

            # resampled (sensor stacks)
            resamp: List[Path] = []
            for pat in [
                "*_landsat_*_envi.parquet",
                "*_micasense*_envi.parquet",
            ]:
                resamp.extend(sorted(flightline_dir.glob(pat)))
            inputs["resamp"] = _exclude_parquets(resamp)
            if not any(inputs.values()):
                return {
                    "orig": _exclude_parquets(
                        sorted(flightline_dir.glob(original_glob))
                    ),
                    "corr": _exclude_parquets(
                        sorted(flightline_dir.glob(corrected_glob))
                    ),
                    "resamp": _exclude_parquets(
                        sorted(flightline_dir.glob(resampled_glob))
                    ),
                }
            return inputs

        # Fallback to user-supplied glob patterns
        return {
            "orig": _exclude_parquets(sorted(flightline_dir.glob(original_glob))),
            "corr": _exclude_parquets(sorted(flightline_dir.glob(corrected_glob))),
            "resamp": _exclude_parquets(sorted(flightline_dir.glob(resampled_glob))),
        }

    inputs = _discover_inputs()

    if not any(inputs.values()):
        raise FileNotFoundError(f"No parquet inputs found in {flightline_dir}")

    con = duckdb.connect()
    try:
        tmp_dir = (flightline_dir / ".duckdb_tmp").resolve()
        tmp_dir.mkdir(exist_ok=True)

        con.execute("PRAGMA threads = " + str(os.cpu_count() or 4))
        try:
            con.execute("PRAGMA memory_limit = 'auto'")
        except duckdb.Error as exc:  # pragma: no cover - parser differences across DuckDB versions
            if "memory limit" not in str(exc).lower():
                raise
        con.execute("SET enable_progress_bar = true")
        con.execute(
            "PRAGMA temp_directory = '"
            + _quote_path(str(tmp_dir))
            + "'"
        )

        with _progress("DuckDB: register + normalize"):
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

            orig_spectral = [c for c in orig_cols if "_wl" in c]
            corr_spectral = [c for c in corr_cols if "_wl" in c]
            resamp_spectral = [c for c in resamp_cols if "_wl" in c]

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

        with _progress("DuckDB: join + materialize"):
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
        if out_parquet.exists():
            if out_parquet.is_dir():
                shutil.rmtree(out_parquet, ignore_errors=True)
            else:
                out_parquet.unlink()

        with _progress("DuckDB: write merged parquet"):
            con.execute(
                f"""
      COPY merged TO '{_quote_path(str(out_parquet))}'
      (FORMAT PARQUET,
       COMPRESSION ZSTD,
       ROW_GROUP_SIZE 8388608,
       PER_THREAD_OUTPUT FALSE);
    """
            )
        _consolidate_parquet_dir_to_file(con, Path(out_parquet), Path(out_parquet))
        print(
            f"[merge] ✅ Wrote parquet: {out_parquet} (exists={Path(out_parquet).exists()})"
        )

        if write_feather:
            out_feather = out_parquet.with_suffix(".feather").resolve()
            with _progress("DuckDB: write feather"):
                con.execute(
                    f"COPY merged TO '{_quote_path(str(out_feather))}' (FORMAT ARROW)"
                )
                print(
                    f"[merge] ✅ Wrote feather: {out_feather} (exists={out_feather.exists()})"
                )
    finally:
        con.close()

    if hasattr(os, "sync"):
        try:
            os.sync()
        except OSError:
            pass

    if emit_qa_panel:
        try:
            from cross_sensor_cal.qa_plots import render_flightline_panel

            with _progress("QA: render panel"):
                out_png_path, _ = render_flightline_panel(
                    flightline_dir, quick=True, save_json=True
                )
            print(
                f"[merge] 🖼️  QA panel → {out_png_path} "
                f"(exists={out_png_path.exists()})"
            )
        except Exception as e:  # pragma: no cover - QA best effort
            print(f"[merge] ⚠️ QA panel after merge failed for {prefix}: {e}")
            traceback.print_exc()

    print(
        f"[merge] ✅ Done for {flightline_dir.name} at {datetime.now().isoformat(timespec='seconds')}"
    )
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
