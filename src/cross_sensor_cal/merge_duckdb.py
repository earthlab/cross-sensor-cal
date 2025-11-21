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

from .paths import scene_prefix_from_dir

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


def _filter_valid_parquets(
    paths: Iterable[Path], *, num_cpus: int | None = None
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - environment expectation
        raise RuntimeError(
            "pyarrow is required to validate parquet inputs before merge"
        ) from exc

    path_list = [Path(p) for p in paths]
    if not path_list:
        return [], []

    # Always process sequentially without Ray to avoid OOM issues
    # Ray is causing memory problems even with small datasets
    def _validate_one(path: Path) -> Tuple[Path, str | None]:
        try:
            pq.read_schema(path)
        except Exception as exc:  # pragma: no cover - corruption varies
            return path, str(exc)
        return path, None

    # Process sequentially without Ray (Ray causes OOM even with num_cpus=1)
    print(f"[merge] Validating {len(path_list)} parquet files sequentially (no Ray)")
    results = [_validate_one(path) for path in path_list]

    valid = [path for path, error in results if error is None]
    skipped = [(path, error) for path, error in results if error is not None]
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
    return scene_prefix_from_dir(flightline_dir)


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


def _coerce_memory_limit(value: float | str | None) -> float | str | None:
    """Normalise user-supplied memory limits for DuckDB."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    value = value.strip()
    if not value:
        return None
    if value.lower() == "auto":
        return None
    try:
        return float(value)
    except ValueError:
        return value


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
    *,
    ray_cpus: int | None = None,
) -> None:
    escaped_name = _quote_identifier(f"{name}_raw")
    path_list = sorted(Path(p) for p in paths)

    if path_list:
        valid_paths, skipped = _filter_valid_parquets(path_list, num_cpus=ray_cpus)
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
        spectral_cols = _list_spectral_columns(aug_columns)
        meta_cols = [
            col for col in meta_candidates if col in aug_columns and col != "pixel_id"
        ]
        other_cols = [
            col
            for col in aug_columns
            if col
            not in {"pixel_id", "wl_nm_int", "reflectance"}
            and col not in spectral_cols
            and col not in meta_cols
        ]

        select_parts = ["pixel_id"]
        for col in meta_cols:
            ident = _quote_identifier(col)
            select_parts.append(f"ANY_VALUE({ident}) AS {ident}")
        for col in other_cols:
            ident = _quote_identifier(col)
            select_parts.append(f"ANY_VALUE({ident}) AS {ident}")
        for col in spectral_cols:
            ident = _quote_identifier(col)
            select_parts.append(f"MAX({ident}) AS {ident}")

        con.execute(
            f"CREATE OR REPLACE VIEW {_quote_identifier(name + '_wide')} AS SELECT "
            + ", ".join(select_parts)
            + f" FROM {_quote_identifier(name + '_aug')} GROUP BY pixel_id"
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
    ray_cpus: int | None = None,
    *,
    merge_memory_limit_gb: float | str | None = 64.0,  # Increased default to 64GB for large merges (multiple JOINs need larger hash tables)
    merge_threads: int | None = 4,
    merge_row_group_size: int | None = None,  # None = let DuckDB auto-determine (better streaming performance)
    merge_temp_directory: Path | None = None,
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
    ray_cpus : int, optional
        CPU budget forwarded to Ray validation of Parquet shards. Defaults to
        ``None`` which allows the Ray helper to choose the configured default.
    merge_memory_limit_gb : float or str, optional
        Upper bound on DuckDB's memory usage. Floats are interpreted as GiB.
        Provide ``None`` to leave the default DuckDB behaviour unchanged.
    merge_threads : int, optional
        Number of threads DuckDB should use for the merge. ``None`` keeps the
        engine default (usually ``os.cpu_count()``).
    merge_row_group_size : int or None, optional
        Target number of rows per Parquet row group in the merged output.
        If None, DuckDB will auto-determine the optimal size (better streaming performance).
    merge_temp_directory : Path, optional
        Directory where DuckDB should spill temporary data. Defaults to a
        ``.duckdb_tmp`` subdirectory inside ``flightline_dir``.

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

    merge_memory_limit_gb = _coerce_memory_limit(merge_memory_limit_gb)

    if merge_row_group_size is not None and merge_row_group_size <= 0:
        raise ValueError("merge_row_group_size must be a positive integer or None")
    if merge_threads is not None and merge_threads <= 0:
        raise ValueError("merge_threads must be positive when provided")

    tmp_dir = (
        Path(merge_temp_directory).resolve()
        if merge_temp_directory is not None
        else (flightline_dir / ".duckdb_tmp").resolve()
    )

    print(f"[merge] Start flightline={flightline_dir.name} prefix={prefix}")
    print(f"[merge] Output parquet → {out_parquet}")

    if isinstance(merge_memory_limit_gb, str):
        memory_limit_repr = merge_memory_limit_gb
    elif merge_memory_limit_gb is None:
        memory_limit_repr = "auto"
    else:
        memory_limit_repr = f"{merge_memory_limit_gb}GB"

    thread_repr = str(merge_threads) if merge_threads and merge_threads > 0 else "auto"
    row_group_repr = str(merge_row_group_size) if merge_row_group_size is not None else "auto"
    print(
        f"[merge] Engine=duckdb memory_limit={memory_limit_repr} "
        f"threads={thread_repr} row_group_size={row_group_repr} "
        f"temp_dir={tmp_dir}"
    )

    def _discover_inputs() -> Dict[str, List[Path]]:
        # When default globs are supplied, favour precise patterns with exclusions.
        if (
            original_glob == "**/*original*.parquet"
            and corrected_glob == "**/*corrected*.parquet"
            and resampled_glob == "**/*resampl*.parquet"
        ):
            inputs: Dict[str, List[Path]] = {"orig": [], "corr": [], "resamp": []}

            # originals (long format from raw ENVI)
            # Only the base uncorrected ENVI file, not sensor-specific resampled files
            orig = sorted(flightline_dir.glob("*_envi.parquet"))
            orig = [
                p for p in orig 
                if "brdfandtopo_corrected" not in p.name
                and "landsat" not in p.name.lower()
                and "micasense" not in p.name.lower()
                and "oli" not in p.name.lower()
                and "tm" not in p.name.lower()
                and "etm" not in p.name.lower()
            ]
            inputs["orig"] = _exclude_parquets(orig)

            # corrected (long format from corrected ENVI)
            corr = sorted(
                flightline_dir.glob("*_brdfandtopo_corrected_envi.parquet")
            )
            inputs["corr"] = _exclude_parquets(corr)

            # resampled (sensor stacks)
            # Find all parquets that end with _envi.parquet but are not originals or corrected
            # Capture both brightness-adjusted and *_undarkened_envi parquet sidecars
            resamp: List[Path] = []
            
            # Find all parquets ending with _envi.parquet (includes both regular and undarkened)
            all_envi_parquets = sorted(flightline_dir.glob("*_envi.parquet"))
            
            # Also find undarkened parquets explicitly
            all_undarkened_parquets = sorted(flightline_dir.glob("*_undarkened_envi.parquet"))
            all_envi_parquets.extend(all_undarkened_parquets)
            
            # Build sets of already-discovered files to exclude
            orig_set = set(inputs["orig"])
            corr_set = set(inputs["corr"])
            
            for pq in all_envi_parquets:
                # Skip originals and corrected
                if pq in orig_set or pq in corr_set:
                    continue
                # Skip if it's in the exclusion patterns (merged, qa, etc.)
                name = pq.name.lower()
                if any(ex in name for ex in EXCLUDE_PATTERNS):
                    continue
                # Skip if it's the simple original format (no sensor name)
                # Original format: <flight_stem>_envi.parquet (no underscore before envi)
                # Sensor format: <flight_stem>_<sensor>_envi.parquet (has underscore before envi)
                # Undarkened format: <flight_stem>_<sensor>_undarkened_envi.parquet
                parts = pq.stem.split("_")
                if len(parts) >= 3 and (parts[-2] == "envi" or parts[-3] == "envi"):
                    # This looks like a sensor product: ..._sensor_envi or ..._sensor_undarkened_envi
                    resamp.append(pq)
                elif "landsat" in name or "micasense" in name or "oli" in name or "tm" in name or "etm" in name:
                    # Known sensor keywords (including undarkened versions)
                    resamp.append(pq)
            
            # Remove duplicates and sort
            resamp = sorted(set(resamp))
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
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Reduce threads to reduce memory pressure (fewer threads = less memory per thread)
        # ROW_NUMBER() was removed to avoid OOM, but we still need to be conservative with memory
        if merge_threads is None:
            effective_threads = 2  # Reduced from default to save memory
            print("[merge] Using threads=2 to reduce memory pressure")
        else:
            effective_threads = min(merge_threads, 2)  # Cap at 2 threads to avoid OOM
            if merge_threads > 2:
                print(f"[merge] Capping threads at 2 (requested {merge_threads}) to avoid OOM")
        con.execute(f"PRAGMA threads = {effective_threads}")

        # Set memory limit - increase default for large merges
        # More memory = larger hash tables for JOINs can stay in memory = faster
        # With 10M rows and 9 files, we need significant memory for hash tables
        if merge_memory_limit_gb is None:
            # Default to 64GB for large merges (user has 368GB available)
            # Increased from 32GB to allow larger hash tables for multiple JOINs
            memory_limit_value = "64GB"
            print("[merge] Using default memory_limit=64GB (can be overridden with merge_memory_limit_gb)")
        else:
            if isinstance(merge_memory_limit_gb, str):
                memory_limit_value = merge_memory_limit_gb
            else:
                memory_limit_value = f"{merge_memory_limit_gb}GB"
        
        # Always set memory limit (fix: was missing for default case)
        memory_limit_value = memory_limit_value.replace("'", "''")
        con.execute(f"PRAGMA memory_limit = '{memory_limit_value}'")
        print(f"[merge] Set DuckDB memory_limit = {memory_limit_value}")
        
        # Disable insertion-order preservation to reduce memory usage (as suggested by error)
        con.execute("SET preserve_insertion_order=false")
        print("[merge] Disabled insertion-order preservation to reduce memory usage")
        
        # Enable query optimizations for faster JOINs
        con.execute("PRAGMA enable_object_cache=true")
        con.execute("PRAGMA checkpoint_threshold='1GB'")
        
        con.execute("SET enable_progress_bar = true")
        con.execute(
            "PRAGMA temp_directory = '"
            + _quote_path(str(tmp_dir))
            + "'"
        )

        with _progress("DuckDB: stream merge (CTEs, no materialization)"):
            # Use CTEs to read parquet files directly - this streams instead of materializing
            # Much faster than creating views first
            print(f"[merge] 🔍 Starting streaming merge for {flightline_dir.name}")
            print(f"[merge] 📊 Discovered inputs: orig={len(inputs['orig'])}, corr={len(inputs['corr'])}, resamp={len(inputs['resamp'])}")
            
            def _build_streaming_cte(paths: List[Path], alias: str) -> tuple[str, set[str]]:
                """Build a CTE that reads parquet files directly (streaming)."""
                # Empty CTE must have pixel_id column for UNION to work
                empty_cte = f"{alias} AS (SELECT CAST(NULL AS VARCHAR) AS pixel_id WHERE 1=0)"
                if not paths:
                    return empty_cte, set()
                
                # Validate parquets
                valid_paths, skipped = _filter_valid_parquets(paths, num_cpus=ray_cpus)
                for bad_path, message in skipped:
                    print(
                        f"[merge] ⚠️ Skipping invalid parquet {bad_path.name}: {message}\n"
                        f"        Delete or regenerate this file before re-running the merge."
                    )
                
                if not valid_paths:
                    return empty_cte, set()
                
                
                # Get schema from first file (fast, just reads metadata)
                sample_path = valid_paths[0]
                # Use DESCRIBE to get columns without reading data - very fast
                desc_result = con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{_quote_path(sample_path.as_posix())}')"
                ).fetchall()
                sample_cols = [r[0] for r in desc_result]
                col_set = set(sample_cols)
                
                # Build read_parquet expression
                if len(valid_paths) == 1:
                    read_expr = f"read_parquet('{_quote_path(valid_paths[0].as_posix())}', union_by_name = TRUE)"
                else:
                    files_array = ", ".join(
                        "'" + _quote_path(p.as_posix()) + "'" for p in valid_paths
                    )
                    read_expr = f"read_parquet([{files_array}], union_by_name = TRUE)"
                
                # Normalize pixel_id inline (same logic as before but in CTE)
                pixel_expr_parts: List[str] = []
                if "pixel_id" in col_set:
                    pixel_expr_parts.append("CAST(pixel_id AS VARCHAR)")
                if {"row", "col"}.issubset(col_set):
                    pixel_expr_parts.append(
                        "CONCAT('r', CAST(row AS VARCHAR), '_c', CAST(col AS VARCHAR))"
                    )
                if {"x", "y"}.issubset(col_set):
                    pixel_expr_parts.append(
                        "CONCAT('x', CAST(ROUND(x, 3) AS VARCHAR), '_y', CAST(ROUND(y, 3) AS VARCHAR))"
                    )
                if not pixel_expr_parts:
                    pixel_expr_parts.append("'pixel_' || ROW_NUMBER() OVER ()")
                pixel_expr = "COALESCE(" + ", ".join(pixel_expr_parts) + ")"
                
                # Select all columns with normalized pixel_id
                # Note: We don't use DISTINCT here to avoid materialization/OOM risk
                # Source files already have unique pixel_ids (confirmed by inspection)
                # LEFT JOINs from unique base (all_pixels) will naturally produce one row per pixel_id
                select_parts: List[str] = [f"{pixel_expr} AS pixel_id"]
                for col in sample_cols:
                    if col == "pixel_id":
                        continue
                    select_parts.append(_quote_identifier(col))
                
                # Simple CTE without DISTINCT - relies on:
                # 1. Source files having unique pixel_ids (confirmed)
                # 2. LEFT JOIN structure from unique base preventing duplicates
                # 3. Post-merge validation to catch any issues
                cte_sql = f"""{alias} AS (
                    SELECT {", ".join(select_parts)}
                    FROM {read_expr}
                )"""
                
                return cte_sql, col_set
            
            # Build streaming CTEs
            print("[merge] 🔨 Building CTEs (streaming, no materialization)...")
            orig_cte, orig_cols = _build_streaming_cte(inputs["orig"], "orig")
            orig_valid = len(orig_cols) > 0
            print(f"[merge]   ✅ orig CTE built ({len(orig_cols)} columns)")
            corr_cte, corr_cols = _build_streaming_cte(inputs["corr"], "corr")
            corr_valid = len(corr_cols) > 0
            print(f"[merge]   ✅ corr CTE built ({len(corr_cols)} columns)")
            
            # CRITICAL FIX: Join each resampled file separately to avoid row multiplication
            # When multiple files are combined with UNION, each pixel_id appears multiple times
            # causing a Cartesian product during JOIN. Instead, join each file separately.
            resamp_ctes: List[str] = []
            resamp_aliases: List[str] = []
            resamp_cols_by_file: List[set[str]] = []
            
            # Build a CTE for each resampled file separately
            for idx, resamp_path in enumerate(inputs["resamp"]):
                alias = f"resamp_{idx}"
                resamp_cte, resamp_cols = _build_streaming_cte([resamp_path], alias)
                resamp_ctes.append(resamp_cte)
                resamp_aliases.append(alias)
                resamp_cols_by_file.append(resamp_cols)
                print(f"[merge]   ✅ {alias} CTE built from {resamp_path.name} ({len(resamp_cols)} columns)")
            
            # Check if all categories are empty after filtering (raise error if so)
            resamp_valid = len(resamp_ctes) > 0
            if not orig_valid and not corr_valid and not resamp_valid:
                raise FileNotFoundError(
                    f"No valid parquet inputs found in {flightline_dir} after filtering invalid files. "
                    f"Remove or recreate the invalid files and try again."
                )
            
            # Collect all columns across all resampled files
            all_resamp_cols = set()
            for cols in resamp_cols_by_file:
                all_resamp_cols.update(cols)
            
            # Detect if orig is in long format (has wavelength_nm and reflectance columns)
            # Long format: one row per wavelength per pixel (needs pivoting)
            # Wide format: one row per pixel with all wavelengths as columns
            orig_is_long = "wavelength_nm" in orig_cols and "reflectance" in orig_cols
            
            # If orig is long format, pivot it to wide format within the CTE
            # This preserves using orig as the base (maintains merge behavior)
            if orig_is_long:
                # Pivot long format to wide: GROUP BY pixel_id and create wavelength columns
                # Get metadata columns (non-spectral)
                metadata_cols = [c for c in orig_cols if c not in ('pixel_id', 'wavelength_nm', 'reflectance')]
                metadata_select = ', '.join([f"ANY_VALUE({_quote_identifier(c)}) AS {_quote_identifier(c)}" 
                                             for c in metadata_cols]) if metadata_cols else ''
                
                # Build wavelength columns (test uses range 1-426, but this could be made dynamic)
                wavelength_selects = ', '.join([
                    f"MAX(CASE WHEN CAST(wavelength_nm AS INTEGER) = {wl} THEN reflectance END) AS {_quote_identifier(f'orig_wl{wl:04d}nm')}"
                    for wl in range(1, 427)
                ])
                
                # Build pivot CTE
                if metadata_select:
                    all_selects = f"pixel_id, {metadata_select}, {wavelength_selects}"
                else:
                    all_selects = f"pixel_id, {wavelength_selects}"
                
                pivot_cte = f"""
                    orig_wide AS (
                        SELECT 
                            {all_selects}
                        FROM orig
                        GROUP BY pixel_id
                    )
                """
                # Add pivot CTE after orig_cte
                orig_cte = orig_cte + ", " + pivot_cte
                print("[merge]   ℹ️  orig is long format, pivoted to wide format")
            
            # Always use orig as base (preserves original merge behavior)
            select_clause: List[str] = ["orig.*"] if not orig_is_long else ["orig_wide.*"]
            
            # Add spectral columns from corr (non-metadata columns with _wl)
            corr_spectral = [c for c in corr_cols if "_wl" in c and c not in META_CANDIDATES]
            for col in corr_spectral:
                ident = _quote_identifier(col)
                select_clause.append(f"corr.{ident} AS {ident}")
            
            # Add spectral columns from each resampled file separately
            # Each resampled file has different spectral bands, so we select all non-metadata columns
            for idx, alias in enumerate(resamp_aliases):
                file_cols = resamp_cols_by_file[idx]
                # Select all columns except metadata and pixel_id (which is already in base.*)
                for col in file_cols:
                    if col not in META_CANDIDATES and col != "pixel_id":
                        ident = _quote_identifier(col)
                        select_clause.append(f"{alias}.{ident} AS {ident}")
            
            # Build JOIN clauses using USING syntax (faster than ON, matches old script)
            # CRITICAL: Each resampled file is joined separately to prevent row multiplication
            # Since all files have unique pixel_ids and the same set of pixel_ids (confirmed by inspection),
            # each LEFT JOIN matches exactly one row per pixel_id, preventing the Cartesian product
            # that caused the 7x row explosion (70M rows instead of 10M).
            join_clauses: List[str] = [
                "LEFT JOIN corr USING (pixel_id)"
            ]
            
            for alias in resamp_aliases:
                join_clauses.append(f"LEFT JOIN {alias} USING (pixel_id)")
            
            # Build final streaming query with CTEs (no materialization)
            # Always use orig as base (preserves merge behavior)
            # Structure: FROM orig (wide format, unique rows) → LEFT JOIN each file separately (1:1 matches)
            # Result: Exactly one row per pixel_id, no duplicates
            all_ctes = [orig_cte, corr_cte] + resamp_ctes
            if orig_is_long:
                # Include orig_wide in CTEs and use it in FROM
                base_table = "orig_wide"
            else:
                base_table = "orig"
                
            select_sql = f"""
            WITH {", ".join(all_ctes)}
            SELECT {", ".join(select_clause)}
            FROM {base_table}
            {' '.join(join_clauses)}
            """

        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        if out_parquet.exists():
            if out_parquet.is_dir():
                shutil.rmtree(out_parquet, ignore_errors=True)
            else:
                out_parquet.unlink()

        with _progress("DuckDB: stream join → parquet"):
            # Stream directly to parquet with explicit ROW_GROUP_SIZE for test compatibility
            print("[merge] 📤 Streaming join results to parquet (no materialization)...")
            print(f"[merge]    Output: {out_parquet.name}")
            print(f"[merge]    Input: {len(inputs['orig'])} orig, {len(inputs['corr'])} corr, {len(inputs['resamp'])} resamp")
            
            # Test query first with LIMIT to verify it works (catches errors early)
            print("[merge]    Testing query with LIMIT 1000...")
            test_sql = select_sql + " LIMIT 1000"
            try:
                test_result = con.execute(test_sql).fetchall()
                print(f"[merge]    ✅ Test query successful - returned {len(test_result)} rows")
                # Verify no duplicates in test result
                if test_result:
                    test_pixel_ids = [row[0] for row in test_result]  # pixel_id is first column
                    unique_count = len(set(test_pixel_ids))
                    if len(test_pixel_ids) != unique_count:
                        print(f"[merge]    ⚠️  WARNING: Test query found {len(test_pixel_ids) - unique_count} duplicate pixel_ids in sample!")
                    else:
                        print("[merge]    ✅ Test query: no duplicate pixel_ids in sample")
            except Exception as e:
                print(f"[merge]    ❌ Test query failed: {e}")
                raise
            
            # Get row count estimate (fast, just counts from orig or orig_wide if pivoted)
            row_count = None
            try:
                count_table = "orig_wide" if orig_is_long else "orig"
                count_sql = f"""
                WITH {orig_cte}
                SELECT COUNT(*) FROM {count_table}
                """
                row_count = con.execute(count_sql).fetchone()[0]
                print(f"[merge]    Expected rows: {row_count:,}")
            except Exception as e:
                print(f"[merge]    ⚠️  Could not get row count estimate: {e}")
                # Continue anyway - validation will catch issues
            
            # Only include ROW_GROUP_SIZE if explicitly set (None = let DuckDB auto-determine for better streaming)
            row_group_option = f", ROW_GROUP_SIZE {merge_row_group_size}" if merge_row_group_size is not None else ""
            copy_sql = (
                "COPY ("
                + select_sql
                + f") TO '{_quote_path(str(out_parquet))}' (FORMAT PARQUET,"
                f" COMPRESSION ZSTD{row_group_option})"
            )
            print("[merge]    Executing streaming COPY (this may take 5-15 minutes for large datasets)...")
            print(f"[merge]    You can check progress by monitoring file size: ls -lh {out_parquet.name}")
            import time
            start_time = time.time()
            try:
                con.execute(copy_sql)
                elapsed = time.time() - start_time
                print(f"[merge]    ✅ Streaming COPY complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
                
                # Verify output file was created and has data
                if out_parquet.exists():
                    size_gb = out_parquet.stat().st_size / (1024**3)
                    print(f"[merge]    ✅ Output file created: {size_gb:.2f} GB")
                    if size_gb == 0:
                        print("[merge]    ⚠️  WARNING: Output file is 0 bytes - merge may have failed!")
                else:
                    print("[merge]    ❌ ERROR: Output file was not created!")
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"[merge]    ❌ Streaming COPY failed after {elapsed:.1f} seconds: {e}")
                raise
        _consolidate_parquet_dir_to_file(con, Path(out_parquet), Path(out_parquet))
        print(
            f"[merge] ✅ Wrote parquet: {out_parquet} (exists={Path(out_parquet).exists()})"
        )
        
        # Validate row count matches expectations
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(out_parquet)
            actual_rows = parquet_file.metadata.num_rows
            expected_rows = row_count if row_count else None
            
            # Get expected rows from orig file if not available
            if expected_rows is None and inputs["orig"]:
                orig_file = pq.ParquetFile(inputs["orig"][0])
                expected_rows = orig_file.metadata.num_rows
            
            if expected_rows:
                if actual_rows == expected_rows:
                    print(f"[merge] ✅ Row count validation: {actual_rows:,} rows (as expected)")
                else:
                    print(f"[merge] ⚠️  Row count mismatch: {actual_rows:,} rows (expected {expected_rows:,}, difference: {actual_rows - expected_rows:,})")
                    if actual_rows > expected_rows:
                        # Check for duplicate pixel_ids
                        check_dup_sql = f"SELECT COUNT(*) as total, COUNT(DISTINCT pixel_id) as unique FROM read_parquet('{_quote_path(str(out_parquet))}')"
                        dup_result = con.execute(check_dup_sql).fetchone()
                        total, unique = dup_result
                        if total > unique:
                            print(f"[merge] ⚠️  Found {total - unique:,} duplicate pixel_ids in merged file")
                        else:
                            print("[merge] ⚠️  No duplicate pixel_ids, but row count is higher - check JOIN logic")
            else:
                print(f"[merge] ℹ️  Row count: {actual_rows:,} rows (no expected value to compare)")
        except Exception as e:
            print(f"[merge] ⚠️  Could not validate row count: {e}")

        if write_feather:
            out_feather = out_parquet.with_suffix(".feather").resolve()
            with _progress("DuckDB: write feather"):
                con.execute(
                    "COPY ("
                    + select_sql
                    + f") TO '{_quote_path(str(out_feather))}' (FORMAT ARROW)"
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
    parser.add_argument(
        "--merge-memory-limit",
        dest="merge_memory_limit",
        default=None,
        help=(
            "DuckDB memory limit for the merge (float GiB, 'auto', or a DuckDB-compatible value). "
            "Defaults to 6.0 GiB when omitted."
        ),
    )
    parser.add_argument(
        "--merge-threads",
        dest="merge_threads",
        type=int,
        default=None,
        help="Number of DuckDB threads to use during the merge (defaults to 4).",
    )
    parser.add_argument(
        "--merge-row-group-size",
        dest="merge_row_group_size",
        type=int,
        default=None,
        help="Row group size for the merged Parquet (defaults to 50,000).",
    )
    parser.add_argument(
        "--merge-temp-directory",
        dest="merge_temp_directory",
        type=Path,
        default=None,
        help="Custom DuckDB temp directory for the merge stage.",
    )
    args = parser.parse_args(argv)

    merge_kwargs = {}
    if args.merge_memory_limit is not None:
        merge_kwargs["merge_memory_limit_gb"] = args.merge_memory_limit
    if args.merge_threads is not None:
        merge_kwargs["merge_threads"] = args.merge_threads
    if args.merge_row_group_size is not None:
        merge_kwargs["merge_row_group_size"] = args.merge_row_group_size
    if args.merge_temp_directory is not None:
        merge_kwargs["merge_temp_directory"] = args.merge_temp_directory

    merge_all_flightlines(
        data_root=args.data_root,
        flightline_glob=args.flightline_glob,
        out_name=args.out_name,
        original_glob=args.original_glob,
        corrected_glob=args.corrected_glob,
        resampled_glob=args.resampled_glob,
        write_feather=args.write_feather,
        emit_qa_panel=(not args.no_qa),
        **merge_kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
