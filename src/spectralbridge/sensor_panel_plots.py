"""Generate multi-panel sensor comparison scatter plots from merged Parquet tables.

This module promotes the plotting helpers that previously lived only inside the
``deprecated/pixel_extraction`` notebooks. The functions exposed here replicate
the layout and styling of the legacy figures so that the modern processing
pipeline can emit the same QA artifacts automatically.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Shared helpers lifted from the deprecated notebook implementation
# ---------------------------------------------------------------------------

_BAND_LABEL_RE = re.compile(r"^(?P<label>.+)_band_(?P<idx>\d+)$")
_CORRECTED_CANDIDATES: Tuple[str, ...] = (
    "NEON_BRDFTopo",
    "NEON_directional_BRDFTopo",
)

DEFAULT_EXPECTED_SETS: Dict[str, int] = {
    "Landsat_5_TM": 6,
    "Landsat_7_ETM+": 6,
    "Landsat_8_OLI": 7,
    "Landsat_9_OLI-2": 7,
    "MicaSense": 10,
    "MicaSense_to-match_OLI_and_OLI-2": 5,
    "MicaSense_to-match_TM_and_ETM+": 4,
}


def _load_json_resource(name: str) -> dict:
    with resources.files("spectralbridge.data").joinpath(name).open("r") as f:
        import json

        return json.load(f)


_HYPERSPEC_BANDS: List[float] = list(
    _load_json_resource("hyperspectral_bands.json")["bands"]
)
_SENSOR_PARAMS: dict = _load_json_resource("landsat_band_parameters.json")
_PARAM_KEY_ALIASES = {
    "Landsat_5_TM": "Landsat 5 TM",
    "Landsat_7_ETM+": "Landsat 7 ETM+",
    "Landsat_8_OLI": "Landsat 8 OLI",
    "Landsat_9_OLI-2": "Landsat 9 OLI-2",
    "MicaSense": "MicaSense",
    "MicaSense_to-match_TM_and_ETM+": "MicaSense-to-match TM and ETM+",
    "MicaSense_to-match_OLI_and_OLI-2": "MicaSense-to-match OLI and OLI-2",
}


def _sensor_band_text(sensor_label: str, band_idx1: int) -> str:
    key = _PARAM_KEY_ALIASES.get(sensor_label)
    if key and key in _SENSOR_PARAMS:
        wl = _SENSOR_PARAMS[key].get("wavelengths", [])
        fw = _SENSOR_PARAMS[key].get("fwhms", [])
        if 1 <= band_idx1 <= len(wl):
            fwhm = fw[band_idx1 - 1] if band_idx1 - 1 < len(fw) else None
            if fwhm is not None:
                return f"{sensor_label} B{band_idx1} ({wl[band_idx1-1]:g} nm ±{fwhm:g})"
            return f"{sensor_label} B{band_idx1} ({wl[band_idx1-1]:g} nm)"
    return f"{sensor_label} band {band_idx1}"


def _corrected_band_text(corrected_label: str, band_idx1: int) -> str:
    if 1 <= band_idx1 <= len(_HYPERSPEC_BANDS):
        return f"{corrected_label} {band_idx1} ({_HYPERSPEC_BANDS[band_idx1-1]:.1f} nm)"
    return f"{corrected_label} band {band_idx1}"


def _read_schema(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> List[str]:
    try:
        rows = con.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        pass

    try:
        sch = con.sql(
            f"SELECT name, type FROM parquet_schema('{parquet_path.as_posix()}')"
        ).fetchall()
        flat_cols = [
            name
            for name, typ in sch
            if isinstance(name, str)
            and isinstance(typ, str)
            and not any(
                t in typ.upper() for t in ("LIST", "STRUCT", "MAP", "UNION", "DICTIONARY")
            )
        ]
        if flat_cols:
            return flat_cols
    except Exception:
        pass

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path.as_posix())
        schema = pf.schema_arrow

        def _is_flat_numeric(t: "pa.DataType") -> bool:
            return pa.types.is_integer(t) or pa.types.is_floating(t)

        cols = [f.name for f in schema if _is_flat_numeric(f.type)]
        if cols:
            return cols
    except Exception:
        pass

    return []


def _band_map(columns: Iterable[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for c in columns:
        m = _BAND_LABEL_RE.match(c)
        if m:
            out.setdefault(m.group("label"), []).append(int(m.group("idx")))
    for k in out:
        out[k] = sorted(out[k])
    return out


def _pick_corrected_label(bandmap: Dict[str, List[int]], need: int = 426) -> Optional[str]:
    for cand in _CORRECTED_CANDIDATES:
        if cand in bandmap and len(bandmap[cand]) >= need:
            return cand
    return None


def _possible_error_cols(label: str, idx: int) -> List[str]:
    suffix = str(idx)
    return [
        f"{label}_band_{suffix}_error",
        f"{label}_band_{suffix}_err",
        f"{label}_error_band_{suffix}",
        f"{label}_err_band_{suffix}",
    ]


def _present_error_cols(all_cols: Sequence[str], label: str, idx: int) -> List[str]:
    return [c for c in _possible_error_cols(label, idx) if c in all_cols]


def _palette_fraction_by_column(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    colnames: Sequence[str],
    lo: int = 1,
    hi: int = 255,
) -> Dict[str, float]:
    if not colnames:
        return {}
    terms = [
        f'SUM((("{c}" BETWEEN {lo} AND {hi}) AND "{c}" = floor("{c}")))*1.0 / NULLIF(COUNT("{c}"),0) AS "{c}"'
        for c in colnames
    ]
    sql = f"SELECT {', '.join(terms)} FROM read_parquet('{parquet_path.as_posix()}')"
    try:
        row = con.sql(sql).fetchone()
    except Exception:
        return {}
    if row is None:
        return {}
    return {col: float(val) if val is not None else 0.0 for col, val in zip(colnames, row)}


def _sample_pair_df(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    xcol: str,
    ycol: str,
    max_points: int,
    sentinel_at_or_below: float,
    extra_zero_filters: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    extra = ""
    if extra_zero_filters:
        extra = " AND " + " AND ".join(f'"{c}" = 0' for c in extra_zero_filters)

    base = f"""
    WITH base AS (
      SELECT "{xcol}" AS x, "{ycol}" AS y
      FROM read_parquet('{parquet_path.as_posix()}')
      WHERE
        isfinite("{xcol}") AND isfinite("{ycol}")
        AND "{xcol}" > {sentinel_at_or_below} AND "{ycol}" > {sentinel_at_or_below}
        {extra}
    )
    """
    q_sample = base + f"SELECT * FROM base USING SAMPLE {max_points} ROWS"
    try:
        df = con.sql(q_sample).df()
    except Exception:
        q_fallback = base + f"SELECT * FROM base ORDER BY random() LIMIT {max_points}"
        df = con.sql(q_fallback).df()
    return df[(df["x"] >= 0) & (df["y"] >= 0)]


def _linreg(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def _regress_and_plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> float:
    ax.plot(x, y, ".", markersize=0.8, alpha=0.3)
    if x.size >= 2 and y.size >= 2:
        mn = float(np.nanmin(np.column_stack([x, y])))
        mx = float(np.nanmax(np.column_stack([x, y])))
        ax.plot([mn, mx], [mn, mx], linewidth=1.0, alpha=0.5, color="gray")

        m, b = _linreg(x, y)
        if np.isfinite(m) and np.isfinite(b):
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 2)
            yy = m * xx + b
            ax.plot(xx, yy, "--", linewidth=1.0, alpha=0.9, color="red")
            ss_res = float(np.nansum((y - (m * x + b)) ** 2))
            ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            ax.text(
                0.02,
                0.98,
                f"y = {m:.3f}x + {b:.3f}\nR² = {r2:.3f}   n={x.size:,}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
            )
    try:
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return float("nan")


def _iter_merged_parquets(merged_dir: Path) -> List[Path]:
    merged_dir = Path(merged_dir)
    if not merged_dir.exists():
        return []
    return sorted(
        [
            p
            for p in merged_dir.glob("*.parquet")
            if "merged" in p.name and not p.name.endswith(".tmp.parquet")
        ]
    )


def make_sensor_vs_neon_panels(
    merged_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    max_points: int = 50_000,
    sentinel_at_or_below: float = -9999.0,
) -> List[Path]:
    merged_dir = Path(merged_dir)
    out_path = Path(out_dir) if out_dir is not None else merged_dir
    out_path.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    outputs: List[Path] = []

    try:
        for parquet in _iter_merged_parquets(merged_dir):
            cols = _read_schema(con, parquet)
            if not cols:
                continue

            bandmap = _band_map(cols)
            corrected = _pick_corrected_label(bandmap)
            if corrected is None:
                continue

            ordered_known = [
                lab
                for lab in DEFAULT_EXPECTED_SETS
                if lab in bandmap and lab != corrected
            ]
            extras = [
                lab
                for lab in sorted(bandmap)
                if lab not in DEFAULT_EXPECTED_SETS and lab != corrected
            ]
            sensors = ordered_known + extras
            if not sensors:
                continue

            need_cols = sorted(
                {
                    f"{corrected}_band_{idx}"
                    for lab in sensors
                    for idx in bandmap[lab]
                }
                | {
                    f"{lab}_band_{idx}"
                    for lab in sensors
                    for idx in bandmap[lab]
                }
            )

            pal_frac = _palette_fraction_by_column(con, parquet, need_cols)

            sensor_pairs: Dict[str, List[Tuple[int, str, str]]] = {}
            for lab in sensors:
                pairs: List[Tuple[int, str, str]] = []
                for idx in bandmap[lab]:
                    xcol = f"{corrected}_band_{idx}"
                    ycol = f"{lab}_band_{idx}"
                    if pal_frac.get(xcol, 0.0) >= 0.95 or pal_frac.get(ycol, 0.0) >= 0.95:
                        continue
                    pairs.append((idx, xcol, ycol))
                if pairs:
                    sensor_pairs[lab] = pairs

            if not sensor_pairs:
                continue

            sensors_in_panel = [lab for lab in sensors if lab in sensor_pairs]
            n_rows = len(sensors_in_panel)
            n_cols = max(len(sensor_pairs[lab]) for lab in sensors_in_panel)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4.0 * n_cols, 3.6 * n_rows),
                squeeze=False,
            )

            for r, lab in enumerate(sensors_in_panel):
                pairs = sensor_pairs[lab]
                for c in range(n_cols):
                    ax = axes[r][c]
                    if c >= len(pairs):
                        ax.axis("off")
                        continue
                    idx, xcol, ycol = pairs[c]
                    err_cols = _present_error_cols(cols, corrected, idx) + _present_error_cols(
                        cols, lab, idx
                    )
                    df = _sample_pair_df(
                        con,
                        parquet,
                        xcol,
                        ycol,
                        max_points,
                        sentinel_at_or_below,
                        extra_zero_filters=err_cols if err_cols else None,
                    )
                    if df.empty:
                        ax.set_title(
                            f"{_sensor_band_text(lab, idx)} vs {_corrected_band_text(corrected, idx)}",
                            fontsize=9,
                        )
                        ax.set_xlabel(_corrected_band_text(corrected, idx), fontsize=8)
                        if c == 0:
                            ax.set_ylabel(_sensor_band_text(lab, idx), fontsize=8)
                        else:
                            ax.set_ylabel("")
                        continue

                    r_val = _regress_and_plot(ax, df["x"].to_numpy(), df["y"].to_numpy())
                    sensor_text = _sensor_band_text(lab, idx)
                    corrected_text = _corrected_band_text(corrected, idx)
                    ax.set_title(
                        f"{sensor_text} vs {corrected_text}  r={r_val:.2f}",
                        fontsize=9,
                    )
                    ax.set_xlabel(corrected_text, fontsize=8)
                    if c == 0:
                        ax.set_ylabel(sensor_text, fontsize=8)
                    else:
                        ax.set_ylabel("")

            fig.suptitle(
                f"{parquet.stem} — Sensors by row; bands left→right vs {corrected}",
                y=0.995,
                fontsize=12,
            )
            fig.tight_layout()

            suffix = f"__BY_SENSOR_vs_{corrected}.png"
            out_png = out_path / f"{parquet.stem}{suffix}"
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out_png)
    finally:
        con.close()

    return outputs


_MS_LANDSAT_PAIRS: Dict[str, Sequence[str]] = {
    "MicaSense_to-match_TM_and_ETM+": ("Landsat_5_TM", "Landsat_7_ETM+"),
    "MicaSense_to-match_OLI_and_OLI-2": ("Landsat_8_OLI", "Landsat_9_OLI-2"),
}


def _collect_ms_ls_pairs(
    bandmap: Dict[str, List[int]]
) -> Dict[Tuple[str, str], List[int]]:
    out: Dict[Tuple[str, str], List[int]] = {}
    for ms_label, ls_labels in _MS_LANDSAT_PAIRS.items():
        if ms_label not in bandmap:
            continue
        ms_bands = bandmap[ms_label]
        for ls_label in ls_labels:
            if ls_label not in bandmap:
                continue
            ls_bands = bandmap[ls_label]
            common = sorted(set(ms_bands) & set(ls_bands))
            if common:
                out[(ms_label, ls_label)] = common
    return out


def make_micasense_vs_landsat_panels(
    merged_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    max_points: int = 50_000,
    sentinel_at_or_below: float = -9999.0,
) -> List[Path]:
    merged_dir = Path(merged_dir)
    out_path = Path(out_dir) if out_dir is not None else merged_dir
    out_path.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    outputs: List[Path] = []

    try:
        for parquet in _iter_merged_parquets(merged_dir):
            cols = _read_schema(con, parquet)
            if not cols:
                continue
            bandmap = _band_map(cols)
            ms_pairs = _collect_ms_ls_pairs(bandmap)
            if not ms_pairs:
                continue

            ordered_pairs: List[Tuple[str, str]] = []
            for ms_label, ls_labels in _MS_LANDSAT_PAIRS.items():
                for ls_label in ls_labels:
                    if (ms_label, ls_label) in ms_pairs:
                        ordered_pairs.append((ms_label, ls_label))
            if not ordered_pairs:
                continue

            need_cols = sorted(
                {
                    f"{label}_band_{idx}"
                    for (ms_label, ls_label), indices in ms_pairs.items()
                    for idx in indices
                    for label in (ms_label, ls_label)
                }
            )
            pal_frac = _palette_fraction_by_column(con, parquet, need_cols)

            n_rows = len(ordered_pairs)
            n_cols = max(len(ms_pairs[pair]) for pair in ordered_pairs)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4.0 * n_cols, 3.6 * n_rows),
                squeeze=False,
            )

            for r, (ms_label, ls_label) in enumerate(ordered_pairs):
                indices = ms_pairs[(ms_label, ls_label)]
                for c in range(n_cols):
                    ax = axes[r][c]
                    if c >= len(indices):
                        ax.axis("off")
                        continue

                    idx = indices[c]
                    xcol = f"{ms_label}_band_{idx}"
                    ycol = f"{ls_label}_band_{idx}"
                    if pal_frac.get(xcol, 0.0) >= 0.95 or pal_frac.get(ycol, 0.0) >= 0.95:
                        ax.axis("off")
                        continue

                    err_cols = _present_error_cols(cols, ms_label, idx) + _present_error_cols(
                        cols, ls_label, idx
                    )
                    df = _sample_pair_df(
                        con,
                        parquet,
                        xcol,
                        ycol,
                        max_points,
                        sentinel_at_or_below,
                        extra_zero_filters=err_cols if err_cols else None,
                    )
                    if df.empty:
                        ax.axis("off")
                        continue

                    r_val = _regress_and_plot(ax, df["x"].to_numpy(), df["y"].to_numpy())
                    ms_text = _sensor_band_text(ms_label, idx)
                    ls_text = _sensor_band_text(ls_label, idx)
                    ax.set_title(f"{ls_text} vs {ms_text}  r={r_val:.2f}", fontsize=9)
                    ax.set_xlabel(ms_text, fontsize=8)
                    if c == 0:
                        ax.set_ylabel(ls_text, fontsize=8)
                    else:
                        ax.set_ylabel("")

            fig.suptitle(
                f"{parquet.stem} — MicaSense (X) vs Landsat (Y) bands",
                y=0.995,
                fontsize=12,
            )
            fig.tight_layout()

            out_png = out_path / f"{parquet.stem}__MS_vs_Landsat_FIXED.png"
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out_png)
    finally:
        con.close()

    return outputs


__all__ = [
    "make_sensor_vs_neon_panels",
    "make_micasense_vs_landsat_panels",
]

