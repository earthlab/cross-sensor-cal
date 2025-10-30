"""Aggregate QA metrics across flightlines and generate dashboards."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _find_metric_tables(base_folder: Path) -> List[Path]:
    """Return all QA metric parquet tables within ``base_folder``."""

    parquets = sorted(base_folder.rglob("*_qa_metrics.parquet"))
    if not parquets:
        logger.warning("No QA metric parquet files found under %s", base_folder)
    return parquets


def _safe_read_parquet(path: Path) -> pd.DataFrame | None:
    """Read a parquet file and annotate flightline metadata."""

    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    df = df.copy()
    df["flight_stem"] = path.stem.replace("_qa_metrics", "")
    df["parent_dir"] = path.parent.name
    return df


def collect_qa_metrics(base_folder: Path) -> pd.DataFrame:
    """Collect all ``*_qa_metrics.parquet`` files into a single dataframe."""

    tables = [_safe_read_parquet(path) for path in _find_metric_tables(base_folder)]
    tables = [t for t in tables if t is not None]
    if not tables:
        logger.warning("No valid QA metric tables loaded.")
        return pd.DataFrame()

    combined = pd.concat(tables, ignore_index=True)
    logger.info(
        "\N{bar chart} Aggregated %d flightlines (%d rows)",
        combined["flight_stem"].nunique(),
        len(combined),
    )
    return combined


def _has_flags(value: object) -> bool:
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return False


def summarize_qa(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-flightline summary statistics."""

    if df.empty:
        return df

    df = df.copy()
    df["has_flag"] = df["flags"].map(_has_flags)

    summary = df.groupby("flight_stem").agg(
        n_bands=("band_index", "count"),
        n_flagged=("has_flag", "sum"),
        median_ratio=("ratio_median", "median"),
        mean_slope_before=("slope_before", "mean"),
        mean_slope_after=("slope_after", "mean"),
        mean_r2_before=("r2_before", "mean"),
        mean_r2_after=("r2_after", "mean"),
    )

    summary["flag_frac"] = summary["n_flagged"] / summary["n_bands"].clip(lower=1)
    summary["slope_reduction"] = summary["mean_slope_before"] - summary["mean_slope_after"]
    summary["r2_reduction"] = summary["mean_r2_before"] - summary["mean_r2_after"]
    summary["flag_status"] = np.where(summary["flag_frac"] > 0.25, "\N{warning sign} check", "\N{white heavy check mark} ok")

    logger.info("\N{white heavy check mark} Computed summary for %d flightlines", len(summary))
    return summary.reset_index()


def plot_qa_dashboard(df_summary: pd.DataFrame, out_png: Path) -> None:
    """Generate an overview plot showing flag fractions per flightline."""

    if df_summary.empty:
        logger.warning("No data to plot.")
        return

    ordered = df_summary.sort_values("flag_frac", ascending=False)
    x_positions = np.arange(len(ordered))

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.bar(x_positions, ordered["flag_frac"], color="tab:red", alpha=0.7, label="Fraction flagged")
    ax.set_ylabel("Fraction of flagged bands")
    ax.set_xlabel("Flightline")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered["flight_stem"], rotation=45, ha="right")
    ax.set_ylim(0, min(1.0, ordered["flag_frac"].max() * 1.1 if not ordered.empty else 1))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.title("QA Summary — BRDF+Topo Correction Performance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    logger.info("\N{framed picture} Saved dashboard → %s", out_png)


def _default_outputs(base_folder: Path) -> tuple[Path, Path]:
    return (
        base_folder / "qa_dashboard_summary.parquet",
        base_folder / "qa_dashboard_summary.png",
    )


def main(argv: Iterable[str] | None = None) -> None:
    """Entry-point for the ``cscal-qa-dashboard`` CLI."""

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Aggregate and summarize QA metrics across flightlines."
    )
    parser.add_argument(
        "--base-folder",
        type=Path,
        required=True,
        help="Root folder containing per-flightline outputs",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        help="Path to combined output parquet file (defaults inside base folder)",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        help="Path to dashboard plot PNG (defaults inside base folder)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_folder: Path = args.base_folder
    df_all = collect_qa_metrics(base_folder)
    if df_all.empty:
        return

    df_summary = summarize_qa(df_all)

    default_parquet, default_png = _default_outputs(base_folder)
    out_parquet = args.out_parquet or default_parquet
    out_png = args.out_png or default_png

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_parquet(out_parquet, index=False)
    logger.info("\N{floppy disk} Wrote aggregated QA summary → %s", out_parquet)

    try:
        plot_qa_dashboard(df_summary, out_png)
    except Exception as exc:  # pragma: no cover - plotting failures should not crash CLI
        logger.warning("Plot generation failed: %s", exc)


if __name__ == "__main__":  # pragma: no cover
    main()

