"""Command line entry point for the cross-sensor pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..pipelines.pipeline import go_forth_and_multiply


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the cross-sensor calibration pipeline on NEON flight lines.",
    )
    parser.add_argument(
        "--base-folder",
        type=Path,
        required=True,
        help="Workspace where downloads and derived products will be stored.",
    )
    parser.add_argument(
        "--site-code",
        required=True,
        help="NEON site code (e.g. NIWO).",
    )
    parser.add_argument(
        "--year-month",
        required=True,
        help="Acquisition year-month in YYYY-MM format.",
    )
    parser.add_argument(
        "--product-code",
        required=True,
        help="NEON product code (e.g. DP1.30006.001).",
    )
    parser.add_argument(
        "--flight-lines",
        nargs="+",
        required=True,
        metavar="FLIGHTLINE",
        help="One or more NEON flight line identifiers to process.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Ray CPU budget (defaults to 8). Used as worker count for non-Ray engines.",
    )
    parser.add_argument(
        "--resample-method",
        choices=["convolution", "legacy", "resample"],
        default="convolution",
        help="Sensor resampling strategy. Defaults to the modern convolution pipeline.",
    )
    parser.add_argument(
        "--brightness-offset",
        type=float,
        default=None,
        help="Optional brightness offset applied during ENVI export.",
    )
    parser.add_argument(
        "--parquet-chunk-size",
        type=int,
        default=50_000,
        help="Row group size (rows per chunk) used during Parquet export to bound memory usage.",
    )
    parser.add_argument(
        "--merge-memory-limit",
        dest="merge_memory_limit",
        default=None,
        help="DuckDB memory limit for the merge stage (float GiB, 'auto', or literal).",
    )
    parser.add_argument(
        "--merge-threads",
        dest="merge_threads",
        type=int,
        default=None,
        help="DuckDB thread count for the merge stage (defaults to 4).",
    )
    parser.add_argument(
        "--merge-row-group-size",
        dest="merge_row_group_size",
        type=int,
        default=None,
        help="Row group size for merged Parquet output (defaults to 50,000).",
    )
    parser.add_argument(
        "--merge-temp-directory",
        dest="merge_temp_directory",
        type=Path,
        default=None,
        help="Custom DuckDB temp directory for merge spill files.",
    )
    parser.add_argument(
        "--engine",
        choices=["thread", "process", "ray"],
        default="ray",
        help="Parallel engine for flightline dispatch. Ray is the default and initialises automatically.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    go_forth_and_multiply(
        base_folder=args.base_folder,
        site_code=args.site_code,
        year_month=args.year_month,
        product_code=args.product_code,
        flight_lines=list(args.flight_lines),
        resample_method=args.resample_method,
        brightness_offset=args.brightness_offset,
        max_workers=args.max_workers,
        parquet_chunk_size=args.parquet_chunk_size,
        engine=args.engine,
        merge_memory_limit_gb=args.merge_memory_limit,
        merge_threads=args.merge_threads,
        merge_row_group_size=args.merge_row_group_size,
        merge_temp_directory=args.merge_temp_directory,
    )

    target = args.base_folder.resolve()
    print(f"[cscal-pipeline] ✅ Finished processing. Outputs live under: {target}")


__all__ = ["main"]
