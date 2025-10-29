"""Command line entry point for QA panel generation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..qa_plots import summarize_all_flightlines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate QA summary panels for processed flight lines.",
    )
    parser.add_argument(
        "--base-folder",
        type=Path,
        required=True,
        help="Workspace containing per-flightline subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Optional directory for writing QA PNGs. Defaults to each flight folder.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    summarize_all_flightlines(base_folder=args.base_folder, out_dir=args.out_dir)

    target = args.out_dir if args.out_dir is not None else args.base_folder
    print(f"[cscal-qa] ✅ QA panels written to: {target.resolve()}")


__all__ = ["main"]
