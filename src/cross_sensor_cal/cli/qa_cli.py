"""Command line entry point for QA panel generation."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

from ..qa_plots import render_flightline_panel


def _iter_flightlines(base_folder: Path) -> Iterable[Path]:
    for child in sorted(base_folder.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*_envi.img")):
            yield child


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate QA panels and metrics JSON for processed flight lines.",
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
        help="Optional directory for copying QA outputs after generation.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quick",
        action="store_true",
        help="Use deterministic sampling limited to 25k pixels (default).",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Use the configured sample size without subsampling.",
    )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the <prefix>_qa.json metrics file alongside the PNG.",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=100_000,
        help="Maximum number of pixels to sample when computing metrics.",
    )
    parser.add_argument(
        "--rgb-bands",
        type=str,
        help="Override RGB targets as '660,560,490' or symbolic 'R,G,B'.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    base_folder = args.base_folder.expanduser().resolve()
    if not base_folder.exists():
        parser.error(f"Base folder does not exist: {base_folder}")

    out_dir: Path | None = None
    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    quick = True
    if args.full:
        quick = False
    elif args.quick:
        quick = True

    had_errors = False

    for flight_dir in _iter_flightlines(base_folder):
        try:
            png_path, metrics = render_flightline_panel(
                flight_dir,
                quick=quick,
                save_json=args.save_json,
                n_sample=args.n_sample,
                rgb_bands=args.rgb_bands,
            )
        except FileNotFoundError as exc:
            print(f"[cscal-qa] ❌ {exc}", file=sys.stderr)
            had_errors = True
            continue
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[cscal-qa] ❌ Failed on {flight_dir.name}: {exc}", file=sys.stderr)
            had_errors = True
            continue

        json_path = png_path.with_suffix(".json")
        if out_dir is not None:
            shutil.copy2(png_path, out_dir / png_path.name)
            if args.save_json and json_path.exists():
                shutil.copy2(json_path, out_dir / json_path.name)

        if metrics.get("issues"):
            print(
                f"[cscal-qa] ⚠ {flight_dir.name}: {', '.join(metrics['issues'])}",
                file=sys.stderr,
            )
        print(f"[cscal-qa] ✅ {png_path}")

    raise SystemExit(1 if had_errors else 0)


__all__ = ["main"]
