"""Command line entry point for QA panel generation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ..qa_plots import render_flightline_panel


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        dest="quick",
        action="store_true",
        default=True,
        help="Run deterministic sampling with a lightweight subset (default).",
    )
    mode.add_argument(
        "--full",
        dest="quick",
        action="store_false",
        help="Process the full sampling budget for detailed QA.",
    )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write <prefix>_qa.json next to the PNG (use --no-save-json to skip).",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=100_000,
        help="Maximum number of deterministic samples (default: 100000).",
    )
    parser.add_argument(
        "--rgb-bands",
        type=str,
        help="Override RGB bands as zero-based indices, e.g. '120,90,45'.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    base = args.base_folder
    if not base.exists():
        raise SystemExit("Base folder does not exist")

    flight_dirs: list[Path]
    if any(base.glob("*_envi.img")):
        flight_dirs = [base]
    else:
        flight_dirs = [child for child in sorted(base.iterdir()) if child.is_dir()]

    if not flight_dirs:
        print("[cscal-qa] ⚠️ No flightline directories discovered", file=sys.stderr)
        raise SystemExit(1)

    hard_failures = 0
    warnings = 0
    out_dir = args.out_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for flight_dir in flight_dirs:
        try:
            png_path, metrics = render_flightline_panel(
                flight_dir,
                quick=args.quick,
                save_json=args.save_json,
                n_sample=args.n_sample,
                rgb_bands=args.rgb_bands,
            )
        except FileNotFoundError as exc:
            print(f"[cscal-qa] ❌ {flight_dir.name}: {exc}", file=sys.stderr)
            hard_failures += 1
            continue
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[cscal-qa] ❌ {flight_dir.name}: {exc}", file=sys.stderr)
            hard_failures += 1
            continue

        if out_dir is not None:
            target_png = out_dir / f"{flight_dir.name}_qa.png"
            if png_path != target_png:
                target_png.parent.mkdir(parents=True, exist_ok=True)
                Path(png_path).replace(target_png)
                if args.save_json:
                    json_src = png_path.with_suffix(".json")
                    if json_src.exists():
                        json_dst = target_png.with_suffix(".json")
                        json_src.replace(json_dst)
        issues = metrics.get("issues", [])
        if issues:
            warnings += 1
            print(
                f"[cscal-qa] ⚠️ {flight_dir.name}: issues recorded -> {', '.join(issues)}",
                file=sys.stderr,
            )

    if hard_failures:
        raise SystemExit(1)

    target = out_dir if out_dir is not None else base
    msg = f"[cscal-qa] ✅ QA panels written to: {target.resolve()}"
    if warnings:
        msg += f" (with {warnings} warnings)"
    print(msg)


__all__ = ["main"]
