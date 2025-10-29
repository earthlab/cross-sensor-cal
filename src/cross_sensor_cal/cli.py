from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn


def _die(msg: str, code: int = 2) -> NoReturn:
    print(f"[cross-sensor-cal] {msg}", file=sys.stderr)
    raise SystemExit(code)


def download_main(argv: list[str] | None = None) -> None:
    """Entry point for the download helper CLI."""

    try:
        from .pipelines.download import run_download
    except Exception:  # pragma: no cover - exercised when module missing
        _die(
            "download pipeline not available (missing module). "
            "Please update your install or open an issue.",
        )
    run_download(argv or sys.argv[1:])


def pipeline_main(argv: list[str] | None = None) -> None:
    """Entry point for the main processing pipeline CLI."""

    try:
        from .pipelines.pipeline import run_pipeline
    except Exception:  # pragma: no cover - exercised when module missing
        _die(
            "main pipeline not available (missing module). "
            "Please update your install or open an issue.",
        )
    run_pipeline(argv or sys.argv[1:])


def qa_main(argv: list[str] | None = None) -> None:
    """Entry point for the QA plotting helper."""

    parser = argparse.ArgumentParser(description="Generate QA plots for flightlines")
    parser.add_argument("--base-folder", required=True, type=Path, help="Pipeline output root")
    parser.add_argument("--out-dir", type=Path, help="Optional directory to store QA PNGs")
    args = parser.parse_args(argv or sys.argv[1:])

    try:
        from .qa_plots import summarize_all_flightlines
    except Exception:  # pragma: no cover - exercised when module missing
        _die(
            "QA plotting module unavailable. Please update your install or open an issue.",
        )

    summarize_all_flightlines(base_folder=args.base_folder, out_dir=args.out_dir)
