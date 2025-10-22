from __future__ import annotations

import sys
from typing import NoReturn


def _die(msg: str, code: int = 2) -> NoReturn:
    print(f"[cross-sensor-cal] {msg}", file=sys.stderr)
    raise SystemExit(code)


def download_main(argv: list[str] | None = None) -> None:
    """Entry point for the download helper CLI."""

    try:
        from .pipelines.download import run_download
    except Exception as exc:  # pragma: no cover - exercised when module missing
        _die(
            "download pipeline not available (missing module). "
            "Please update your install or open an issue.",
        )
    run_download(argv or sys.argv[1:])


def pipeline_main(argv: list[str] | None = None) -> None:
    """Entry point for the main processing pipeline CLI."""

    try:
        from .pipelines.pipeline import run_pipeline
    except Exception as exc:  # pragma: no cover - exercised when module missing
        _die(
            "main pipeline not available (missing module). "
            "Please update your install or open an issue.",
        )
    run_pipeline(argv or sys.argv[1:])
