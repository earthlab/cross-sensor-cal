"""CLI entry points exposed for backwards compatibility."""
from __future__ import annotations

import sys
from typing import NoReturn

from spectralbridge._cli_compat import warn_if_legacy_command

from .pipeline_cli import main as pipeline_cli_main
from .qa_cli import main as qa_cli_main

__all__ = ["download_main", "pipeline_main", "qa_main", "warn_if_legacy_command"]


def _die(msg: str, code: int = 2) -> NoReturn:
    print(f"[SpectralBridge] {msg}", file=sys.stderr)
    raise SystemExit(code)


def download_main(argv: list[str] | None = None) -> None:
    """Entry point for the download helper CLI."""

    warn_if_legacy_command()

    try:
        from ..pipelines.download import run_download
    except Exception:  # pragma: no cover - exercised when module missing
        _die(
            "download pipeline not available (missing module). "
            "Please update your install or open an issue.",
        )

    run_download(argv or sys.argv[1:])


def pipeline_main(argv: list[str] | None = None) -> None:
    """Backward compatible wrapper around :mod:`pipeline_cli`."""

    warn_if_legacy_command()
    pipeline_cli_main(argv)


def qa_main(argv: list[str] | None = None) -> None:
    """Backward compatible wrapper around :mod:`qa_cli`."""

    warn_if_legacy_command()
    qa_cli_main(argv)
