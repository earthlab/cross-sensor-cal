from __future__ import annotations

import sys
import warnings
from pathlib import Path

_LEGACY_PREFIXES = ("cscal-", "csc-")


def warn_if_legacy_command(argv0: str | None = None) -> None:
    command = Path(argv0 or sys.argv[0]).name
    if not command.startswith(_LEGACY_PREFIXES):
        return

    suggested = command
    if command.startswith("cscal-"):
        suggested = command.replace("cscal-", "spectralbridge-", 1)
    elif command.startswith("csc-"):
        suggested = command.replace("csc-", "spectralbridge-", 1)

    warnings.warn(
        f"This command is deprecated; use {suggested} instead.",
        UserWarning,
        stacklevel=2,
    )
