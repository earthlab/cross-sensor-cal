"""Centralised HyTools import gateway with friendly diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


class HyToolsNotAvailable(RuntimeError):
    """Raised when HyTools cannot be imported."""


@dataclass
class HyToolsInfo:
    """Metadata about the imported HyTools package."""

    version: str
    has_flexbrdf: bool = True


def _friendly_msg(prefix: str, exc: Exception) -> str:
    return (
        f"{prefix}\n\n"
        "Troubleshooting steps:\n"
        "  1) Confirm pinned install:\n"
        "       pip install -U pip\n"
        "       pip install -e . -c constraints/lock-hytools.txt\n"
        "  2) Verify h5py/numpy versions match constraints.\n"
        "  3) Confirm NEON/ENVI metadata fields expected by BRDF/topo stage.\n"
        f"\nOriginal error: {type(exc).__name__}: {exc}"
    )


def import_hytools() -> Tuple[Any, HyToolsInfo]:
    """Return the HyTools module object and metadata."""

    try:
        import hytools as ht  # type: ignore
    except Exception as exc:  # pragma: no cover - import errors depend on env
        raise HyToolsNotAvailable(
            _friendly_msg(
                "HyTools failed to import (required for multiple stages).", exc
            )
        ) from exc

    version = getattr(ht, "__version__", "unknown")
    info = HyToolsInfo(version=str(version), has_flexbrdf=True)

    return ht, info


__all__ = ["HyToolsInfo", "HyToolsNotAvailable", "import_hytools"]
