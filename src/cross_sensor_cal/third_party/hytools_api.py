from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class HyToolsNotAvailable(RuntimeError):
    pass


@dataclass
class HyToolsInfo:
    version: str = "unknown"


def _friendly_msg(prefix: str, exc: Exception) -> str:
    return (
        f"{prefix}\n\n"
        "Troubleshooting:\n"
        "  1) Use CI pins locally:\n"
        "       pip install -U \"pip<25\"\n"
        "       pip install -r constraints/requirements-ci.txt\n"
        "  2) Ensure GDAL/PROJ are available (see README install instructions).\n"
        f"\nOriginal error: {type(exc).__name__}: {exc}"
    )


def import_hytools() -> tuple[Any, HyToolsInfo]:
    try:
        import hytools as ht  # type: ignore
    except Exception as e:  # pragma: no cover - exercised when hytools missing
        raise HyToolsNotAvailable(
            _friendly_msg(
                "HyTools failed to import (required for multiple stages).",
                e,
            )
        ) from e
    info = HyToolsInfo(version=str(getattr(ht, "__version__", "unknown")))
    return ht, info
