"""Centralised HyTools import gateway with friendly diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

try:  # pragma: no cover - optional dependency, fallback handled below
    from packaging.version import InvalidVersion, Version
except Exception:  # pragma: no cover - packaging may be unavailable in env
    InvalidVersion = Version = None  # type: ignore


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


_MIN_VERSION = "1.6.1"
_MAX_VERSION = "2.0.0"


def _version_in_range(raw_version: str) -> bool:
    """Return ``True`` when the HyTools version is supported."""

    if Version is not None:
        try:
            parsed = Version(raw_version)
            return Version(_MIN_VERSION) <= parsed < Version(_MAX_VERSION)
        except InvalidVersion:
            return False

    def _split(v: str) -> Tuple[int, ...]:
        parts = []
        for piece in v.split("."):
            if piece.isdigit():
                parts.append(int(piece))
            else:
                break
        return tuple(parts)

    parsed = _split(raw_version)
    return _split(_MIN_VERSION) <= parsed < _split(_MAX_VERSION)


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

    if not _version_in_range(info.version):
        raise HyToolsNotAvailable(
            _friendly_msg(
                (
                    "HyTools version "
                    f"{info.version!r} is unsupported. Expected versions within "
                    f"[{_MIN_VERSION}, {_MAX_VERSION})."
                ),
                RuntimeError("unsupported HyTools version"),
            )
        )

    return ht, info


__all__ = ["HyToolsInfo", "HyToolsNotAvailable", "import_hytools"]
