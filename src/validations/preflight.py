"""Preflight validation utilities to fail fast on missing inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional


class PreflightError(RuntimeError):
    """Raised when validation checks fail before running heavy processing."""


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def require_paths(paths: Mapping[str, Path], label: str) -> None:
    missing = [name for name, candidate in paths.items() if not _exists(candidate)]
    if missing:
        message = f"Missing {label}: " + ", ".join(missing)
        raise PreflightError(message)


def validate_inputs(
    input_path: Path,
    ancillary: Mapping[str, Path],
    required_keys: Optional[Iterable[str]] = None,
) -> None:
    if not _exists(input_path):
        raise PreflightError(f"Input not found: {input_path}")

    keys = list(required_keys or [])
    needed = {key: ancillary.get(key, Path("")) for key in keys}
    require_paths({key: Path(value) for key, value in needed.items()}, "ancillary inputs")

    # TODO: add CRS/shape/nodata checks as appropriate to the pipeline.
