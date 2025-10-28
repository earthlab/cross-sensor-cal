"""Lightweight file integrity helpers for pipeline idempotency checks."""

from __future__ import annotations

import json
from pathlib import Path


def _nonempty_file(path: Path | None) -> bool:
    """Return ``True`` if *path* exists, is a file, and has non-zero size."""

    return (
        path is not None
        and isinstance(path, Path)
        and path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


def is_valid_envi_pair(img_path: Path, hdr_path: Path) -> bool:
    """Return ``True`` if both ENVI files exist and are non-empty."""

    return _nonempty_file(img_path) and _nonempty_file(hdr_path)


def is_valid_json(json_path: Path) -> bool:
    """Return ``True`` if the JSON file exists, is non-empty, and parses."""

    if not _nonempty_file(json_path):
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            json.load(handle)
    except Exception:  # pragma: no cover - defensive guard
        return False
    return True


__all__ = ["_nonempty_file", "is_valid_envi_pair", "is_valid_json"]
