"""Path utilities for locating package data files."""
from __future__ import annotations

from pathlib import Path


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_package_data_path(filename: str) -> Path:
    """Return the absolute path to a bundled data file."""

    data_path = _package_root() / "data" / filename
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    return data_path
