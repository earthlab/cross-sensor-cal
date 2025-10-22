from __future__ import annotations
from pathlib import Path
from .sort_core import classify_name

def scan_and_categorize(root: Path) -> dict[str, list[Path]]:
    """
    Recursively scan a directory and group files by category using classify_name().
    Safe for tests with tmp dirs; no network or external deps.
    """
    out: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            cat = classify_name(p.name)
            out.setdefault(cat, []).append(p)
    return out
