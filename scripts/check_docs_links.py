#!/usr/bin/env python3
"""Check documentation links and markers.

This script scans Markdown files in the ``docs`` directory and ensures
that:

* All Markdown links are either HTTP(S) or resolve to existing local files.
* No ``FILLME`` markers remain in the documentation.

The script exits with a non-zero status if any issue is detected.
"""
from __future__ import annotations

import os
import pathlib
import re
import sys
from typing import Iterable


def extract_links(text: str) -> Iterable[str]:
    """Return an iterable of links found in Markdown text."""
    pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)|\[[^\]]*\]\(([^)]+)\)")
    for match in pattern.findall(text):
        # ``findall`` with alternation returns tuples; pick the non-empty part.
        link = match[0] or match[1]
        yield link.strip()


def check_file(md_file: pathlib.Path) -> list[str]:
    """Check a single markdown file for broken links and FILLME markers."""
    errors: list[str] = []
    text = md_file.read_text(encoding="utf-8")

    if "FILLME" in text:
        errors.append(f"FILLME marker found in {md_file}")

    for link in extract_links(text):
        if link.startswith("http://") or link.startswith("https://"):
            continue
        if link.startswith("#") or link.startswith("mailto:"):
            continue
        cleaned = link.split("#", 1)[0].split("?", 1)[0]
        if not cleaned:
            continue
        target = (md_file.parent / cleaned).resolve()
        if not target.exists():
            errors.append(f"Broken link in {md_file}: {link}")
    return errors


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    md_files = sorted(docs_dir.glob("*.md"))

    all_errors: list[str] = []
    for md_file in md_files:
        all_errors.extend(check_file(md_file))

    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        return 1
    print("All doc links valid and no FILLME markers found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
