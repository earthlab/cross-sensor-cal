"""Compatibility shim for legacy installation workflows.

This project is configured as a PEP 517/660 package using ``pyproject.toml``.
Direct execution of ``setup.py`` is no longer supported. Please install the
package in editable mode with ``pip install -e .``.
"""

from __future__ import annotations

import sys


def main() -> None:
    message = (
        "This project uses pyproject.toml-based builds. "
        "Run 'pip install -e .' to perform an editable install."
    )
    print(message)
    sys.exit(0)


if __name__ == "__main__":
    main()
