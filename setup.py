"""Compatibility shim for legacy installation workflows.

This project is configured as a PEP 517/660 package using ``pyproject.toml``.
Direct execution of ``setup.py`` is discouraged, but the file needs to keep
working when tooling such as ``pip`` imports it to generate package metadata.
"""

from __future__ import annotations

import sys

from setuptools import setup


def main() -> None:
    message = (
        "This project uses pyproject.toml-based builds. "
        "Run 'pip install -e .' to perform an editable install."
    )

    if len(sys.argv) == 1:
        print(message)
        sys.exit(0)

    # Allow build front-ends (for example ``pip``) to continue with the
    # standard setuptools build process so metadata can be generated.
    print(message)
    setup()


if __name__ == "__main__":
    main()
