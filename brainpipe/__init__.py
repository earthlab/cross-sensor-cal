"""A lightweight stub of the :mod:`brainpipe` package used by hytools.

The upstream hytools package optionally imports :mod:`brainpipe` for MEG
processing utilities.  That dependency is not required for the hyperspectral
processing workflows in this repository, but the import still happens when
hytools is imported which results in a :class:`ModuleNotFoundError` in
environments where ``brainpipe`` is not installed.  This module provides a tiny
stand-in so that hytools can be imported.  If any functionality from the real
package is accessed we raise a clear error directing users to install the
optional dependency instead of failing with a cryptic import error.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any


class _FeatureModule(ModuleType):
    """Placeholder module that raises helpful errors when accessed."""

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError(
            "brainpipe is not installed. Install the real 'brainpipe' package to "
            "use its feature module."
        )


feature = _FeatureModule("brainpipe.feature")

__all__ = ["feature"]
