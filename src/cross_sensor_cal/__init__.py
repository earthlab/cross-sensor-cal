"""Compatibility shim for the legacy ``cross_sensor_cal`` namespace.

Importing this module re-exports :mod:`spectralbridge` while emitting a
minimal deprecation warning.
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings

warnings.warn(
    "cross_sensor_cal is deprecated; use spectralbridge instead.",
    DeprecationWarning,
    stacklevel=2,
)

_spectralbridge = importlib.import_module("spectralbridge")

__all__ = getattr(_spectralbridge, "__all__", [])
__path__ = getattr(_spectralbridge, "__path__", [])
__file__ = getattr(_spectralbridge, "__file__", None)
__spec__ = importlib.util.spec_from_loader(__name__, loader=None, is_package=True)
if __spec__ is not None and __path__:
    __spec__.submodule_search_locations = list(__path__)


def __getattr__(name: str):
    return getattr(_spectralbridge, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_spectralbridge)))
