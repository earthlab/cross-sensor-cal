"""Compatibility helpers for importing HyTools components across versions.

HyTools' internal module layout has changed across releases (e.g., ``WriteENVI``
was historically located at ``hytools.io.envi`` but can appear in other modules
in more recent versions).  Downstream code in this repository only needs the
``WriteENVI`` class, so we centralise the import logic here and try a handful of
known locations before surfacing a helpful error message.
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Iterable, TypeVar

T = TypeVar("T")


def _load_attribute(
    module_paths: Iterable[str],
    attribute: str,
    *,
    error_hint: str,
) -> T:
    """Attempt to load ``attribute`` from the first importable module.

    Parameters
    ----------
    module_paths
        Candidate module import paths to try (in order).
    attribute
        Name of the attribute to fetch once the module imports successfully.

    Returns
    -------
    Any
        The requested attribute.

    Raises
    ------
    ModuleNotFoundError
        If none of the candidate modules can be imported or the attribute is
        missing from the imported module.
    """

    last_exception: Exception | None = None
    for path in module_paths:
        try:
            module: ModuleType = import_module(path)
        except ModuleNotFoundError as exc:
            last_exception = exc
            continue

        if hasattr(module, attribute):
            return getattr(module, attribute)

        last_exception = AttributeError(
            f"Module '{path}' does not provide attribute '{attribute}'."
        )

    raise ModuleNotFoundError(error_hint) from last_exception


def get_write_envi() -> T:
    """Return the ``WriteENVI`` class from HyTools regardless of version."""

    candidate_modules = (
        "hytools.io.envi",
        "hytools.io",
        "hytools.envi",
    )
    return _load_attribute(
        candidate_modules,
        "WriteENVI",
        error_hint=(
            "Could not import 'WriteENVI' from HyTools. Ensure the 'hytools' package "
            "is installed and up-to-date."
        ),
    )


def get_hytools_class() -> T:
    """Return the ``HyTools`` class across supported package layouts."""

    candidate_modules = (
        "hytools",
        "hytools.hytools",
        "hytools.base",
        "hytools.core",
        "hytools.core.hytools",
        "hytools.core.base",
    )
    return _load_attribute(
        candidate_modules,
        "HyTools",
        error_hint=(
            "Could not import the 'HyTools' class. Ensure the installed 'hytools' package "
            "exposes the HyTools API."
        ),
    )
