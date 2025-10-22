"""Compatibility helpers for importing HyTools components across versions.

HyTools' internal module layout has changed across releases (e.g., ``WriteENVI``
was historically located at ``hytools.io.envi`` but can appear in other modules
in more recent versions).  Downstream code in this repository only needs the
``WriteENVI`` class, so we centralise the import logic here and try a handful of
known locations before surfacing a helpful error message.
"""
from __future__ import annotations

from importlib import import_module
from pkgutil import walk_packages
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
    imported_modules: list[ModuleType] = []
    missing_roots: set[str] = set()

    for path in module_paths:
        try:
            module: ModuleType = import_module(path)
        except ModuleNotFoundError as exc:
            last_exception = exc
            missing_roots.add(path.split(".")[0])
            continue

        imported_modules.append(module)

        if hasattr(module, attribute):
            return getattr(module, attribute)

        last_exception = AttributeError(
            f"Module '{path}' does not provide attribute '{attribute}'."
        )

    # If direct imports failed, search any imported package for the attribute. This
    # accounts for HyTools reorganisations where the public class may live in a
    # nested module that is not part of our historical candidate list.
    seen_packages: set[str] = set()
    for module in imported_modules:
        package_name = module.__name__.split(".")[0]
        if package_name in seen_packages:
            continue
        seen_packages.add(package_name)

        try:
            package = import_module(package_name)
        except ModuleNotFoundError:
            continue

        if hasattr(package, attribute):
            return getattr(package, attribute)

        if not hasattr(package, "__path__"):
            continue

        for module_info in walk_packages(package.__path__, package.__name__ + "."):
            try:
                candidate = import_module(module_info.name)
            except ModuleNotFoundError:
                continue

            if hasattr(candidate, attribute):
                return getattr(candidate, attribute)

    install_hint = ""
    if imported_modules:
        missing_packages = sorted(missing_roots - {module.__name__.split(".")[0] for module in imported_modules})
    else:
        missing_packages = sorted(missing_roots)

    if missing_packages:
        install_hint = (
            " None of the expected HyTools modules could be imported. "
            "Install HyTools (for example `conda install -c conda-forge hytools` "
            "or `pip install hytools`) and ensure it is available on the Python path."
        )

    raise ModuleNotFoundError(error_hint + install_hint) from last_exception


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
