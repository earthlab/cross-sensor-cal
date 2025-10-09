"""Compatibility helpers for importing HyTools components across versions.

HyTools' internal module layout has changed across releases (e.g., ``WriteENVI``
was historically located at ``hytools.io.envi`` but can appear in other modules
in more recent versions).  Downstream code in this repository only needs the
``WriteENVI`` class, so we centralise the import logic here and try a handful of
known locations before surfacing a helpful error message.
"""
from __future__ import annotations

import inspect
import os
from importlib import import_module
from pkgutil import walk_packages
from types import ModuleType
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def _load_attribute(
    module_paths: Iterable[str],
    attribute: str,
    *,
    error_hint: str,
    fallback: Callable[[list[ModuleType]], T | None] | None = None,
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

    for path in module_paths:
        try:
            module: ModuleType = import_module(path)
        except ModuleNotFoundError as exc:
            last_exception = exc
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

    if fallback is not None:
        candidate = fallback(imported_modules)
        if candidate is not None:
            return candidate

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
    env_override = os.getenv("CROSS_SENSOR_CAL_HYTOOLS_CLASS")

    if env_override:
        module_path, _, attr_name = env_override.rpartition(":")
        if not module_path:
            module_path, attr_name = attr_name.rsplit(".", 1)
        try:
            module = import_module(module_path)
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive
            raise ModuleNotFoundError(
                "Environment variable CROSS_SENSOR_CAL_HYTOOLS_CLASS points to "
                f"unknown module '{module_path}'."
            ) from exc

        if not hasattr(module, attr_name):
            raise ModuleNotFoundError(
                "Environment variable CROSS_SENSOR_CAL_HYTOOLS_CLASS references "
                f"missing attribute '{attr_name}' in module '{module_path}'."
            )

        return getattr(module, attr_name)

    def _hytools_fallback(imported_modules: list[ModuleType]) -> T | None:
        try:
            root_package = import_module("hytools")
        except ModuleNotFoundError:
            return None

        modules_to_search: list[ModuleType] = []

        def _maybe_add(module: ModuleType) -> None:
            if module not in modules_to_search:
                modules_to_search.append(module)

        for module in imported_modules:
            if module.__name__.split(".")[0] == "hytools":
                _maybe_add(module)

        _maybe_add(root_package)

        if hasattr(root_package, "__path__"):
            for module_info in walk_packages(root_package.__path__, root_package.__name__ + "."):
                try:
                    candidate_module = import_module(module_info.name)
                except ModuleNotFoundError:
                    continue
                except Exception:
                    # Some HyTools extras lazily import optional dependencies. Skip modules
                    # that fail to import so that we can continue scanning the package.
                    continue

                _maybe_add(candidate_module)

        required_methods = ("read_file", "iterate", "get_header")

        for module in modules_to_search:
            try:
                members = inspect.getmembers(module, inspect.isclass)
            except Exception:
                continue

            for _, cls in members:
                if not getattr(cls, "__module__", "").startswith("hytools"):
                    continue
                name = getattr(cls, "__name__", "")
                if "hytools" not in name.lower():
                    continue
                if all(hasattr(cls, method) for method in required_methods):
                    return cls  # type: ignore[return-value]

        return None

    return _load_attribute(
        candidate_modules,
        "HyTools",
        error_hint=(
            "Could not import the 'HyTools' class. Ensure the installed 'hytools' package "
            "exposes the HyTools API."
        ),
        fallback=_hytools_fallback,
    )
