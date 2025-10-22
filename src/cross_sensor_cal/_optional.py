"""Helpers for optional heavy dependencies."""
from __future__ import annotations


def _missing(extra: str, package: str) -> RuntimeError:
    return RuntimeError(
        "Optional dependency '{package}' is required for this feature. "
        "Install cross-sensor-cal with the '{extra}' extra, e.g. "
        "`pip install cross-sensor-cal[{extra}]`.".format(package=package, extra=extra)
    )


def require_geopandas():
    try:
        import geopandas as gpd  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in lite environments
        raise _missing("full", "geopandas") from exc
    return gpd


def require_rasterio():
    try:
        import rasterio  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in lite environments
        raise _missing("full", "rasterio") from exc
    return rasterio


def require_spectral():
    try:
        import spectral  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in lite environments
        raise _missing("full", "spectral") from exc
    return spectral


def require_ray():
    try:
        import ray  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in lite environments
        raise _missing("full", "ray") from exc
    return ray


def require_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in lite environments
        raise _missing("full", "h5py") from exc
    return h5py


def require_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "Optional dependency 'matplotlib' is required for plotting utilities. "
            "Install it via `pip install matplotlib` or the 'full' extra."
        ) from exc
    return plt
