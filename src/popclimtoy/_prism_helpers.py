"""PRISM helper utilities vendored from the ESIIL data library.

This module provides utilities for streaming PRISM rasters directly to an
arbitrary area of interest (AOI).  The implementation is based on the helper
script that ships with the Cooperative Institute for Research in Environmental
Sciences (CIRES) ESIIL data library.  The helper is intentionally designed to be
vendored into small projects – we include it here so the notebook examples can
run without additional setup.

The public API intentionally mirrors the original helper so external notebooks
and downstream projects continue to function unchanged.
"""

from __future__ import annotations

import datetime as _dt
import io
import pathlib
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.mask import mask
from rasterio.transform import Affine
import requests
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

__all__ = ["stream_prism_to_polygon"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PrismDescriptor:
    """Container describing a PRISM dataset collection."""

    dataset: str
    requires_zip: bool = True


_FREQ_CODES = {
    "daily": "d",
    "monthly": "m",
    "annual": "y",
}

_NETWORK_CODES = {
    "an": "AN",
    "provisional": "PR",
    "pro": "PR",
    "stable": "ST",
    "st": "ST",
}

# Dataset codes documented by PRISM: https://prism.oregonstate.edu/documents/PRISM_downloads_handout.pdf
_DATASET_LOOKUP: Dict[Tuple[str, str], _PrismDescriptor] = {
    ("an", "daily"): _PrismDescriptor("AN81d"),
    ("an", "monthly"): _PrismDescriptor("AN81m"),
    ("an", "annual"): _PrismDescriptor("AN81y"),
    ("provisional", "daily"): _PrismDescriptor("PRISM_daily_provisional", requires_zip=False),
    ("provisional", "monthly"): _PrismDescriptor("PRISM_monthly_provisional", requires_zip=False),
    ("stable", "monthly"): _PrismDescriptor("STABLE_monthly"),
}


def _coerce_geometry(polygon: Any) -> BaseGeometry:
    """Coerce *polygon* into a shapely geometry."""

    if isinstance(polygon, BaseGeometry):
        return polygon
    if hasattr(polygon, "geometry") and hasattr(polygon.geometry, "iloc"):
        # GeoDataFrame row / GeoSeries
        geom = polygon.geometry.iloc[0] if getattr(polygon, "geometry", None) is not None else polygon.iloc[0]
        if isinstance(geom, BaseGeometry):
            return geom
    if hasattr(polygon, "__geo_interface__"):
        return shape(polygon.__geo_interface__)
    if isinstance(polygon, Mapping):
        return shape(polygon)
    if isinstance(polygon, (list, tuple)):
        return shape(polygon)
    raise TypeError("Unsupported polygon type: {!r}".format(type(polygon)))


def _normalise_date(date: Any, freq: str) -> str:
    if isinstance(date, (np.datetime64,)):
        date = np.datetime_as_string(date, unit="D")
    if isinstance(date, (_dt.datetime, _dt.date)):
        if freq == "daily":
            return date.strftime("%Y%m%d")
        if freq in {"monthly", "annual"}:
            return date.strftime("%Y%m")
    if isinstance(date, str):
        date = date.strip()
        if not date:
            raise ValueError("Date string may not be empty")
        if freq == "daily" and len(date) == 10 and date[4] == "-" and date[7] == "-":
            return date.replace("-", "")
        if freq in {"monthly", "annual"} and len(date) == 7 and date[4] == "-":
            return date.replace("-", "")
        if len(date) == 8 and freq == "daily":
            return date
        if len(date) == 6 and freq in {"monthly", "annual"}:
            return date
    raise ValueError(f"Unsupported date format '{date}' for frequency '{freq}'")


def _dataset_for(network: str, freq: str) -> _PrismDescriptor:
    key = (network.lower(), freq.lower())
    if key in _DATASET_LOOKUP:
        return _DATASET_LOOKUP[key]
    prefix = _NETWORK_CODES.get(network.lower())
    suffix = _FREQ_CODES.get(freq.lower())
    if not prefix or not suffix:
        raise ValueError(f"Unsupported PRISM dataset combination network={network!r}, freq={freq!r}")
    # Default dataset naming scheme: e.g. AN81m, PR81d, etc.
    dataset = f"{prefix}81{suffix}"
    return _PrismDescriptor(dataset)


def _build_prism_url(
    *,
    dataset: str,
    variable: str,
    time_string: str,
    resolution: str,
    region: str,
) -> str:
    params = {
        "dataset": dataset,
        "variable": variable,
        "time": time_string,
        "resolution": resolution,
        "format": "BIL",
        "region": region,
    }
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return f"https://prism.oregonstate.edu/fetchData.php?{query}"


def _read_prism_to_memory(url: str, *, requires_zip: bool) -> Tuple[DatasetReader, tempfile.TemporaryDirectory]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    tmpdir = tempfile.TemporaryDirectory()
    tmp_path = pathlib.Path(tmpdir.name)

    if requires_zip:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(tmp_path)
    else:
        filename = tmp_path / "download.bil"
        filename.write_bytes(response.content)

    # Identify a raster file in the download folder.
    candidate = None
    for ext in (".bil", ".tif", ".tiff", ".img"):
        matches = list(tmp_path.glob(f"*{ext}"))
        if matches:
            candidate = matches[0]
            break
    if candidate is None:
        raise RuntimeError("Unable to locate raster file in PRISM download")

    dataset = rasterio.open(candidate)
    return dataset, tmpdir


def _calculate_extent(transform: Affine, width: int, height: int) -> Tuple[float, float, float, float]:
    xmin, ymax = transform * (0, 0)
    xmax, ymin = transform * (width, height)
    return float(xmin), float(xmax), float(ymin), float(ymax)


def stream_prism_to_polygon(
    polygon: Any,
    polygon_srs: str = "EPSG:4326",
    variable: str = "ppt",
    date: Any = "2015-01",
    resolution: str = "4km",
    freq: str = "monthly",
    region: str = "us",
    network: str = "an",
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mask_outside: bool = True,
    return_array: bool = True,
    compute_stats: bool = False,
) -> Dict[str, Any]:
    """Stream PRISM gridded climate data and clip it to a polygon AOI."""

    geom = _coerce_geometry(polygon)
    gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=polygon_srs)
    gdf4326 = gdf.to_crs("EPSG:4326")
    geom4326 = gdf4326.geometry.iloc[0]

    descriptor = _dataset_for(network, freq)
    time_string = _normalise_date(date, freq)
    url = _build_prism_url(
        dataset=descriptor.dataset,
        variable=variable,
        time_string=time_string,
        resolution=resolution,
        region=region,
    )

    dataset, tmpdir = _read_prism_to_memory(url, requires_zip=descriptor.requires_zip)
    try:
        geometry = [mapping(geom4326)]
        if mask_outside:
            data, transform = mask(dataset, geometry, crop=True)
        else:
            data, transform = mask(dataset, geometry, crop=False, filled=False)
        array = data[0]

        result: Dict[str, Any] = {
            "array": array if return_array else data,
            "transform": transform,
            "extent": _calculate_extent(transform, array.shape[1], array.shape[0]),
            "crs": dataset.crs.to_string() if dataset.crs else "EPSG:4326",
            "metadata": {
                "variable": variable,
                "date": time_string,
                "resolution": resolution,
                "freq": freq,
                "region": region,
                "network": network,
                "title": title,
            },
        }

        if compute_stats:
            valid = array.astype(float)
            if dataset.nodata is not None:
                valid = np.where(array == dataset.nodata, np.nan, array)
            result["stats"] = {
                "mean": float(np.nanmean(valid)),
                "median": float(np.nanmedian(valid)),
                "min": float(np.nanmin(valid)),
                "max": float(np.nanmax(valid)),
                "count": int(np.sum(~np.isnan(valid))),
            }

        return result
    finally:
        dataset.close()
        tmpdir.cleanup()
