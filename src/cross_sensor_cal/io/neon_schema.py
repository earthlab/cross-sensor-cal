"""Lightweight helpers for inspecting NEON reflectance HDF5 layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import h5py
import numpy as np

from .neon import (
    _as_str,
    _extract_no_data,
    _extract_units,
    _map_info_core,
    _prepare_map_info,
)
from .neon_legacy import NeonPaths, detect_legacy_neon_schema, resolve_neon_paths

__all__ = [
    "NeonResolved",
    "resolve",
    "iter_reflectance_rows",
    "canonical_vectors",
    "compact_ancillary",
]


@dataclass
class NeonResolved:
    """Resolved dataset handles for a NEON reflectance cube."""

    ds_reflectance: h5py.Dataset
    ds_wavelength: h5py.Dataset
    ds_fwhm: Optional[h5py.Dataset]
    ds_to_sun_zenith: Optional[h5py.Dataset]
    ds_to_sensor_zenith: Optional[h5py.Dataset]
    is_legacy: bool
    metadata: Dict[str, Any]


def _with_prefix(base_key: Optional[str], paths: NeonPaths) -> NeonPaths:
    if not base_key:
        return paths

    prefix = f"{base_key}/"

    def _prefix(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return prefix + value

    return NeonPaths(
        reflectance=_prefix(paths.reflectance) or paths.reflectance,
        wavelength=_prefix(paths.wavelength) or paths.wavelength,
        fwhm=_prefix(paths.fwhm) if paths.fwhm else None,
        solar_zenith=_prefix(paths.solar_zenith) if paths.solar_zenith else None,
        sensor_zenith=_prefix(paths.sensor_zenith) if paths.sensor_zenith else None,
    )


def _resolve_base_group(h5_file: h5py.File) -> Tuple[h5py.Group | h5py.File, Optional[str]]:
    base_key: Optional[str] = None
    for key in h5_file.keys():
        candidate = f"{key}/Reflectance/Reflectance_Data"
        if candidate in h5_file:
            base_key = key
            break

    if base_key is None and "Reflectance/Reflectance_Data" not in h5_file:
        raise KeyError("Could not locate NEON reflectance dataset within the HDF5 file.")

    if base_key is None:
        return h5_file, None
    return h5_file[base_key], base_key


def resolve(h5_file: h5py.File) -> NeonResolved:
    """Resolve dataset handles and metadata for a NEON reflectance cube."""

    base_group, base_key = _resolve_base_group(h5_file)
    is_legacy = detect_legacy_neon_schema(base_group)
    paths = _with_prefix(base_key, resolve_neon_paths(base_group))

    reflectance_ds = h5_file[paths.reflectance]
    wavelength_ds = h5_file[paths.wavelength]
    fwhm_ds = h5_file[paths.fwhm] if paths.fwhm and paths.fwhm in h5_file else None
    sun_ds = h5_file[paths.solar_zenith] if paths.solar_zenith and paths.solar_zenith in h5_file else None
    sensor_ds = (
        h5_file[paths.sensor_zenith]
        if paths.sensor_zenith and paths.sensor_zenith in h5_file
        else None
    )

    reflectance_group = reflectance_ds.parent
    if not isinstance(reflectance_group, h5py.Group):
        raise KeyError("Reflectance dataset is not within a group as expected.")

    metadata_group = reflectance_group.get("Metadata")
    if metadata_group is None:
        raise KeyError("Missing 'Metadata' group within NEON reflectance file.")

    spectral_group = wavelength_ds.parent
    if not isinstance(spectral_group, h5py.Group):
        raise KeyError("Missing 'Spectral_Data' group within NEON reflectance metadata.")

    wavelength_units = _extract_units(wavelength_ds, spectral_group) or "Unknown"

    coordinate_group = metadata_group.get("Coordinate_System")
    map_info_dataset = coordinate_group.get("Map_Info") if coordinate_group is not None else None
    projection_dataset = (
        coordinate_group.get("Coordinate_System_String")
        if coordinate_group is not None
        else None
    )

    map_info_list: list[str] = []
    if map_info_dataset is not None:
        map_info_list = _prepare_map_info(map_info_dataset[()])

    projection_wkt = ""
    if projection_dataset is not None:
        projection_wkt = _as_str(projection_dataset[()])

    transform = None
    ulx = uly = None
    if map_info_list:
        ref_x, ref_y, ref_easting, ref_northing, pixel_x, pixel_y = _map_info_core(map_info_list)
        ulx = ref_easting - pixel_x * (ref_x - 0.5)
        uly = ref_northing + abs(pixel_y) * (ref_y - 0.5)
        yres = -abs(pixel_y)
        transform = (ulx, pixel_x, 0.0, uly, 0.0, yres)

    no_data = _extract_no_data(reflectance_ds)

    metadata: Dict[str, Any] = {
        "map_info": map_info_list,
        "projection": projection_wkt,
        "transform": transform,
        "ulx": ulx,
        "uly": uly,
        "wavelength_units": wavelength_units,
        "no_data": no_data,
        "samples": int(reflectance_ds.shape[1]),
        "lines": int(reflectance_ds.shape[0]),
        "bands": int(reflectance_ds.shape[2]),
        "base_key": base_key,
    }

    return NeonResolved(
        ds_reflectance=reflectance_ds,
        ds_wavelength=wavelength_ds,
        ds_fwhm=fwhm_ds,
        ds_to_sun_zenith=sun_ds,
        ds_to_sensor_zenith=sensor_ds,
        is_legacy=is_legacy,
        metadata=metadata,
    )


def iter_reflectance_rows(
    ds_reflectance: h5py.Dataset,
    row_chunk: int = 128,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Yield ``(row_start, row_stop, block)`` slices without loading the full cube."""

    nrows, ncols, _ = ds_reflectance.shape
    for r0 in range(0, nrows, row_chunk):
        r1 = min(r0 + row_chunk, nrows)
        block = ds_reflectance[r0:r1, :, :]
        if block.dtype != np.float32:
            block = block.astype(np.float32, copy=False)
        yield r0, r1, block


def canonical_vectors(
    nr: NeonResolved,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return wavelength, FWHM, and ancillary angle vectors as float32 arrays."""

    wavelength_nm = np.asarray(nr.ds_wavelength[()], dtype=np.float32).reshape(-1)
    fwhm_nm = None
    if nr.ds_fwhm is not None:
        fwhm_nm = np.asarray(nr.ds_fwhm[()], dtype=np.float32).reshape(-1)

    sun = None
    if nr.ds_to_sun_zenith is not None:
        sun = np.asarray(nr.ds_to_sun_zenith[()], dtype=np.float32)

    sensor = None
    if nr.ds_to_sensor_zenith is not None:
        sensor = np.asarray(nr.ds_to_sensor_zenith[()], dtype=np.float32)

    return wavelength_nm, fwhm_nm, sun, sensor


def compact_ancillary(
    to_sun_zenith: Optional[np.ndarray],
    to_sensor_zenith: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Reduce ancillary angle rasters to compact float32 vectors or scalars."""

    sun = None
    sen = None
    if to_sun_zenith is not None:
        arr = np.asarray(to_sun_zenith)
        if arr.ndim >= 2:
            sun = np.array([np.nanmean(arr, dtype=np.float64)], dtype=np.float32)
        else:
            sun = arr.astype(np.float32, copy=False)
    if to_sensor_zenith is not None:
        arr = np.asarray(to_sensor_zenith)
        if arr.ndim >= 2:
            sen = np.array([np.nanmean(arr, dtype=np.float64)], dtype=np.float32)
        else:
            sen = arr.astype(np.float32, copy=False)
    return sun, sen

