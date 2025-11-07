"""Helpers for reading NEON reflectance HDF5 products across layout versions."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np

__all__ = [
    "is_pre_2021",
    "read_neon_cube",
    "_prepare_map_info",
    "_map_info_core",
]

_DATE_RE = re.compile(r"_([0-9]{8})_")


def is_pre_2021(h5_path: Path) -> bool:
    """Return ``True`` when ``h5_path`` appears to be a pre-2021 NEON export."""

    path = Path(h5_path)
    match = _DATE_RE.search(path.name)
    if match:
        year = int(match.group(1)[:4])
        return year < 2021
    return False


def _prepare_map_info(map_info: np.ndarray | bytes | str) -> list[str]:
    """Parse the NEON map info dataset into an ENVI-style list of strings."""

    def _normalise(component: Any) -> str:
        if isinstance(component, (bytes, np.bytes_)):
            return component.decode("utf-8").strip()
        return str(component).strip()

    if isinstance(map_info, np.ndarray):
        if map_info.ndim == 0:
            return _prepare_map_info(map_info.item())
        if map_info.dtype.kind in {"S", "U", "O"}:
            return [_normalise(value) for value in map_info.tolist()]

    if isinstance(map_info, (bytes, np.bytes_)):
        map_info_str = map_info.decode("utf-8")
    else:
        map_info_str = str(map_info)

    map_info_str = map_info_str.strip()
    if map_info_str.startswith("{") and map_info_str.endswith("}"):
        map_info_str = map_info_str[1:-1]

    return [component.strip() for component in map_info_str.split(",")]


def _map_info_core(map_info_list: list[str]) -> tuple[float, float, float, float, float, float]:
    """Extract numeric components from the map info list for transforms."""

    if len(map_info_list) < 7:
        raise RuntimeError("Map info dataset is shorter than expected for ENVI metadata.")

    def _to_float(value: str) -> float:
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Cannot interpret map info value '{value}' as float.") from exc

    ref_x = _to_float(map_info_list[1])
    ref_y = _to_float(map_info_list[2])
    ref_easting = _to_float(map_info_list[3])
    ref_northing = _to_float(map_info_list[4])
    pixel_x = _to_float(map_info_list[5])
    pixel_y = _to_float(map_info_list[6])

    return ref_x, ref_y, ref_easting, ref_northing, pixel_x, pixel_y


def _as_str(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _extract_units(wavelength_ds: h5py.Dataset, spectral_group: Optional[h5py.Group]) -> Optional[str]:
    for attr_name in ("Units", "Unit", "units"):
        if attr_name in wavelength_ds.attrs:
            return _as_str(wavelength_ds.attrs[attr_name])
    if spectral_group is not None:
        alt = spectral_group.get("Wavelength_Units")
        if alt is not None:
            return _as_str(alt[()])
    return None


def _extract_no_data(dataset: h5py.Dataset) -> float:
    for attr_name in ("Data_Ignore_Value", "_FillValue", "NoData", "no_data"):
        if attr_name in dataset.attrs:
            attr_value = dataset.attrs[attr_name]
            if isinstance(attr_value, (np.ndarray, list, tuple)):
                if len(attr_value) == 0:
                    continue
                attr_value = attr_value[0]
            return float(attr_value)
    raise RuntimeError("Reflectance dataset missing a recognised no-data attribute.")


def _find_dataset_path(h5_file: h5py.File, candidates: Iterable[str], ndim: int | None = None) -> Optional[str]:
    lowered = [candidate.lower() for candidate in candidates]
    matches: list[str] = []

    def _visitor(name: str, obj: h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):  # pragma: no cover - h5py typing quirk
            return
        if ndim is not None and obj.ndim != ndim:
            return
        path_lower = name.lower()
        for candidate in lowered:
            if path_lower.endswith(candidate):
                matches.append(name)
                break

    h5_file.visititems(_visitor)
    if matches:
        matches.sort(key=len)
        return matches[0]
    return None


def _orient_cube(data: np.ndarray, wavelength_count: int) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 3:
        raise RuntimeError("Reflectance data does not have (lines, columns, bands) dimensions.")
    if array.shape[2] == wavelength_count:
        return array
    if array.shape[0] == wavelength_count:
        return np.moveaxis(array, 0, 2)
    if array.shape[1] == wavelength_count:
        return np.moveaxis(array, 1, 2)
    return array


def _metadata_root_from_path(dataset_path: str) -> Optional[str]:
    parts = dataset_path.split("/")
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "metadata":
            return "/".join(parts[: idx + 1])
    if len(parts) > 1:
        return "/".join(parts[:-1])
    return None


def _read_new_neon_layout(h5_file: h5py.File) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    base_key: Optional[str] = None
    for key in h5_file.keys():
        candidate = f"{key}/Reflectance/Reflectance_Data"
        if candidate in h5_file:
            base_key = key
            break
    if base_key is None:
        raise KeyError("Could not locate NEON reflectance dataset within the HDF5 file.")

    reflectance_group = h5_file[f"{base_key}/Reflectance"]
    data_ds = reflectance_group["Reflectance_Data"]
    data = np.asarray(data_ds[()], dtype=np.float32)

    metadata_group = reflectance_group.get("Metadata")
    if metadata_group is None:
        raise KeyError("Missing 'Metadata' group within NEON reflectance file.")

    spectral_group = metadata_group.get("Spectral_Data")
    if spectral_group is None:
        raise KeyError("Missing 'Spectral_Data' group within NEON reflectance metadata.")

    wavelength_ds = spectral_group.get("Wavelength")
    if wavelength_ds is None:
        raise KeyError("NEON file missing spectral 'Wavelength' dataset.")

    wavelengths = np.asarray(wavelength_ds[()], dtype=np.float32).reshape(-1)
    fwhm_ds = spectral_group.get("FWHM")
    fwhm = np.asarray(fwhm_ds[()], dtype=np.float32).reshape(-1) if fwhm_ds is not None else None
    wavelength_units = _extract_units(wavelength_ds, spectral_group) or "Unknown"

    coordinate_group = metadata_group.get("Coordinate_System")
    map_info_dataset = coordinate_group.get("Map_Info") if coordinate_group is not None else None
    projection_dataset = (
        coordinate_group.get("Coordinate_System_String") if coordinate_group is not None else None
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

    no_data = _extract_no_data(data_ds)
    cube = _orient_cube(data, len(wavelengths))

    meta: Dict[str, Any] = {
        "map_info": map_info_list,
        "projection": projection_wkt,
        "transform": transform,
        "ulx": ulx,
        "uly": uly,
        "wavelength_units": wavelength_units,
        "fwhm": fwhm,
        "no_data": no_data,
        "samples": int(cube.shape[1]),
        "lines": int(cube.shape[0]),
        "bands": int(cube.shape[2]),
        "metadata_group_paths": [metadata_group.name],
        "base_key": base_key,
        "layout": "reflectance_group",
    }
    return cube, wavelengths, meta


def _read_old_neon_layout(h5_file: h5py.File) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    data_path = _find_dataset_path(h5_file, ("reflectance_data", "reflectance"), ndim=3)
    if data_path is None:
        raise KeyError("Legacy NEON file missing a reflectance dataset.")
    data_ds = h5_file[data_path]
    data = np.asarray(data_ds[()], dtype=np.float32)

    wavelength_path = _find_dataset_path(
        h5_file,
        ("wavelength", "wavelengths", "center_wavelength"),
        ndim=1,
    )
    if wavelength_path is None:
        raise KeyError("Legacy NEON file missing a wavelength dataset.")
    wavelength_ds = h5_file[wavelength_path]
    wavelengths = np.asarray(wavelength_ds[()], dtype=np.float32).reshape(-1)

    fwhm_path = _find_dataset_path(h5_file, ("fwhm", "full_width_half_max"), ndim=1)
    fwhm = np.asarray(h5_file[fwhm_path][()], dtype=np.float32).reshape(-1) if fwhm_path else None
    wavelength_units = _extract_units(wavelength_ds, None) or "Unknown"

    map_info_path = _find_dataset_path(h5_file, ("map_info",))
    map_info_list: list[str] = []
    if map_info_path:
        map_info_list = _prepare_map_info(h5_file[map_info_path][()])

    projection_path = _find_dataset_path(
        h5_file,
        ("coordinate_system_string", "projection", "wkt"),
    )
    projection_wkt = ""
    if projection_path:
        projection_wkt = _as_str(h5_file[projection_path][()])

    transform = None
    ulx = uly = None
    if map_info_list:
        ref_x, ref_y, ref_easting, ref_northing, pixel_x, pixel_y = _map_info_core(map_info_list)
        ulx = ref_easting - pixel_x * (ref_x - 0.5)
        uly = ref_northing + abs(pixel_y) * (ref_y - 0.5)
        yres = -abs(pixel_y)
        transform = (ulx, pixel_x, 0.0, uly, 0.0, yres)

    no_data = _extract_no_data(data_ds)
    cube = _orient_cube(data, len(wavelengths))

    metadata_root = _metadata_root_from_path(wavelength_path)
    base_key = data_path.split("/")[0] if "/" in data_path else data_path

    meta: Dict[str, Any] = {
        "map_info": map_info_list,
        "projection": projection_wkt,
        "transform": transform,
        "ulx": ulx,
        "uly": uly,
        "wavelength_units": wavelength_units,
        "fwhm": fwhm,
        "no_data": no_data,
        "samples": int(cube.shape[1]),
        "lines": int(cube.shape[0]),
        "bands": int(cube.shape[2]),
        "metadata_group_paths": [metadata_root] if metadata_root else [],
        "base_key": base_key,
        "layout": "legacy_hdf5",
    }
    return cube, wavelengths, meta


def _read_site_group_legacy_layout(
    h5_file: h5py.File,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    root_keys = list(h5_file.keys())
    if len(root_keys) != 1:
        raise KeyError("Site-group legacy layout expects a single root group.")

    site_group_key = root_keys[0]
    site_group_obj = h5_file.get(site_group_key)
    if not isinstance(site_group_obj, h5py.Group):
        raise KeyError("Root entry for legacy site-group layout is not a group.")

    reflectance_group = site_group_obj.get("Reflectance")
    if not isinstance(reflectance_group, h5py.Group):
        raise KeyError("Legacy site-group layout missing 'Reflectance' group.")

    data_ds = reflectance_group.get("Reflectance_Data")
    if not isinstance(data_ds, h5py.Dataset):
        raise KeyError("Legacy site-group layout missing 'Reflectance_Data' dataset.")

    data = np.asarray(data_ds[()], dtype=np.float32)

    metadata_group = reflectance_group.get("Metadata")
    if metadata_group is None:
        raise KeyError("Legacy site-group layout missing 'Metadata' group.")

    spectral_group = metadata_group.get("Spectral_Data")
    if spectral_group is None:
        raise KeyError("Legacy site-group layout missing 'Spectral_Data'.")

    wavelength_ds: Optional[h5py.Dataset] = None
    for key, value in spectral_group.items():
        if isinstance(value, h5py.Dataset) and value.ndim >= 1:
            if key.lower() in {"wavelength", "wavelengths"}:
                wavelength_ds = value
                break
    if wavelength_ds is None:
        raise KeyError("Legacy site-group layout missing spectral wavelength dataset.")

    wavelengths = np.asarray(wavelength_ds[()], dtype=np.float32).reshape(-1)

    fwhm_ds: Optional[h5py.Dataset] = None
    for key, value in spectral_group.items():
        if isinstance(value, h5py.Dataset) and value.ndim >= 1:
            if key.lower() == "fwhm":
                fwhm_ds = value
                break

    fwhm = np.asarray(fwhm_ds[()], dtype=np.float32).reshape(-1) if fwhm_ds else None
    wavelength_units = _extract_units(wavelength_ds, spectral_group) or "Unknown"

    coordinate_group = metadata_group.get("Coordinate_System")
    map_info_dataset = coordinate_group.get("Map_Info") if coordinate_group else None
    projection_dataset = (
        coordinate_group.get("Coordinate_System_String") if coordinate_group else None
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
        ref_x, ref_y, ref_easting, ref_northing, pixel_x, pixel_y = _map_info_core(
            map_info_list
        )
        ulx = ref_easting - pixel_x * (ref_x - 0.5)
        uly = ref_northing + abs(pixel_y) * (ref_y - 0.5)
        yres = -abs(pixel_y)
        transform = (ulx, pixel_x, 0.0, uly, 0.0, yres)

    no_data = _extract_no_data(data_ds)
    cube = _orient_cube(data, len(wavelengths))

    meta: Dict[str, Any] = {
        "map_info": map_info_list,
        "projection": projection_wkt,
        "transform": transform,
        "ulx": ulx,
        "uly": uly,
        "wavelength_units": wavelength_units,
        "fwhm": fwhm,
        "no_data": no_data,
        "samples": int(cube.shape[1]),
        "lines": int(cube.shape[0]),
        "bands": int(cube.shape[2]),
        "metadata_group_paths": [metadata_group.name],
        "base_key": f"{site_group_key}/Reflectance",
        "layout": "legacy_site_group",
        "site": site_group_key,
    }

    return cube, wavelengths, meta


def read_neon_cube(h5_path: Path) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return ``(cube, wavelengths, metadata)`` for ``h5_path`` regardless of layout."""

    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(path)

    layout_error: Exception | None = None

    with h5py.File(path, "r") as h5_file:
        root_keys = list(h5_file.keys())
        if is_pre_2021(path):
            readers = (
                _read_site_group_legacy_layout,
                _read_old_neon_layout,
                _read_new_neon_layout,
            )
        else:
            readers = (
                _read_new_neon_layout,
                _read_old_neon_layout,
                _read_site_group_legacy_layout,
            )

        for reader in readers:
            try:
                return reader(h5_file)
            except Exception as exc:  # pragma: no cover - defensive cascade
                if layout_error is None:
                    layout_error = exc
                continue

    root_summary = ", ".join(root_keys) if root_keys else "<no root groups>"
    if layout_error is not None:
        raise RuntimeError(
            f"Unable to interpret NEON HDF5 layout for {path} (root groups: {root_summary}): {layout_error}"
        ) from layout_error

    raise RuntimeError(
        f"Unable to interpret NEON HDF5 layout for {path} (root groups: {root_summary})."
    )
