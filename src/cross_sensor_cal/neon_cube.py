"""NEON-specific in-memory hyperspectral cube handling.

This module vendors the small portion of HyTools' NEON reader that is needed for
cross-sensor-cal workflows.  The logic here is adapted and simplified from
HyTools' ``open_neon`` and ENVI header helpers so that we can operate on NEON AOP
reflectance products without depending on the full HyTools package at runtime.

HyTools: Hyperspectral image processing library
Authors: Adam Chlus, Zhiwei Ye, Philip Townsend.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import math

import numpy as np
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from .io.neon import _map_info_core, read_neon_cube


_RADIANS_COMPATIBLE_KEYS = {
    "slope",
    "sensor_az",
    "sensor_zn",
    "aspect",
    "solar_zn",
    "solar_az",
}


@dataclass
class _MetadataEntry:
    """Container linking metadata names to HDF5 dataset paths."""

    name: str
    path: str


class NeonCube:
    """In-memory NEON hyperspectral cube loader.

    The implementation intentionally targets NEON AOP reflectance products only;
    it mirrors the pieces of HyTools that cross-sensor-cal relies upon while
    avoiding a runtime dependency on HyTools itself.  The spectral cube is loaded
    fully into memory (as ``float32``) upon initialisation along with essential
    metadata required for BRDF/topographic corrections and ENVI exports.
    """

    def __init__(
        self,
        h5_path: str | Path,
        ancillary_paths: Optional[Dict[str, str | Path]] = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise RuntimeError(f"NEON HDF5 file not found: {self.h5_path}")

        self.ancillary_paths: Dict[str, Path] = {
            key: Path(value) for key, value in (ancillary_paths or {}).items()
        }

        self._metadata_index: Dict[str, list[_MetadataEntry]] = {}

        cube_data, wavelengths, meta = read_neon_cube(self.h5_path)

        self.data = np.asarray(cube_data, dtype=np.float32)
        if self.data.ndim != 3:
            raise RuntimeError(
                "Reflectance data does not have (lines, columns, bands) dimensions."
            )

        self.lines, self.columns, self.bands = self.data.shape

        self.wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
        if self.wavelengths.size != self.bands:
            raise RuntimeError(
                "Spectral wavelength metadata length does not match hyperspectral band count."
            )

        fwhm = meta.get("fwhm")
        self.fwhm = np.asarray(fwhm, dtype=np.float32).reshape(-1) if fwhm is not None else None
        if self.fwhm is not None and self.fwhm.shape != self.wavelengths.shape:
            raise RuntimeError(
                "Spectral wavelength and FWHM arrays do not share the same length."
            )

        self.wavelength_units = str(meta.get("wavelength_units", "Unknown"))
        self.map_info_list = list(meta.get("map_info", []))
        self.projection_wkt = str(meta.get("projection", ""))

        transform = meta.get("transform")
        if transform is None and self.map_info_list:
            ref_x, ref_y, ref_easting, ref_northing, pixel_x, pixel_y = _map_info_core(
                self.map_info_list
            )
            ulx = ref_easting - pixel_x * (ref_x - 0.5)
            uly = ref_northing + abs(pixel_y) * (ref_y - 0.5)
            yres = -abs(pixel_y)
            transform = (ulx, pixel_x, 0.0, uly, 0.0, yres)
            meta.setdefault("ulx", ulx)
            meta.setdefault("uly", uly)

        if transform is None:
            raise RuntimeError("Cannot determine geospatial transform for NEON cube.")

        self.transform = tuple(float(value) for value in transform)
        self.ulx = float(meta.get("ulx", self.transform[0]))
        self.uly = float(meta.get("uly", self.transform[3]))

        no_data_value = meta.get("no_data")
        if no_data_value is None:
            raise RuntimeError("Reflectance dataset missing a recognised no-data attribute.")
        self.no_data = float(no_data_value)

        band0 = self.data[:, :, 0]
        self.mask_no_data = band0 != self.no_data

        mem_gb = self.data.nbytes / (1024 ** 3)
        print(
            "NeonCube: loaded "
            f"{self.lines}x{self.columns}x{self.bands} ~{mem_gb:.2f} GB into memory (float32)"
        )

        self.base_key = meta.get("base_key")

        metadata_roots = [path for path in meta.get("metadata_group_paths", []) if path]
        with h5py.File(self.h5_path, "r") as h5_file:
            if metadata_roots:
                for group_path in metadata_roots:
                    group = h5_file.get(group_path)
                    if isinstance(group, h5py.Group):
                        self._index_metadata(group)
            else:
                self._index_metadata(h5_file)

    def _index_metadata(self, metadata_group: h5py.Group) -> None:
        """Recursively index metadata datasets for ancillary lookup."""

        def recurse(group: h5py.Group, prefix: str) -> None:
            for key, item in group.items():
                current_path = f"{prefix}/{key}" if prefix else key
                if isinstance(item, h5py.Dataset):
                    normalized = key.lower()
                    sanitized = normalized.replace("-", "_")
                    entry = _MetadataEntry(name=sanitized, path=current_path)
                    self._metadata_index.setdefault(sanitized, []).append(entry)
                    if sanitized != normalized:
                        self._metadata_index.setdefault(normalized, []).append(entry)
                elif isinstance(item, h5py.Group):
                    recurse(item, current_path)

        recurse(metadata_group, metadata_group.name)

    def _lookup_metadata_path(self, name: str) -> Optional[str]:
        """Return the first dataset path matching ``name`` in the metadata index."""

        candidates = self._metadata_index.get(name)
        if candidates:
            return candidates[0].path
        return None

    def get_ancillary(self, name: str, radians: bool = True) -> np.ndarray:
        """Return an ancillary raster or angle for correction workflows.

        This method adapts HyTools' ancillary retrieval logic for the NEON-only
        use case within cross-sensor-cal (HyTools: Hyperspectral image processing
        library; Authors: Adam Chlus, Zhiwei Ye, Philip Townsend. GPLv3).
        """

        normalized_name = name.lower().replace("-", "_")

        metadata_aliases = {
            "solar_zn": [
                "solar_zenith_angle",
                "mean_solar_zenith_angle",
                "solar_zn",
            ],
            "solar_az": [
                "solar_azimuth_angle",
                "solar_az",
            ],
            "sensor_zn": [
                "to_sensor_zenith_angle",
                "sensor_zenith_angle",
                "sensor_zn",
            ],
            "sensor_az": [
                "to_sensor_azimuth_angle",
                "sensor_azimuth_angle",
                "sensor_az",
            ],
            "slope": ["slope"],
            "aspect": ["aspect"],
        }

        metadata_path: Optional[str] = None
        for alias in metadata_aliases.get(normalized_name, []):
            metadata_path = self._lookup_metadata_path(alias)
            if metadata_path:
                break

        data_array: Optional[np.ndarray] = None

        if metadata_path:
            with h5py.File(self.h5_path, "r") as h5_file:
                dataset = h5_file.get(metadata_path)
                if dataset is None:
                    raise RuntimeError(
                        f"Indexed ancillary dataset missing from file: {metadata_path}"
                    )
                data_array = dataset[()]

        if data_array is None:
            # Fall back to ancillary ENVI rasters when available.
            if normalized_name in {"slope", "aspect"}:
                ancillary_path = self.ancillary_paths.get(normalized_name)
                if ancillary_path is not None:
                    data_array = _read_envi_single_band(ancillary_path)
                else:
                    raise ValueError(
                        f"Required ancillary '{name}' not found in HDF5 and no "
                        f"ancillary_paths['{name}'] was provided. "
                        "Topographic/BRDF correction cannot proceed."
                    )
            else:
                raise ValueError(
                    f"Ancillary '{name}' not available in NEON metadata."
                )

        array = np.asarray(data_array)

        if array.ndim == 0:
            array = np.full((self.lines, self.columns), float(array), dtype=np.float32)
        elif array.ndim == 1:
            # Use the mean of the available log entries and broadcast.
            mean_value = float(np.nanmean(array.astype(np.float64)))
            array = np.full((self.lines, self.columns), mean_value, dtype=np.float32)
        elif array.ndim == 2:
            if array.shape != (self.lines, self.columns):
                raise ValueError(
                    f"Ancillary '{name}' has shape {array.shape} which does not "
                    f"match the reflectance cube ({self.lines}, {self.columns})."
                )
            array = array.astype(np.float32, copy=False)
        else:
            raise ValueError(
                f"Ancillary '{name}' has unsupported dimensionality: {array.shape}."
            )

        if radians and normalized_name in _RADIANS_COMPATIBLE_KEYS:
            array = np.asarray(np.radians(array), dtype=np.float32)
        else:
            array = array.astype(np.float32, copy=False)

        return array

    def iter_chunks(
        self,
        chunk_y: int = 100,
        chunk_x: int = 100,
    ) -> Iterator[Tuple[int, int, int, int, np.ndarray]]:
        """Yield sequential spatial chunks of the reflectance cube."""

        for ys in range(0, self.lines, chunk_y):
            ye = min(self.lines, ys + chunk_y)
            for xs in range(0, self.columns, chunk_x):
                xe = min(self.columns, xs + chunk_x)
                chunk = self.data[ys:ye, xs:xe, :]
                yield ys, ye, xs, xe, chunk

    def chunk_count(self, chunk_y: int = 100, chunk_x: int = 100) -> int:
        """Return the number of spatial chunks produced by :meth:`iter_chunks`."""

        if chunk_y <= 0 or chunk_x <= 0:
            raise ValueError("chunk dimensions must be positive")

        return max(
            1,
            math.ceil(self.lines / chunk_y) * math.ceil(self.columns / chunk_x),
        )

    def build_envi_header(self) -> dict:
        """Construct an ENVI header compatible with NEON reflectance outputs.

        Adapted from HyTools' ENVI export routines for NEON reflectance data
        (HyTools: Hyperspectral image processing library; Authors: Adam Chlus,
        Zhiwei Ye, Philip Townsend. GPLv3).
        """

        wavelengths = [float(value) for value in np.asarray(self.wavelengths).ravel().tolist()]
        if len(wavelengths) != self.bands:
            raise RuntimeError(
                "Spectral wavelength metadata length does not match hyperspectral band count."
            )

        fwhm: list[float]
        if self.fwhm is None:
            fwhm = []
        else:
            fwhm = [float(value) for value in np.asarray(self.fwhm).ravel().tolist()]
            if len(fwhm) != len(wavelengths):
                raise RuntimeError(
                    "Spectral FWHM metadata length does not match wavelength list length."
                )

        wavelength_units = str(self.wavelength_units) if self.wavelength_units is not None else ""

        return {
            "samples": int(self.columns),
            "lines": int(self.lines),
            "bands": int(self.bands),
            "interleave": "bsq",
            "data type": 4,
            "byte order": 0,
            "wavelength": wavelengths,
            "fwhm": fwhm,
            "wavelength units": wavelength_units,
            "map info": list(self.map_info_list),
            "projection": self.projection_wkt,
            "transform": tuple(self.transform),
            "ulx": float(self.ulx),
            "uly": float(self.uly),
        }


def _read_envi_single_band(img_path: Path) -> np.ndarray:
    """Read a single-band ENVI raster for ancillary information.

    This helper is a NEON-focused adaptation of HyTools' ENVI parsing utilities
    (HyTools: Hyperspectral image processing library; Authors: Adam Chlus,
    Zhiwei Ye, Philip Townsend. GPLv3).  It assumes BSQ interleave, ``float32``
    data type, and returns the first (and typically only) band as a ``float32``
    array shaped ``(lines, samples)``.
    """

    img_path = Path(img_path)
    if not img_path.exists():
        raise RuntimeError(f"Ancillary ENVI raster not found: {img_path}")

    hdr_path = img_path.with_suffix(".hdr")
    if not hdr_path.exists():
        raise RuntimeError(f"ENVI header file missing alongside ancillary raster: {hdr_path}")

    header_values: Dict[str, str] = {}
    multiline_key: Optional[str] = None

    with hdr_path.open("r", encoding="utf-8") as hdr_file:
        for line in hdr_file:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue

            if multiline_key is not None:
                header_values[multiline_key] += " " + stripped
                if stripped.endswith("}"):
                    header_values[multiline_key] = header_values[multiline_key].rstrip("}")
                    multiline_key = None
                continue

            if "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip().lower()
            value = value.strip()

            if value.startswith("{") and not value.endswith("}"):
                multiline_key = key
                header_values[key] = value.lstrip("{").strip()
                continue

            header_values[key] = value.strip().strip("{}")

    required_keys = {"samples", "lines", "data type", "interleave", "byte order"}
    missing = required_keys - header_values.keys()
    if missing:
        raise RuntimeError(
            f"ENVI header missing required keys: {', '.join(sorted(missing))}"
        )

    samples = int(float(header_values["samples"]))
    lines = int(float(header_values["lines"]))
    bands = int(float(header_values.get("bands", "1")))
    data_type = int(float(header_values["data type"]))
    interleave = header_values["interleave"].strip().lower()
    byte_order = int(float(header_values["byte order"]))

    if interleave != "bsq":
        raise RuntimeError(
            f"Ancillary raster interleave '{interleave}' is not supported (expected 'bsq')."
        )

    if data_type != 4:
        raise RuntimeError(
            f"Ancillary raster data type {data_type} is not supported (expected ENVI code 4)."
        )

    dtype = np.dtype("<f4" if byte_order == 0 else ">f4")

    expected_size = lines * samples * bands
    data = np.fromfile(img_path, dtype=dtype, count=expected_size)
    if data.size != expected_size:
        raise RuntimeError(
            f"Ancillary raster size mismatch. Expected {expected_size} float32 values, got {data.size}."
        )

    data = data.reshape(bands, lines, samples)
    return data[0].astype(np.float32)

