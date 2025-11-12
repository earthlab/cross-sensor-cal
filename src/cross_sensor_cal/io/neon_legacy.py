from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import h5py


@dataclass
class NeonPaths:
    reflectance: str
    wavelength: str
    fwhm: Optional[str]
    solar_zenith: Optional[str]
    sensor_zenith: Optional[str]
    # add more fields here only if the code already uses them elsewhere


def _has(g: h5py.File, path: str) -> bool:
    try:
        g[path]
        return True
    except KeyError:
        return False


def detect_legacy_neon_schema(h5: h5py.File) -> bool:
    """
    Heuristically detect pre-2021 NEON AOP DP1.30006 HDF5 layout.
    Legacy signals (any True -> legacy):
      - Spectral wavelength dataset capitalized as 'Wavelength' not 'wavelength'
      - FWHM capitalized 'FWHM' not 'fwhm'
      - Angle metadata capitalized: 'to-sun_Zenith_Angle', 'to-sensor_Zenith_Angle'
    """
    legacy_caps = _has(h5, "Reflectance/Metadata/Spectral_Data/Wavelength") \
                  or _has(h5, "Reflectance/Metadata/Spectral_Data/FWHM")
    legacy_angles = _has(h5, "Reflectance/Metadata/to-sun_Zenith_Angle") \
                    or _has(h5, "Reflectance/Metadata/to-sensor_Zenith_Angle")
    # Reflectance data path has been stable; still keep a check in case of older names.
    refl_ok = _has(h5, "Reflectance/Reflectance_Data")
    return (legacy_caps or legacy_angles) and refl_ok


def resolve_neon_paths(h5: h5py.File) -> NeonPaths:
    """
    Return canonical dataset paths that work for both legacy and modern files.
    """
    legacy = detect_legacy_neon_schema(h5)
    if legacy:
        return NeonPaths(
            reflectance="Reflectance/Reflectance_Data",
            wavelength="Reflectance/Metadata/Spectral_Data/Wavelength",
            fwhm="Reflectance/Metadata/Spectral_Data/FWHM",
            solar_zenith="Reflectance/Metadata/to-sun_Zenith_Angle",
            sensor_zenith="Reflectance/Metadata/to-sensor_Zenith_Angle",
        )
    else:
        return NeonPaths(
            reflectance="Reflectance/Reflectance_Data",
            wavelength="Reflectance/Metadata/Spectral_Data/wavelength",
            fwhm="Reflectance/Metadata/Spectral_Data/fwhm",
            solar_zenith="Reflectance/Metadata/to-sun_zenith_angle",
            sensor_zenith="Reflectance/Metadata/to-sensor_zenith_angle",
        )
