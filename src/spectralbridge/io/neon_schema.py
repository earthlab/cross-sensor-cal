from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np


@dataclass
class NeonResolved:
    """References to key NEON HDF5 datasets and legacy flag."""

    ds_reflectance: h5py.Dataset
    ds_wavelength: h5py.Dataset
    ds_fwhm: Optional[h5py.Dataset]
    ds_sun_zen: Optional[h5py.Dataset]
    ds_sensor_zen: Optional[h5py.Dataset]
    is_legacy: bool


def _has(h5: h5py.File | h5py.Group, path: str) -> bool:
    try:
        h5[path]
        return True
    except KeyError:
        return False


def detect_legacy(h5: h5py.File | h5py.Group) -> bool:
    """Return ``True`` when the legacy (pre-2021) NEON schema is detected."""

    caps_spec = _has(h5, "Reflectance/Metadata/Spectral_Data/Wavelength") or _has(
        h5, "Reflectance/Metadata/Spectral_Data/FWHM"
    )
    caps_angles = _has(h5, "Reflectance/Metadata/to-sun_Zenith_Angle") or _has(
        h5, "Reflectance/Metadata/to-sensor_Zenith_Angle"
    )
    return (caps_spec or caps_angles) and _has(h5, "Reflectance/Reflectance_Data")


def resolve(h5: h5py.File | h5py.Group) -> NeonResolved:
    """Return dataset handles for reflectance, wavelength, FWHM, and angles."""

    legacy = detect_legacy(h5)
    reflectance = h5["Reflectance/Reflectance_Data"]
    if legacy:
        wavelength = h5["Reflectance/Metadata/Spectral_Data/Wavelength"]
        fwhm = (
            h5["Reflectance/Metadata/Spectral_Data/FWHM"]
            if _has(h5, "Reflectance/Metadata/Spectral_Data/FWHM")
            else None
        )
        sun = (
            h5["Reflectance/Metadata/to-sun_Zenith_Angle"]
            if _has(h5, "Reflectance/Metadata/to-sun_Zenith_Angle")
            else None
        )
        sensor = (
            h5["Reflectance/Metadata/to-sensor_Zenith_Angle"]
            if _has(h5, "Reflectance/Metadata/to-sensor_Zenith_Angle")
            else None
        )
    else:
        wavelength = h5["Reflectance/Metadata/Spectral_Data/wavelength"]
        fwhm = (
            h5["Reflectance/Metadata/Spectral_Data/fwhm"]
            if _has(h5, "Reflectance/Metadata/Spectral_Data/fwhm")
            else None
        )
        sun = (
            h5["Reflectance/Metadata/to-sun_zenith_angle"]
            if _has(h5, "Reflectance/Metadata/to-sun_zenith_angle")
            else None
        )
        sensor = (
            h5["Reflectance/Metadata/to-sensor_zenith_angle"]
            if _has(h5, "Reflectance/Metadata/to-sensor_zenith_angle")
            else None
        )

    return NeonResolved(reflectance, wavelength, fwhm, sun, sensor, legacy)


def canonical_vectors(
    nr: NeonResolved,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load spectral/angle vectors in canonical float32 units."""

    wavelength = np.asarray(nr.ds_wavelength[...], dtype=np.float32)
    fwhm = (
        np.asarray(nr.ds_fwhm[...], dtype=np.float32)
        if nr.ds_fwhm is not None
        else None
    )

    sun = (
        np.asarray(nr.ds_sun_zen[...], dtype=np.float32)
        if nr.ds_sun_zen is not None
        else None
    )
    sensor = (
        np.asarray(nr.ds_sensor_zen[...], dtype=np.float32)
        if nr.ds_sensor_zen is not None
        else None
    )

    return wavelength, fwhm, sun, sensor


__all__ = ["NeonResolved", "detect_legacy", "resolve", "canonical_vectors"]
