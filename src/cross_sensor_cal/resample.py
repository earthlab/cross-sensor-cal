"""Spectral resampling utilities for cross-sensor calibration workflows."""

from typing import Dict, List

import numpy as np


def resample_chunk_to_sensor(
    chunk: np.ndarray,
    wavelengths: np.ndarray,
    sensor_srf: Dict[str, np.ndarray],
) -> np.ndarray:
    """Resample a hyperspectral chunk to target sensor bands using SRFs.

    Parameters
    ----------
    chunk
        Hyperspectral reflectance cube with shape ``(y, x, bands)``.
    wavelengths
        Wavelength grid aligned to the last axis of ``chunk``.
    sensor_srf
        Mapping of sensor band name to spectral response sampled on ``wavelengths``.

    Returns
    -------
    np.ndarray
        Resampled cube with shape ``(y, x, len(sensor_srf))`` and dtype ``float32``.
    """

    if chunk.ndim != 3:
        raise ValueError(
            "chunk must be a 3D array with shape (y, x, bands); "
            f"received shape {chunk.shape}"
        )

    if wavelengths.ndim != 1:
        raise ValueError(
            "wavelengths must be a 1D array with shape (bands,); "
            f"received shape {wavelengths.shape}"
        )

    chunk_float = np.asarray(chunk, dtype=np.float32)
    wavelength_array = np.asarray(wavelengths)

    n_spatial_y, n_spatial_x, n_bands = chunk_float.shape

    if wavelength_array.shape[0] != n_bands:
        raise ValueError(
            "wavelengths must have the same number of elements as chunk bands; "
            f"got {wavelength_array.shape[0]} wavelengths for {n_bands} bands"
        )

    band_keys: List[str] = list(sensor_srf.keys())
    n_sensor_bands = len(band_keys)

    for key in band_keys:
        srf = np.asarray(sensor_srf[key])
        if srf.ndim != 1 or srf.shape[0] != n_bands:
            raise ValueError(
                "Each spectral response function must be a 1D array matching the number "
                "of hyperspectral bands; "
                f"band '{key}' has shape {srf.shape} with {srf.ndim} dims"
            )

    if n_sensor_bands == 0:
        return np.empty((n_spatial_y, n_spatial_x, 0), dtype=np.float32)

    spectra_2d = chunk_float.reshape(-1, n_bands)
    resampled = np.empty((spectra_2d.shape[0], n_sensor_bands), dtype=np.float32)

    for idx, key in enumerate(band_keys):
        srf = np.asarray(sensor_srf[key], dtype=np.float32)
        weight_sum = float(np.sum(srf, dtype=np.float64))
        weights = srf / (weight_sum + 1e-12)
        resampled[:, idx] = spectra_2d @ weights

    return resampled.reshape(n_spatial_y, n_spatial_x, n_sensor_bands)
