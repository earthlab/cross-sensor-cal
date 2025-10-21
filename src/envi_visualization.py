"""Utility functions for visualising ENVI hyperspectral datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _require_spectral():
    """Return :func:`spectral.open_image`, importing ``spectral`` lazily."""

    try:
        from spectral import open_image  # type: ignore import-error
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError(
            "The 'spectral' package is required for ENVI visualization functions.\n"
            "Install it with: pip install spectral"
        ) from exc
    return open_image


def _resolve_hdr_path(img_path: Path | str) -> Path:
    """Return the path to the ENVI header file for ``img_path``.

    Parameters
    ----------
    img_path:
        Path to the ENVI ``.img`` file or the header file itself.

    Returns
    -------
    pathlib.Path
        Path to the ENVI header (``.hdr``) file.
    """

    img_path = Path(img_path)
    if img_path.suffix.lower() == ".hdr":
        return img_path
    if img_path.suffix.lower() != ".img":
        raise ValueError(
            "img_path must point to an ENVI .img file or its .hdr counterpart"
        )
    return img_path.with_suffix(".hdr")


def plot_envi_band(img_path: Path | str, band_index: int = 0, cmap: str = "gray") -> None:
    """Plot a single band from an ENVI ``.img`` file.

    Parameters
    ----------
    img_path:
        Path to the ENVI ``.img`` file (expects the corresponding ``.hdr`` next to it).
    band_index:
        Index of the band to plot (0-based).
    cmap:
        Name of the Matplotlib colourmap to use for the display.
    """

    hdr_path = _resolve_hdr_path(img_path)
    open_image = _require_spectral()
    img = open_image(str(hdr_path))

    data = np.asarray(img[:, :, band_index], dtype=float)

    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap=cmap)
    plt.title(f"ENVI Band {band_index}")
    plt.colorbar(label="Reflectance")
    plt.axis("off")
    plt.show()


def plot_envi_rgb(
    img_path: Path | str,
    rgb_bands: Iterable[int] = (29, 19, 9),
    stretch: Tuple[float, float] = (2, 98),
) -> None:
    """Plot an RGB composite from an ENVI ``.img`` file.

    Parameters
    ----------
    img_path:
        Path to the ENVI ``.img`` file (expects the corresponding ``.hdr`` next to it).
    rgb_bands:
        Iterable with three band indices (R, G, B) to composite.
    stretch:
        Percentile stretch (e.g., ``(2, 98)``) for contrast enhancement.
    """

    hdr_path = _resolve_hdr_path(img_path)
    open_image = _require_spectral()
    img = open_image(str(hdr_path))

    band_indices = tuple(rgb_bands)
    if len(band_indices) != 3:
        raise ValueError("rgb_bands must contain exactly three band indices")

    rgb = np.asarray(img[:, :, list(band_indices)], dtype=float)

    p_low, p_high = np.percentile(rgb, stretch, axis=(0, 1))
    rgb = (rgb - p_low) / (p_high - p_low)
    rgb = np.clip(rgb, 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(f"RGB Composite (bands {band_indices})")
    plt.axis("off")
    plt.show()


__all__ = ["plot_envi_band", "plot_envi_rgb"]
