"""Cross-sensor calibration utility functions."""

__all__ = [
    "plot_envi_band",
    "plot_envi_rgb",
    "RoiResult",
    "extract_roi_spectra",
    "plot_roi_spectral_comparison",
]

try:  # pragma: no cover - optional heavy deps
    from .envi_visualization import plot_envi_band, plot_envi_rgb
    from .roi_spectral_comparison import (
        RoiResult,
        extract_roi_spectra,
        plot_roi_spectral_comparison,
    )
except Exception:  # Allow importing lightweight modules without optional deps
    plot_envi_band = plot_envi_rgb = None
    RoiResult = extract_roi_spectra = plot_roi_spectral_comparison = None
