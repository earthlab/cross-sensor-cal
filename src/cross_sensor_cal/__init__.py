"""Cross-Sensor Calibration public package surface."""
from __future__ import annotations

from importlib import import_module

from .brightness_config import load_brightness_coefficients

__version__ = "2.2.0"

__all__ = ["__version__"]


__all__ = sorted(
    set(
        __all__
        + [
            "apply_brightness_correction",
            load_brightness_coefficients.__name__,
        ]
    )
)

def __getattr__(name: str):  # pragma: no cover - thin lazy import helper
    if name == "apply_brightness_correction":
        from .brightness import apply_brightness_correction as _apply_brightness_correction

        globals()[name] = _apply_brightness_correction
        return _apply_brightness_correction
    if name == "pipeline":
        module = import_module("cross_sensor_cal.pipelines.pipeline")
        globals()[name] = module
        __all__.append(name)
        return module
    if name == "brdf_topo":
        module = import_module("cross_sensor_cal.brdf_topo")
        globals()[name] = module
        __all__.append(name)
        return module
    raise AttributeError(f"module 'cross_sensor_cal' has no attribute '{name}'")
