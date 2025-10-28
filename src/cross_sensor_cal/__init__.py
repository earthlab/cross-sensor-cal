"""Cross-Sensor Calibration public package surface."""
from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - version metadata absent in editable installs
    __version__ = version("cross_sensor_cal")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = [
    "__version__",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy import helper
    if name in {"pipeline", "brdf_topo", "convolution"}:
        module = import_module(f"cross_sensor_cal.{name}")
        globals()[name] = module
        __all__.append(name)
        return module
    raise AttributeError(f"module 'cross_sensor_cal' has no attribute '{name}'")
