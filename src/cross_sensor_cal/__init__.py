"""Cross-Sensor Calibration public package surface."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - version metadata absent in editable installs
    __version__ = version("cross_sensor_cal")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = ["__version__"]
