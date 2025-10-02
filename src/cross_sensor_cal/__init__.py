"""Public package interface for cross_sensor_cal."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cross_sensor_cal")
except PackageNotFoundError:  # pragma: no cover - version only available when installed
    __version__ = "0.0.0"

from .convolution_resample import resample
from .envi_download import download_neon_flight_lines
from .file_sort import generate_file_move_list
from .mask_raster import mask_raster_with_polygons
from .neon_to_envi import flight_lines_to_envi, neon_to_envi
from .polygon_extraction import control_function_for_extraction
from .standard_resample import translate_to_other_sensors
from .topo_and_brdf_correction import (
    apply_offset_to_envi,
    generate_config_json,
    topo_and_brdf_correction,
)

__all__ = [
    "__version__",
    "apply_offset_to_envi",
    "control_function_for_extraction",
    "download_neon_flight_lines",
    "flight_lines_to_envi",
    "generate_config_json",
    "generate_file_move_list",
    "mask_raster_with_polygons",
    "neon_to_envi",
    "resample",
    "topo_and_brdf_correction",
    "translate_to_other_sensors",
]
