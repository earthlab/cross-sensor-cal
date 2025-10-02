"""Public package interface for cross_sensor_cal."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cross_sensor_cal")
except PackageNotFoundError:  # pragma: no cover - version only available when installed
    __version__ = "0.1.0"

from importlib import import_module
from types import ModuleType
from typing import Any

_PUBLIC_ATTRS: dict[str, tuple[str, str]] = {
    "apply_offset_to_envi": ("topo_and_brdf_correction", "apply_offset_to_envi"),
    "control_function_for_extraction": ("polygon_extraction", "control_function_for_extraction"),
    "download_neon_flight_lines": ("envi_download", "download_neon_flight_lines"),
    "flight_lines_to_envi": ("neon_to_envi", "flight_lines_to_envi"),
    "generate_config_json": ("topo_and_brdf_correction", "generate_config_json"),
    "generate_file_move_list": ("file_sort", "generate_file_move_list"),
    "mask_raster_with_polygons": ("mask_raster", "mask_raster_with_polygons"),
    "neon_to_envi": ("neon_to_envi", "neon_to_envi"),
    "resample": ("convolution_resample", "resample"),
    "topo_and_brdf_correction": ("topo_and_brdf_correction", "topo_and_brdf_correction"),
    "translate_to_other_sensors": ("standard_resample", "translate_to_other_sensors"),
}


def __getattr__(name: str) -> Any:
    if name in _PUBLIC_ATTRS:
        module_name, attr_name = _PUBLIC_ATTRS[name]
        module: ModuleType = import_module(f".{module_name}", __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PUBLIC_ATTRS.keys()))


__all__ = ["__version__", *_PUBLIC_ATTRS.keys()]
