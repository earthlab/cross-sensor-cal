"""Lightweight ENVI header parsing and cube reading utilities."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np


def _coerce_scalar(value: str):
    token = value.strip().strip('"').strip("'")
    if not token:
        return ""
    lowered = token.lower()
    if lowered == "nan":
        return float("nan")
    for caster in (int, float):
        try:
            return caster(token)
        except (TypeError, ValueError):
            continue
    return token


def _parse_wavelength_list(val: str | None) -> list[float] | None:
    if val is None:
        return None
    s = val.strip()
    if not s:
        return None
    s = s.strip("{}[]()")
    s = s.replace("\n", " ")
    parts = [part.strip() for part in s.replace("\t", " ").split(",")]
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        tokens.extend(token for token in part.split() if token)
    nums: list[float] = []
    for token in tokens:
        try:
            nums.append(float(token))
        except ValueError:
            return None
    return nums if nums else None


def _split_envi_block(value: str) -> list[str]:
    inner = value.strip()
    if inner.startswith("{"):
        inner = inner[1:]
    if inner.endswith("}"):
        inner = inner[:-1]
    inner = inner.replace("\n", " ")
    tokens = [token.strip() for token in inner.split(",") if token.strip()]
    return tokens


def _parse_envi_header_tolerant(hdr_path: Path) -> dict[str, Any]:
    if not hdr_path.exists():
        raise FileNotFoundError(hdr_path)

    raw_entries: dict[str, str] = {}
    collecting_key: str | None = None
    collecting_value: list[str] = []

    with hdr_path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            stripped = raw_line.strip()
            if not stripped or stripped.upper() == "ENVI":
                continue

            if collecting_key is not None:
                collecting_value.append(stripped)
                if "}" in stripped:
                    value = " ".join(collecting_value)
                    raw_entries[collecting_key] = value
                    collecting_key = None
                    collecting_value = []
                continue

            if "=" not in stripped:
                continue

            key_part, value_part = stripped.split("=", 1)
            key = key_part.strip().lower()
            value = value_part.strip()

            if value.startswith("{") and "}" not in value:
                collecting_key = key
                collecting_value = [value]
                continue

            raw_entries[key] = value

    list_float_keys = {"fwhm"}
    list_string_keys = {"map info", "band names"}
    int_scalar_keys = {"samples", "lines", "bands", "data type", "byte order"}

    processed: dict[str, Any] = {}
    for key, raw_value in raw_entries.items():
        if raw_value.startswith("{") and raw_value.endswith("}"):
            if key == "wavelength":
                processed[key] = _parse_wavelength_list(raw_value)
                continue

            tokens = _split_envi_block(raw_value)
            if key in list_float_keys:
                processed[key] = [float(token) for token in tokens]
            elif key in list_string_keys:
                processed[key] = [token.strip('"').strip("'") for token in tokens]
            else:
                processed[key] = [
                    _coerce_scalar(token.strip('"').strip("'")) for token in tokens
                ]
            continue

        cleaned = raw_value.strip().strip('"').strip("'")
        if key in int_scalar_keys:
            try:
                processed[key] = int(cleaned)
            except ValueError as exc:  # pragma: no cover - malformed header unexpected
                raise RuntimeError(
                    f"Header value for '{key}' is not an integer"
                ) from exc
            continue

        processed[key] = _coerce_scalar(cleaned)

    if "wavelength" not in processed:
        processed["wavelength"] = None

    return processed


def hdr_to_dict(hdr_path: Path) -> dict[str, Any]:
    try:
        header = _parse_envi_header(hdr_path)
    except RuntimeError as exc:
        msg = str(exc)
        if "Could not parse numeric list for 'wavelength'" not in msg:
            raise
        header = _parse_envi_header_tolerant(hdr_path)

    wavelength = header.get("wavelength")
    if wavelength is None:
        header["wavelength"] = None
    elif isinstance(wavelength, list) and not wavelength:
        header["wavelength"] = None

    return header


def band_axis_from_header(arr: np.ndarray, hdr: dict[str, Any]) -> int:
    nb = int(hdr.get("bands", 0) or 0)
    if nb <= 0:
        raise ValueError("Header has no valid 'bands' value.")

    candidates = [i for i, dim in enumerate(arr.shape) if dim == nb]
    if candidates:
        return candidates[0]
    if arr.ndim == 3:
        return 2
    return 0


def wavelength_array(header_dict: dict[str, Any]) -> np.ndarray | None:
    wavs = header_dict.get("wavelength")
    if wavs is None:
        return None
    try:
        wavs_arr = np.asarray(wavs, dtype=float)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if wavs_arr.size == 0:
        return None
    return wavs_arr


_ENVI_DTYPE_MAP: dict[int, np.dtype] = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
}


def _dtype_from_header(header: dict[str, Any], hdr_path: Path) -> np.dtype:
    try:
        code = int(header["data type"])
    except KeyError as exc:  # pragma: no cover - malformed header unexpected
        raise RuntimeError(f"ENVI header missing 'data type': {hdr_path}") from exc
    try:
        base_dtype = np.dtype(_ENVI_DTYPE_MAP[code])
    except KeyError as exc:  # pragma: no cover - unsupported dtype unexpected
        raise RuntimeError(f"Unsupported ENVI data type {code} for {hdr_path}") from exc

    byte_order = header.get("byte order")
    if byte_order is None:
        return base_dtype

    try:
        bo_int = int(byte_order)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Invalid ENVI 'byte order' value {byte_order!r} for {hdr_path}"
        ) from exc

    if base_dtype.byteorder == "|":
        return base_dtype

    if bo_int == 0:
        return base_dtype.newbyteorder("<")
    if bo_int == 1:
        return base_dtype.newbyteorder(">")

    raise RuntimeError(f"Unsupported ENVI byte order {bo_int} for {hdr_path}")


def memmap_bsq(img_path: Path, header: dict[str, Any]) -> np.memmap:
    try:
        samples = int(header["samples"])
        lines = int(header["lines"])
        bands = int(header["bands"])
    except KeyError as exc:  # pragma: no cover - malformed header unexpected
        raise RuntimeError(f"ENVI header missing dimension {exc.args[0]!r}: {img_path}") from exc

    interleave = str(header.get("interleave", "")).lower()
    if interleave != "bsq":
        raise RuntimeError(f"Only BSQ interleave supported for ENVI reader: {img_path}")

    dtype = _dtype_from_header(header, img_path.with_suffix(".hdr"))
    shape = (bands, lines, samples)
    return np.memmap(img_path, mode="r", dtype=dtype, shape=shape, order="C")


def read_envi_cube(img_path: Path, header: dict[str, Any] | None = None) -> np.ndarray:
    hdr_path = img_path.with_suffix(".hdr")
    if header is None:
        header = hdr_to_dict(hdr_path)
    else:
        header = dict(header)

    interleave = str(header.get("interleave", "")).lower()
    if interleave and interleave != "bsq":
        raise RuntimeError(f"Only BSQ interleave supported for ENVI reader: {img_path}")

    if not img_path.exists() or not hdr_path.exists():
        missing = hdr_path if not hdr_path.exists() else img_path
        raise FileNotFoundError(f"ENVI resource missing: {missing}")

    cube = memmap_bsq(img_path, header)

    if not np.issubdtype(cube.dtype, np.floating):
        return cube.astype(np.float32)

    return cube.astype(np.float32, copy=False)


def to_unitless_reflectance(arr: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(arr))
    if np.isnan(med):
        return arr
    return arr / 10000.0 if med > 1.5 else arr


__all__ = [
    "hdr_to_dict",
    "band_axis_from_header",
    "wavelength_array",
    "read_envi_cube",
    "memmap_bsq",
    "to_unitless_reflectance",
]


def _parse_envi_header(hdr_path: Path) -> dict[str, Any]:
    if not hdr_path.exists():
        raise FileNotFoundError(hdr_path)

    raw_entries: dict[str, str] = {}
    collecting_key: str | None = None
    collecting_value: list[str] = []

    with hdr_path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            stripped = raw_line.strip()
            if not stripped or stripped.upper() == "ENVI":
                continue

            if collecting_key is not None:
                collecting_value.append(stripped)
                if "}" in stripped:
                    value = " ".join(collecting_value)
                    raw_entries[collecting_key] = value
                    collecting_key = None
                    collecting_value = []
                continue

            if "=" not in stripped:
                continue

            key_part, value_part = stripped.split("=", 1)
            key = key_part.strip().lower()
            value = value_part.strip()

            if value.startswith("{") and "}" not in value:
                collecting_key = key
                collecting_value = [value]
                continue

            raw_entries[key] = value

    list_float_keys = {"fwhm"}
    list_string_keys = {"map info", "band names"}
    int_scalar_keys = {"samples", "lines", "bands", "data type", "byte order"}

    processed: dict[str, Any] = {}
    for key, raw_value in raw_entries.items():
        if raw_value.startswith("{") and raw_value.endswith("}"):
            if key == "wavelength":
                processed[key] = _parse_wavelength_list(raw_value)
                continue

            tokens = _split_envi_block(raw_value)
            if key in list_float_keys:
                processed[key] = [float(token) for token in tokens]
            elif key in list_string_keys:
                processed[key] = [token.strip('"').strip("'") for token in tokens]
            else:
                processed[key] = [
                    _coerce_scalar(token.strip('"').strip("'")) for token in tokens
                ]
            continue

        cleaned = raw_value.strip().strip('"').strip("'")
        if key in int_scalar_keys:
            try:
                processed[key] = int(cleaned)
            except ValueError as exc:  # pragma: no cover - malformed header unexpected
                raise RuntimeError(
                    f"Header value for '{key}' is not an integer"
                ) from exc
            continue

        processed[key] = _coerce_scalar(cleaned)

    if "wavelength" not in processed:
        processed["wavelength"] = None

    return processed
