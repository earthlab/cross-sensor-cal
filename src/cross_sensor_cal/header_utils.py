"""Robust ENVI header helpers for QA metrics."""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np


def _parse_numeric_list(value: str) -> np.ndarray:
    s = value.strip().strip("{}[]()")
    parts = re.split(r"[\,\s]+", s)
    vals = []
    for p in parts:
        if not p:
            continue
        token = p.replace("nm", "").replace("nanometers", "").replace(
            "nanometres", ""
        )
        try:
            vals.append(float(token))
            continue
        except ValueError:
            match = re.search(r"(\d+(?:\.\d+)?)", token)
            if match:
                vals.append(float(match.group(1)))
    return np.array(vals, dtype=float) if vals else np.array([], dtype=float)


def wavelengths_from_hdr(
    hdr: dict,
    sensor_default: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str]:
    # case-insensitive lookup for wavelength metadata
    lowered = {k.lower(): k for k in hdr.keys()}
    wl_key = next(
        (lowered[k] for k in lowered if k in ("wavelength", "wavelengths", "band centers")),
        None,
    )
    unit_key = next(
        (
            lowered[k]
            for k in lowered
            if k in ("wavelength units", "wavelength_unit", "units")
        ),
        None,
    )
    unit = hdr.get(unit_key) if unit_key else None

    if wl_key:
        wl = hdr[wl_key]
        if isinstance(wl, str):
            arr = _parse_numeric_list(wl)
        else:
            try:
                arr = np.array(wl, dtype=float)
            except Exception:  # pragma: no cover - fall back to parsing
                arr = _parse_numeric_list(str(wl))
        return arr, "header" if arr.size else "absent"

    if sensor_default is not None and sensor_default.size > 0:
        return sensor_default.astype(float), "sensor_default"

    return np.array([], dtype=float), "absent"


__all__ = ["_parse_numeric_list", "wavelengths_from_hdr"]

