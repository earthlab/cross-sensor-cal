from __future__ import annotations

import re
import numpy as np
from typing import Optional, Dict, Any


def _parse_numeric_list(value: str) -> np.ndarray:
    s = value.strip().strip("{}[]()")
    parts = re.split(r"[,\s]+", s)
    vals = []
    for p in parts:
        if not p:
            continue
        p = p.replace("nm", "").replace("nanometers", "").replace("nanometres", "")
        try:
            vals.append(float(p))
        except ValueError:
            m = re.search(r"(\d+(?:\.\d+)?)", p)
            if m:
                vals.append(float(m.group(1)))
    return np.array(vals, dtype=float) if vals else np.array([], dtype=float)


def wavelengths_from_hdr(hdr: Dict[str, Any], sensor_default: Optional[np.ndarray] = None) -> tuple[np.ndarray, str]:
    keys = {k.lower(): k for k in hdr.keys()}
    wl_key = next((keys[k] for k in keys if k in ("wavelength", "wavelengths", "band centers")), None)
    if wl_key:
        wl = hdr[wl_key]
        if isinstance(wl, str):
            arr = _parse_numeric_list(wl)
        else:
            # Handle list/array - filter out non-numeric values
            try:
                arr = np.array(wl, dtype=float)
                # Check if we got all NaN values (e.g., from non-numeric strings)
                if arr.size > 0 and not np.any(np.isfinite(arr)):
                    # All values are NaN, return empty array
                    arr = np.array([], dtype=float)
            except (ValueError, TypeError):
                # If conversion fails, return empty array
                arr = np.array([], dtype=float)
        return arr, "header"
    if sensor_default is not None and sensor_default.size > 0:
        return sensor_default.astype(float), "sensor_default"
    return np.array([], dtype=float), "absent"


__all__ = ["_parse_numeric_list", "wavelengths_from_hdr"]
