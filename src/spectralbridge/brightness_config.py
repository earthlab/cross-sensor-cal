from __future__ import annotations

import json
from typing import Dict

import importlib.resources as resources


def load_brightness_coefficients(
    system_pair: str = "landsat_to_micasense",
) -> Dict[int, float]:
    """Load brightness coefficients for a given system pair.

    Parameters
    ----------
    system_pair : str
        Key identifying the pair of systems, e.g. "landsat_to_micasense".

    Returns
    -------
    dict[int, float]
        Mapping from 1-based band index to brightness coefficient (percent).

    Notes
    -----
    - Values are stored in percent (e.g., -7.3959 means "reduce by 7.3959%").
    """
    filename = f"{system_pair}.json"
    with resources.files("spectralbridge.data.brightness").joinpath(filename).open(
        "r", encoding="utf-8"
    ) as f:
        cfg = json.load(f)

    bands = cfg.get("bands", {})
    return {int(k): float(v) for k, v in bands.items()}
