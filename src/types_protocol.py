from __future__ import annotations
from typing import Protocol

class HasKind(Protocol):
    kind: str      # e.g., "envi","envi_hdr","resampled","resampled_mask","ancillary","generic"
    is_masked: bool

def categorize_file(obj: HasKind) -> str:
    # Minimal decision logic driven by attributes, not class identity
    k = getattr(obj, "kind", "generic")
    if k == "resampled_mask":
        return "Mask"
    if k == "resampled":
        return "Resampled"
    if k == "envi_hdr":
        return "HDR"
    if k == "envi":
        return "ENVI"
    if k == "ancillary":
        return "Ancillary"
    if getattr(obj, "is_masked", False):
        return "Mask"
    return "Generic"
