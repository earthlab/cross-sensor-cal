from __future__ import annotations
import re

# String-only file categorization. No filesystem access.
# Returns one of: "Generic","ENVI","HDR","Mask","Resampled","Ancillary","Corrected"

_CANON = re.compile(r"^NEON_[A-Z0-9]+_[A-Z0-9]+_DP1(?:\.\d+\.\d+)?_L\d{3}-\d_\d{8}_")

def classify_name(fname: str) -> str:
    if not fname:
        return "Generic"
    if not _CANON.match(fname):
        return "Generic"

    s = fname.lower()
    # order matters (more specific first)
    if "resampled" in s and "mask" in s:
        return "Mask"
    if "resampled" in s:
        return "Resampled"
    if s.endswith("_reflectance_envi.img"):
        return "ENVI"
    if s.endswith("_reflectance_envi.hdr"):
        return "HDR"
    if "ancillary" in s:
        return "Ancillary"
    if "brdf" in s or "topo" in s or "corrected" in s:
        return "Corrected"
    return "Generic"
