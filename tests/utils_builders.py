from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
try:
    import numpy as np
except Exception:
    np = None

try:
    import rasterio
    from rasterio.transform import from_origin
except Exception:
    rasterio = None

ENVI_HDR_MIN = "ENVI\nsamples = 16\nlines   = 16\nbands   = 3\ndata type = 4\ninterleave = bsq\nbyte order = 0\n"

def fake_neon_name(domain="D13", site="NIWO", tile="L020-1", date="20230815", suffix="_reflectance_envi.img") -> str:
    return f"NEON_{domain}_{site}_DP1_{tile}_{date}{suffix}"

def make_tiny_envi(tmp_path: Path, name: str | None = None) -> tuple[Path, Path]:
    """
    Writes a tiny ENVI pair (.img + .hdr). The .img is empty bytes; hdr is minimal text.
    Enough to test path logic without needing real ENVI content.
    """
    if name is None:
        name = fake_neon_name()
    img = tmp_path / name
    hdr = img.with_suffix(".hdr")
    img.write_bytes(b"\x00" * 64)  # placeholder
    hdr.write_text(ENVI_HDR_MIN)
    return img, hdr

def make_tiny_geotiff(tmp_path: Path, fname: str = "tiny.tif") -> Path:
    """
    If rasterio is present, writes a 16x16 float32 GeoTIFF with deterministic values.
    Otherwise, writes a placeholder file.
    """
    path = tmp_path / fname
    if rasterio is None or np is None:
        path.write_bytes(b"\x00" * 64)
        return path

    arr = np.arange(16*16, dtype=np.float32).reshape(16,16) / (16*16)
    transform = from_origin(500000.0, 4400000.0, 10.0, 10.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=16, width=16, count=1, dtype="float32",
        crs="EPSG:32613", transform=transform, tiled=True
    ) as dst:
        dst.write(arr, 1)
    return path

@dataclass
class FakeFile:
    kind: str = "generic"
    is_masked: bool = False
