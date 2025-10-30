from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from pyproj import Transformer


@dataclass
class GeoContext:
    crs_epsg: Optional[int]
    a: float  # pixel width
    e: float  # pixel height (negative for north-up)
    c: float  # origin x (upper-left corner)
    f: float  # origin y (upper-left corner)


def _get_val(block: str, key: str) -> Optional[str]:
    m = re.search(rf'^{key}\s*=\s*(.+)$', block, flags=re.I | re.M)
    return m.group(1).strip() if m else None


def _parse_map_info(block: str) -> Optional[Tuple[float, float, float, float]]:
    m = re.search(r"map info\s*=\s*\{(.+?)\}", block, flags=re.I | re.S)
    if not m:
        return None
    parts = [p.strip().strip("\"'") for p in m.group(1).split(",")]
    if len(parts) < 7:
        return None
    try:
        u_left = float(parts[3])
        v_top = float(parts[4])
        x_res = float(parts[5])
        y_res = float(parts[6])
        return (x_res, y_res, u_left, v_top)
    except Exception:
        return None


def _guess_epsg(block: str) -> Optional[int]:
    for k in ("projection info", "coordinate system string"):
        s = _get_val(block, k)
        if s and "epsg" in s.lower():
            m = re.search(r"epsg[:\s]*([0-9]{4,5})", s, re.I)
            if m:
                return int(m.group(1))
    m = re.search(r"map info\s*=\s*\{(.+?)\}", block, flags=re.I | re.S)
    if m:
        parts = [p.strip().strip("\"'") for p in m.group(1).split(",")]
        if len(parts) >= 9:
            try:
                zone = int(parts[7])
                hemi = parts[8].lower()
                north = ("north" in hemi) or (hemi == "n")
                return int(f"32{6 if north else 7}{zone:02d}")  # EPSG 326## or 327##
            except Exception:
                pass
    return None


def load_geo_from_hdr(hdr_path: Path) -> Optional[GeoContext]:
    if not hdr_path or not Path(hdr_path).exists():
        return None
    text = Path(hdr_path).read_text(encoding="utf-8", errors="ignore")
    mi = _parse_map_info(text)
    if not mi:
        return None
    x_res, y_res, u_left, v_top = mi
    epsg = _guess_epsg(text)
    a = float(x_res)
    e = -abs(float(y_res))  # north-up
    return GeoContext(crs_epsg=epsg, a=a, e=e, c=float(u_left), f=float(v_top))


def add_lonlat_inplace(df, geo: Optional[GeoContext]):
    """Fill x,y from row/col + affine if needed; then lon,lat from x,y + CRS."""
    if geo is None:
        return df
    if ("x" not in df or "y" not in df) and {"row", "col"} <= set(df.columns):
        df["x"] = df["col"] * geo.a + geo.c + geo.a / 2.0
        df["y"] = df["row"] * geo.e + geo.f + geo.e / 2.0
    if ("lon" not in df or "lat" not in df) and {"x", "y"} <= set(df.columns) and geo.crs_epsg:
        transformer = Transformer.from_crs(geo.crs_epsg, 4326, always_xy=True)
        lon, lat = transformer.transform(df["x"].to_numpy(), df["y"].to_numpy())
        df["lon"] = lon
        df["lat"] = lat
    return df


def write_parquet_with_lonlat(df_long, out_parquet: Path, hdr_path: Optional[Path]):
    """Write a dataframe to parquet after enriching lon/lat from an ENVI header."""
    geo = None
    if hdr_path:
        hdr_path = Path(hdr_path)
        if hdr_path.exists():
            geo = load_geo_from_hdr(hdr_path)
    df_long = add_lonlat_inplace(df_long, geo)
    df_long.to_parquet(out_parquet, index=False)
    return out_parquet
