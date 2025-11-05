from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import List, Optional, Tuple

from cross_sensor_cal.exports.geo_utils import (
    GeoContext,
    add_lonlat_inplace,
    write_parquet_with_lonlat,
)


class _LazyModule:
    """Delay importing optional heavy dependencies until they are accessed."""

    def __init__(self, module_path: str, *, attr: str | None = None, install_hint: str):
        self._module_path = module_path
        self._attr = attr
        self._install_hint = install_hint
        self._cached = None

    def _load(self):
        if self._cached is None:
            try:
                module = import_module(self._module_path)
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
                raise ModuleNotFoundError(
                    "Optional dependency '{name}' is required for export schema utilities. "
                    "Install it via {hint}.".format(name=self._module_path, hint=self._install_hint)
                ) from exc
            self._cached = getattr(module, self._attr) if self._attr else module
        return self._cached

    def __getattr__(self, name):
        return getattr(self._load(), name)


np = _LazyModule("numpy", install_hint="`pip install numpy`")
pd = _LazyModule("pandas", install_hint="`pip install pandas`")
Transformer = _LazyModule("pyproj", attr="Transformer", install_hint="`pip install pyproj`")

# --- Known band centers (nm) for resampled products (adjust if your repo stores these elsewhere) ---
SENSOR_WAVELENGTHS_NM = {
    "landsat_tm":    [485, 570, 660, 830, 1650, 2210],                       # TM 1–5,7 (omit TIR)
    "landsat_etm+":  [485, 560, 660, 835, 1650, 2220],                       # ETM+ 1–5,7
    "landsat_oli":   [443, 482, 561, 655, 865, 1609, 2201],                  # OLI 1–7 (omit cirrus/pan)
    "landsat_oli2":  [443, 482, 561, 655, 865, 1610, 2201],                  # OLI-2 approx
    "micasense":     [475, 560, 668, 717, 840],                              # tweak to your rig
    "micasense_to_match_tm_etm+":  [485, 560, 660, 835, 1650, 2220],
    "micasense_to_match_oli_oli2": [482, 561, 655, 865, 1609, 2201],
}

def infer_stage_from_name(name: str) -> str:
    n = name.lower()
    if "_brdfandtopo_corrected" in n:
        return "corr"
    if any(k in n for k in ("landsat_tm","landsat_etm+","landsat_oli2","landsat_oli","micasense")):
        # return the sensor key as stage (short)
        for k in SENSOR_WAVELENGTHS_NM:
            if k in n:
                return k.replace("landsat_", "oli" if "oli" in k else k.split("_")[-1])
        return "sensor"
    return "raw"

def sort_and_rename_spectral_columns(df: pd.DataFrame, stage_key: str, wavelengths_nm: List[int]) -> pd.DataFrame:
    """
    Rebuild DataFrame with non-spectral first, then spectral columns sorted by wavelength
    and renamed as: <stage>_b###_wl####nm
    """
    # detect spectral cols (anything with 'wl' + digits)
    spec_cols = [c for c in df.columns if re.search(r"wl(\d+)", c)]
    other = [c for c in df.columns if c not in spec_cols]

    if not wavelengths_nm:
        # try to extract from column names
        wl_map = {}
        for c in spec_cols:
            m = re.search(r"wl(\d+)", c)
            wl_map[c] = int(m.group(1)) if m else None
        spec_sorted = sorted(spec_cols, key=lambda c: wl_map[c] or 0)
        renamed = {c: f"{stage_key}_b{idx:03d}_wl{(wl_map[c] or 0):04d}nm" for idx, c in enumerate(spec_sorted, 1)}
        return df[other + spec_sorted].rename(columns=renamed)

    # wavelengths provided: ensure equal length
    if len(wavelengths_nm) != len(spec_cols):
        # best-effort: sort by any existing wl in name; otherwise keep order
        order = np.argsort(np.array([int(re.search(r"wl(\d+)", c).group(1)) if re.search(r"wl(\d+)", c) else 10**9 for c in spec_cols]))
        spec_sorted = [spec_cols[i] for i in order]
        wls_sorted = [wavelengths_nm[i] if i < len(wavelengths_nm) else 0 for i in range(len(spec_sorted))]
    else:
        spec_sorted = list(spec_cols)
        wls_sorted = list(wavelengths_nm)

    # sort by wavelengths ascending
    pairs = sorted(zip(spec_sorted, wls_sorted), key=lambda t: t[1])
    spec_sorted, wls_sorted = zip(*pairs)
    renamed = {c: f"{stage_key}_b{idx:03d}_wl{wl:04d}nm" for idx, (c, wl) in enumerate(zip(spec_sorted, wls_sorted), 1)}
    return df[other + list(spec_sorted)].rename(columns=renamed)

def parse_envi_wavelengths_nm(hdr_text: str) -> Optional[List[int]]:
    """
    Parse ENVI header text for numeric wavelengths, return nm ints or None.
    """
    text = hdr_text.lower()
    m = re.search(r"wavelength(?:s)?\s*=\s*\{(.+?)\}", text, re.S)
    if not m:
        return None
    raw = m.group(1)
    parts = [p.strip() for p in re.split(r",\s*", raw) if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            return None
    units = "nm"
    mu = re.search(r"wavelength(?:\s+units)?\s*=\s*([^\n\r]+)", text)
    if mu:
        units = mu.group(1).lower().strip()
    if any(u in units for u in ["micro","µ","um"]):
        vals = [int(round(v*1000)) for v in vals]
    else:
        vals = [int(round(v)) for v in vals]
    return vals

def compute_lonlat(xs: np.ndarray, ys: np.ndarray, src_epsg: int | str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform planar x,y to lon,lat. Returns (lon, lat).
    """
    transformer = Transformer.from_crs(src_epsg, 4326, always_xy=True)
    lon, lat = transformer.transform(xs, ys)
    return np.asarray(lon), np.asarray(lat)

def _maybe_epsg_int(value) -> Optional[int]:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"(\d{4,5})", value)
        if m:
            return int(m.group(1))
    return None


def _geo_from_transform(transform, crs_epsg: int | str) -> Optional[GeoContext]:
    if transform is None:
        return None
    try:
        a = float(getattr(transform, "a"))
        e = float(getattr(transform, "e"))
        c = float(getattr(transform, "c"))
        f = float(getattr(transform, "f"))
    except (AttributeError, TypeError, ValueError):
        return None
    return GeoContext(crs_epsg=_maybe_epsg_int(crs_epsg), a=a, e=e, c=c, f=f)


def ensure_coord_columns(df: pd.DataFrame, transform, crs_epsg: int | str) -> pd.DataFrame:
    """
    Make sure df has x, y, lon, lat columns. If missing, compute from row/col with affine transform and the CRS.
    """
    need_xy = any(k not in df for k in ("x","y"))
    if need_xy and all(k in df for k in ("row","col")) and transform is not None:
        # x = col * a + x0 ; y = row * e + y0 (for north-up affine)
        a = transform.a
        e = transform.e
        x0 = transform.c
        y0 = transform.f
        df["x"] = df["col"] * a + x0 + a/2.0
        df["y"] = df["row"] * e + y0 + e/2.0

    geo = _geo_from_transform(transform, crs_epsg)
    if geo is not None:
        df = add_lonlat_inplace(df, geo)

    if any(k not in df for k in ("lon", "lat")) and all(k in df for k in ("x", "y")) and crs_epsg:
        lon, lat = compute_lonlat(df["x"].to_numpy(), df["y"].to_numpy(), crs_epsg)
        df["lon"] = lon
        df["lat"] = lat

    return df

def write_parquet_standardized(
    df: pd.DataFrame,
    out_path: Path,
    hdr_path: Path,
    transform=None,
    crs_epsg: int | str = 32613,
):
    """
    Standardized Parquet writer:
      - ensures x,y,lon,lat present
      - ensures spectral columns sorted and renamed as <stage>_b###_wl####nm
    """
    out_path = Path(out_path)
    stage_key = infer_stage_from_name(out_path.name)

    wavelengths_nm: Optional[List[int]] = None
    if hdr_path and Path(hdr_path).exists():
        hdr_text = Path(hdr_path).read_text(encoding="utf-8", errors="ignore")
        wavelengths_nm = parse_envi_wavelengths_nm(hdr_text)

    # If resampled product lacks wavelengths, supply sensor defaults
    if wavelengths_nm is None:
        for sensor_key, centers in SENSOR_WAVELENGTHS_NM.items():
            if sensor_key in out_path.name.lower():
                wavelengths_nm = centers
                break

    # 1) coords
    df = ensure_coord_columns(df, transform=transform, crs_epsg=crs_epsg)
    # 2) spectral
    df = sort_and_rename_spectral_columns(
        df,
        stage_key=stage_key,
        wavelengths_nm=wavelengths_nm or [],
    )

    # Optional sanity checks: must have lon/lat, no all-null spectral
    if "lon" not in df or "lat" not in df:
        raise ValueError(f"Missing lon/lat in parquet export: {out_path}")
    spec_cols = [c for c in df.columns if "_wl" in c]
    if not spec_cols:
        raise ValueError(f"No spectral columns detected for export: {out_path}")

    write_parquet_with_lonlat(df, out_path, hdr_path)
    return out_path
