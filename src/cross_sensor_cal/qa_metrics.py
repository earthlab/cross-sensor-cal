from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to reuse shared ENVI utilities without pulling in heavy plotting deps
try:
    from cross_sensor_cal.envi import (
        band_axis_from_header,
        hdr_to_dict,
        read_envi_cube,
        to_unitless_reflectance,
    )
except Exception:
    hdr_to_dict = None
    read_envi_cube = None
    band_axis_from_header = None

    def to_unitless_reflectance(arr: np.ndarray) -> np.ndarray:
        med = float(np.nanmedian(arr))
        return arr / 10000.0 if med > 1.5 else arr

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------
# Threshold definitions
# -----------------------
DEFAULT_THRESHOLDS = {
    "ratio_median_min": 0.5,
    "ratio_median_max": 1.5,
    "pct_lt0_max": 0.01,
    "pct_gt1_max": 0.01,
    "toc_min_slope_reduction": 0.30,
    "toc_min_r2_reduction": 0.30,
    "aspect_min_reduction": 0.30,
    "brdf_cv_min_reduction": 0.30,
    "nbar_rmse_min_improvement": 0.10,
}

VIEW_ZENITH_BINS = np.array([0, 10, 20, 30, 40, 50, 60], dtype=float)

# -----------------------
# Helpers
# -----------------------
def _load_envi(img_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    hdr_path = img_path.with_suffix(".hdr")
    if read_envi_cube is None:
        raise RuntimeError("read_envi_cube() not available. Import qa_plots helpers or add a local ENVI reader.")
    if not img_path.exists() or not hdr_path.exists():
        raise FileNotFoundError(f"ENVI pair missing: {img_path.name} / {hdr_path.name}")
    hdr = hdr_to_dict(hdr_path) if hdr_to_dict else {}
    arr = read_envi_cube(img_path, hdr)
    return arr, hdr


def _band_axis(arr: np.ndarray, hdr: Dict[str, Any]) -> int:
    if band_axis_from_header is None:
        nb = int(hdr.get("bands", 0) or 0)
        cands = [i for i, d in enumerate(arr.shape) if d == nb]
        return cands[0] if cands else (2 if arr.ndim == 3 else 0)
    return band_axis_from_header(arr, hdr)


def _pick_band_index_for_nm(hdr: Dict[str, Any], target_nm: float = 860.0) -> int:
    wl = hdr.get("wavelength")
    if not isinstance(wl, (list, tuple)) or len(wl) == 0:
        return max(0, int((int(hdr.get("bands", 1)) - 1) // 2))
    wl = np.array(wl, dtype=float)
    return int(np.nanargmin(np.abs(wl - target_nm)))


def _extract_band(arr: np.ndarray, hdr: Dict[str, Any], band_idx: int) -> np.ndarray:
    b_ax = _band_axis(arr, hdr)
    return np.take(arr, indices=band_idx, axis=b_ax)


def _safe_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr)


# -----------------------
# Dataclasses
# -----------------------
@dataclass
class MetricBand:
    band_index: int
    wavelength_nm: Optional[float]
    ratio_median: Optional[float]
    pct_lt0: Optional[float]
    pct_gt1: Optional[float]
    slope_before: Optional[float]
    slope_after: Optional[float]
    r2_before: Optional[float]
    r2_after: Optional[float]
    angular_cv_before: Optional[float]
    angular_cv_after: Optional[float]
    aspect_contrast_before: Optional[float]
    aspect_contrast_after: Optional[float]
    flags: List[str]


@dataclass
class MetricSummary:
    flight_stem: str
    created_from: str
    thresholds: Dict[str, Any]
    bands: List[MetricBand]
    nbar_rmse_before: Optional[float]
    nbar_rmse_after: Optional[float]
    nbar_improved: Optional[bool]
    global_flags: List[str]


# -----------------------
# Core metrics
# -----------------------
def ratio_sanity(corrected: np.ndarray, raw: np.ndarray, ref_hdr: Dict[str, Any]) -> Tuple[float, float, float]:
    bi = _pick_band_index_for_nm(ref_hdr, 860.0)
    corr_band = _extract_band(corrected, ref_hdr, bi)
    raw_band = _extract_band(raw, ref_hdr, bi)
    corr_u = to_unitless_reflectance(corr_band)
    raw_u = to_unitless_reflectance(raw_band)
    denom = np.maximum(raw_u, 1e-6)
    ratio = corr_u / denom
    m = _safe_mask(ratio)
    ratio_med = float(np.nanmedian(ratio[m])) if m.any() else np.nan
    corr_mask = _safe_mask(corr_u)
    if corr_mask.any():
        pct_lt0 = float(np.mean(corr_u[corr_mask] < 0.0))
        pct_gt1 = float(np.mean(corr_u[corr_mask] > 1.0))
    else:
        pct_lt0 = np.nan
        pct_gt1 = np.nan
    return ratio_med, pct_lt0, pct_gt1


def illumination_regression_band(
    refl: np.ndarray, hdr: Dict[str, Any], cosi: np.ndarray, band_idx: int
) -> Tuple[float, float]:
    band = _extract_band(refl, hdr, band_idx)
    b_ax = _band_axis(refl, hdr)
    spatial_axes = tuple(sorted({0, 1, 2} - {b_ax}))
    band2d = np.moveaxis(band, source=[spatial_axes[0], spatial_axes[1]], destination=[0, 1])
    y = band2d.reshape(-1)
    x = cosi.reshape(-1).astype(float)
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 100:
        return np.nan, np.nan
    x1 = np.vstack([np.ones(m.sum()), x[m]]).T
    beta, *_ = np.linalg.lstsq(x1, y[m], rcond=None)
    yhat = x1 @ beta
    ss_res = np.sum((y[m] - yhat) ** 2)
    ss_tot = np.sum((y[m] - np.nanmean(y[m])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    slope = float(beta[1])
    return slope, float(r2)


def aspect_contrast_band(
    refl: np.ndarray, hdr: Dict[str, Any], aspect_deg: np.ndarray, band_idx: int
) -> float:
    band = _extract_band(refl, hdr, band_idx)
    sun_mask = ((aspect_deg >= 135) & (aspect_deg <= 225))
    shade_mask = ((aspect_deg <= 45) | (aspect_deg >= 315))
    m_sun = np.isfinite(band) & np.isfinite(aspect_deg) & sun_mask
    m_sha = np.isfinite(band) & np.isfinite(aspect_deg) & shade_mask
    if m_sun.sum() < 100 or m_sha.sum() < 100:
        return np.nan
    band_u = to_unitless_reflectance(band)
    return float(np.nanmean(band_u[m_sun]) - np.nanmean(band_u[m_sha]))


def brdf_view_zenith_cv_band(
    refl: np.ndarray, hdr: Dict[str, Any], vza_deg: np.ndarray, band_idx: int
) -> float:
    band = _extract_band(refl, hdr, band_idx)
    band_u = to_unitless_reflectance(band)
    vals = []
    for lo, hi in zip(VIEW_ZENITH_BINS[:-1], VIEW_ZENITH_BINS[1:]):
        m = np.isfinite(band_u) & np.isfinite(vza_deg) & (vza_deg >= lo) & (vza_deg < hi)
        if m.sum() > 200:
            vals.append(float(np.nanmean(band_u[m])))
    if len(vals) < 2:
        return np.nan
    vals = np.array(vals, dtype=float)
    mu = np.nanmean(vals)
    sd = np.nanstd(vals)
    return float(sd / mu) if mu > 0 else np.nan


# -----------------------
# Main orchestrator
# -----------------------
def compute_metrics_for_flightline(
    base_folder: Path, flight_stem: str, thresholds: Dict[str, Any] = DEFAULT_THRESHOLDS
) -> MetricSummary:
    work = base_folder / flight_stem
    raw_img = work / f"{flight_stem}_envi.img"
    cor_img = work / f"{flight_stem}_brdfandtopo_corrected_envi.img"
    raw, raw_hdr = _load_envi(raw_img)
    cor, cor_hdr = _load_envi(cor_img)

    def _read_tif(grd: Optional[Path]) -> Optional[np.ndarray]:
        if grd is None or not grd.exists():
            return None
        try:
            import rasterio

            with rasterio.open(grd) as ds:
                return ds.read(1)
        except Exception as e:
            logger.warning("Failed to read %s: %s", grd.name, e)
            return None

    cosi_path = next((p for p in work.glob("*cosi*.tif")), None)
    aspect_path = next((p for p in work.glob("*aspect*.tif")), None)
    vza_path = next((p for p in work.glob("*view_zenith*.tif")), None)
    cosi = _read_tif(cosi_path)
    aspect = _read_tif(aspect_path)
    vza = _read_tif(vza_path)

    wl = cor_hdr.get("wavelength")
    n_bands = int(cor_hdr.get("bands", 0) or raw_hdr.get("bands", 0) or 0)
    wl_arr = (
        np.array(wl, dtype=float)
        if isinstance(wl, (list, tuple)) and len(wl) == n_bands
        else None
    )
    band_indices = (
        list(range(n_bands))
        if n_bands <= 60
        else list(np.unique(np.linspace(0, n_bands - 1, 24).astype(int)))
    )
    bands_out: List[MetricBand] = []

    for bi in band_indices:
        wl_nm = float(wl_arr[bi]) if wl_arr is not None else None
        raw_b = to_unitless_reflectance(_extract_band(raw, raw_hdr, bi))
        cor_b = to_unitless_reflectance(_extract_band(cor, cor_hdr, bi))
        denom = np.maximum(raw_b, 1e-6)
        ratio = cor_b / denom
        m = _safe_mask(ratio)
        ratio_median = float(np.nanmedian(ratio[m])) if m.any() else np.nan
        m2 = _safe_mask(cor_b)
        pct_lt0 = float(np.mean(cor_b[m2] < 0.0)) if m2.any() else np.nan
        pct_gt1 = float(np.mean(cor_b[m2] > 1.0)) if m2.any() else np.nan

        slope_bef = r2_bef = slope_aft = r2_aft = np.nan
        if cosi is not None:
            slope_bef, r2_bef = illumination_regression_band(raw, raw_hdr, cosi, bi)
            slope_aft, r2_aft = illumination_regression_band(cor, cor_hdr, cosi, bi)

        aspect_before = aspect_after = np.nan
        if aspect is not None:
            aspect_before = aspect_contrast_band(raw, raw_hdr, aspect, bi)
            aspect_after = aspect_contrast_band(cor, cor_hdr, aspect, bi)

        cv_bef = cv_aft = np.nan
        if vza is not None:
            cv_bef = brdf_view_zenith_cv_band(raw, raw_hdr, vza, bi)
            cv_aft = brdf_view_zenith_cv_band(cor, cor_hdr, vza, bi)

        flags: List[str] = []
        if np.isfinite(ratio_median):
            if (
                ratio_median < thresholds["ratio_median_min"]
                or ratio_median > thresholds["ratio_median_max"]
            ):
                flags.append("ratio_out_of_range")
        if np.isfinite(pct_lt0) and pct_lt0 > thresholds["pct_lt0_max"]:
            flags.append("reflectance_negatives_excess")
        if np.isfinite(pct_gt1) and pct_gt1 > thresholds["pct_gt1_max"]:
            flags.append("reflectance_over_one_excess")
        if np.isfinite(slope_bef) and np.isfinite(slope_aft) and abs(slope_bef) > 1e-9:
            red = (abs(slope_bef) - abs(slope_aft)) / abs(slope_bef)
            if red < thresholds["toc_min_slope_reduction"]:
                flags.append("toc_slope_reduction_insufficient")
        if np.isfinite(r2_bef) and np.isfinite(r2_aft) and r2_bef > 1e-9:
            r2_red = (r2_bef - r2_aft) / r2_bef
            if r2_red < thresholds["toc_min_r2_reduction"]:
                flags.append("toc_r2_reduction_insufficient")
        if (
            np.isfinite(aspect_before)
            and np.isfinite(aspect_after)
            and abs(aspect_before) > 1e-9
        ):
            a_red = (abs(aspect_before) - abs(aspect_after)) / abs(aspect_before)
            if a_red < thresholds["aspect_min_reduction"]:
                flags.append("aspect_contrast_reduction_insufficient")
        if np.isfinite(cv_bef) and np.isfinite(cv_aft) and cv_bef > 1e-9:
            cv_red = (cv_bef - cv_aft) / cv_bef
            if cv_red < thresholds["brdf_cv_min_reduction"]:
                flags.append("brdf_cv_reduction_insufficient")

        bands_out.append(
            MetricBand(
                band_index=bi,
                wavelength_nm=wl_nm,
                ratio_median=ratio_median,
                pct_lt0=pct_lt0,
                pct_gt1=pct_gt1,
                slope_before=slope_bef,
                slope_after=slope_aft,
                r2_before=r2_bef,
                r2_after=r2_aft,
                angular_cv_before=cv_bef,
                angular_cv_after=cv_aft,
                aspect_contrast_before=aspect_before,
                aspect_contrast_after=aspect_after,
                flags=flags,
            )
        )

    nbar_rmse_before = nbar_rmse_after = nbar_improved = None
    global_flags: List[str] = []
    try:
        r_med, p_lt0, p_gt1 = ratio_sanity(cor, raw, cor_hdr)
        if np.isfinite(r_med):
            if (
                r_med < thresholds["ratio_median_min"]
                or r_med > thresholds["ratio_median_max"]
            ):
                global_flags.append("scene_ratio_out_of_range")
        if np.isfinite(p_lt0) and p_lt0 > thresholds["pct_lt0_max"]:
            global_flags.append("scene_negatives_excess")
        if np.isfinite(p_gt1) and p_gt1 > thresholds["pct_gt1_max"]:
            global_flags.append("scene_over_one_excess")
    except Exception as e:
        logger.warning("Scene ratio sanity failed: %s", e)

    return MetricSummary(
        flight_stem=flight_stem,
        created_from=str(work),
        thresholds=thresholds,
        bands=bands_out,
        nbar_rmse_before=nbar_rmse_before,
        nbar_rmse_after=nbar_rmse_after,
        nbar_improved=nbar_improved,
        global_flags=global_flags,
    )


# -----------------------
# Write outputs (JSON+Parquet)
# -----------------------
def write_metrics(
    base_folder: Path,
    flight_stem: str,
    out_json: Optional[Path] = None,
    out_parquet: Optional[Path] = None,
    thresholds: Dict[str, Any] = DEFAULT_THRESHOLDS,
) -> Tuple[Path, Path]:
    ms = compute_metrics_for_flightline(base_folder, flight_stem, thresholds=thresholds)
    work = base_folder / flight_stem
    out_json = out_json or (work / f"{flight_stem}_qa_metrics.json")
    out_parquet = out_parquet or (work / f"{flight_stem}_qa_metrics.parquet")
    with open(out_json, "w") as f:
        json.dump(asdict(ms), f, indent=2)
    rows = []
    for b in ms.bands:
        d = asdict(b)
        d["flight_stem"] = ms.flight_stem
        rows.append(d)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    logger.info(
        "📑 Wrote QA metrics → %s / %s", out_json.name, out_parquet.name
    )
    bad_bands = sum(1 for b in ms.bands if b.flags)
    if ms.global_flags or bad_bands:
        logger.warning(
            "⚠️  Flags present: global=%s; bands_with_flags=%d",
            ms.global_flags,
            bad_bands,
        )
    return out_json, out_parquet


def read_metrics(base_folder: Path, flight_stem: str) -> Optional[pd.DataFrame]:
    pq_path = base_folder / flight_stem / f"{flight_stem}_qa_metrics.parquet"
    if not pq_path.exists():
        logger.warning("No QA metrics parquet found for %s", flight_stem)
        return None
    return pd.read_parquet(pq_path)


def main():
    ap = argparse.ArgumentParser(
        description="Compute BRDF+TOC QA metrics and flags for a flightline (outputs JSON + Parquet)."
    )
    ap.add_argument("--base-folder", type=Path, required=True)
    ap.add_argument("--flight-stem", type=str, required=True)
    args = ap.parse_args()
    write_metrics(args.base_folder, args.flight_stem)


if __name__ == "__main__":
    main()
