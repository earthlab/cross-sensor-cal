"""Utilities for rendering QA panels with paired metrics JSON."""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import math
import subprocess
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .brightness_config import load_brightness_coefficients
from .envi import hdr_to_dict, read_envi_cube
from .header_utils import wavelengths_from_hdr
from .qa_metrics import (
    ConvolutionReport,
    CorrectionReport,
    HeaderReport,
    MaskReport,
    Provenance,
    QAMetrics,
    write_json,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DEFAULT_RGB_TARGETS = (660.0, 560.0, 490.0)
_EXPECTED_HEADER_KEYS = ["wavelength", "fwhm", "band names"]
_NEGATIVE_WARN_THRESHOLD = 1.0
_OVERBRIGHT_WARN_THRESHOLD = 1.0
_DELTA_WARN_THRESHOLD = 0.05


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_primary_cube(flightline_dir: Path) -> tuple[Path, Path]:
    """
    Discover the raw and corrected ENVI cubes for a flightline.
    
    Handles naming mismatches between raw files (which may use legacy T-separator format)
    and corrected files (which use canonical flight_id format).
    """
    raw_candidates = sorted(
        p
        for p in flightline_dir.glob("*_envi.img")
        if "corrected" not in p.stem and "qa" not in p.stem
    )
    if not raw_candidates:
        raise FileNotFoundError(f"No ENVI cube found in {flightline_dir}")
    raw_path = raw_candidates[0]
    prefix = raw_path.stem.rsplit("_envi", 1)[0]
    
    # Strategy 1: Try exact prefix match first (works if naming is consistent)
    corrected = flightline_dir / f"{prefix}_brdfandtopo_corrected_envi.img"
    if corrected.exists():
        return raw_path, corrected
    
    # Strategy 2: Try to construct canonical flight_id from parsed raw file
    # Raw file might have legacy format: NEON_D13_NIWO_DP1_20200720T163210_20200720_163210_reflectance_envi.img
    # Corrected file uses canonical format: NEON_D13_NIWO_DP1_20200720_163210_reflectance_brdfandtopo_corrected_envi.img
    try:
        from .file_types import NEONReflectanceENVIFile
        
        # Try parsing the raw ENVI filename
        envi_file = NEONReflectanceENVIFile.from_filename(raw_path)
        
        # Extract components for canonical flight_id format
        # Format: NEON_{domain}_{site}_DP1_{date}_{time}_reflectance
        domain = getattr(envi_file, "domain", None)
        site = getattr(envi_file, "site", None)
        date = getattr(envi_file, "date", None)
        time = getattr(envi_file, "time", None)
        
        if domain and site and date and time:
            # Construct canonical flight_id (remove T-separator and duplicate date/time)
            canonical_flight_id = f"NEON_{domain}_{site}_DP1_{date}_{time}_reflectance"
            canonical_corrected = flightline_dir / f"{canonical_flight_id}_brdfandtopo_corrected_envi.img"
            if canonical_corrected.exists():
                return raw_path, canonical_corrected
    except (ValueError, AttributeError):
        # If parsing fails, fall through to Strategy 3
        pass
    except Exception:
        # Catch any other unexpected errors and log them, then fall through to Strategy 3
        pass
    
    # Strategy 3: Fallback - glob search for any corrected file
    corrected_candidates = sorted(
        p
        for p in flightline_dir.glob("*_brdfandtopo_corrected_envi.img")
        if p.is_file()
    )
    if corrected_candidates:
        return raw_path, corrected_candidates[0]
    
    # If all strategies fail, raise error with helpful message
    raise FileNotFoundError(
        f"Missing corrected cube: Could not find corrected ENVI file for raw file {raw_path.name}.\n"
        f"Tried:\n"
        f"  1. {corrected.name} (Strategy 1: exact prefix match)\n"
        f"  2. Canonical format based on flight_id (Strategy 2: parsed from raw file)\n"
        f"  3. Any *_brdfandtopo_corrected_envi.img in {flightline_dir} (Strategy 3: glob search)\n"
        f"Raw file may use legacy format (T-separator) while corrected uses canonical format.\n"
        f"Found {len(corrected_candidates)} corrected file(s) but none matched."
    )


def _list_envi_products(flightline_dir: Path, prefix: str) -> list[Path]:
    """Return all ENVI .img products for this flightline (for page 1 overview).

    Includes the raw ENVI, corrected ENVI, and any convolved ENVI products,
    but excludes QA images and non-ENVI artifacts.
    """

    products: list[Path] = []
    for img_path in sorted(flightline_dir.glob(f"{prefix}*_envi.img")):
        if "qa" in img_path.stem:
            continue
        products.append(img_path)
    if not products:
        for img_path in sorted(flightline_dir.glob("*_envi.img")):
            if "qa" in img_path.stem:
                continue
            products.append(img_path)
    return products


def _flightline_prefix(raw_path: Path) -> str:
    stem = raw_path.stem
    return stem.rsplit("_envi", 1)[0]


def _rgb_targets_from_arg(rgb_bands: str | None) -> tuple[float, float, float]:
    if not rgb_bands:
        return _DEFAULT_RGB_TARGETS
    tokens = [token.strip() for token in rgb_bands.split(",") if token.strip()]
    if len(tokens) != 3:
        return _DEFAULT_RGB_TARGETS
    mapping = {"R": 660.0, "G": 560.0, "B": 490.0}
    resolved: list[float] = []
    for token in tokens:
        upper = token.upper()
        if upper in mapping:
            resolved.append(mapping[upper])
            continue
        try:
            resolved.append(float(token))
        except ValueError:
            return _DEFAULT_RGB_TARGETS
    return tuple(resolved)  # type: ignore[return-value]


def _deterministic_sample(
    cube: np.ndarray,
    mask: np.ndarray,
    n_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    bands, rows, cols = cube.shape
    total_pixels = rows * cols
    if total_pixels <= n_sample:
        return cube.reshape(bands, -1), mask.reshape(bands, -1)
    step = max(1, int(math.sqrt(total_pixels / n_sample)))
    ys = np.arange(0, rows, step)
    xs = np.arange(0, cols, step)
    sampled = cube[:, ys][:, :, xs]
    sampled_mask = mask[:, ys][:, :, xs]
    flat = sampled.reshape(bands, -1)
    flat_mask = sampled_mask.reshape(bands, -1)
    if flat.shape[1] > n_sample:
        idx = np.linspace(0, flat.shape[1] - 1, num=n_sample, dtype=int)
        flat = flat[:, idx]
        flat_mask = flat_mask[:, idx]
    return flat, flat_mask


def _percentile_stretch(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    lo_val = np.nanpercentile(arr, lo)
    hi_val = np.nanpercentile(arr, hi)
    if not np.isfinite(lo_val) or not np.isfinite(hi_val) or hi_val == lo_val:
        return np.zeros_like(arr)
    scaled = np.clip((arr - lo_val) / (hi_val - lo_val), 0.0, 1.0)
    return scaled


def _rgb_preview(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    rgb_targets: Sequence[float],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    bands, rows, cols = cube.shape
    if wavelengths.size:
        # Filter out NaN values before using nanargmin
        finite_mask = np.isfinite(wavelengths)
        if not np.any(finite_mask):
            # All wavelengths are NaN - fall back to first few bands
            indices = [0, min(1, bands - 1), min(2, bands - 1)]
        else:
            # Use only finite wavelengths for band selection
            finite_wavelengths = wavelengths[finite_mask]
            finite_indices = np.where(finite_mask)[0]
            indices = []
            for target in rgb_targets:
                try:
                    idx = int(np.nanargmin(np.abs(finite_wavelengths - target)))
                    indices.append(finite_indices[idx])
                except ValueError:
                    # Fallback if nanargmin fails
                    indices.append(0)
    else:
        indices = [0, min(1, bands - 1), min(2, bands - 1)]
    rgb = np.stack([_percentile_stretch(cube[idx]) for idx in indices], axis=-1)
    return rgb, (indices[0], indices[1], indices[2])


def _header_report(header: dict, wavelengths: np.ndarray, source: str) -> HeaderReport:
    keys_present = [key for key in _EXPECTED_HEADER_KEYS if key in header]
    keys_missing = [key for key in _EXPECTED_HEADER_KEYS if key not in keys_present]
    unit = header.get("wavelength units") or header.get("wavelength_units")
    unit = unit if isinstance(unit, str) else None
    finite = wavelengths[np.isfinite(wavelengths)]
    first_nm = float(finite[0]) if finite.size else None
    last_nm = float(finite[-1]) if finite.size else None
    monotonic = bool(np.all(np.diff(finite) > 0)) if finite.size > 1 else None
    n_bands = int(header.get("bands") or wavelengths.size or 0)
    return HeaderReport(
        keys_present=keys_present,
        keys_missing=keys_missing,
        wavelength_unit=unit,
        n_bands=n_bands,
        n_wavelengths_finite=int(finite.size),
        first_nm=first_nm,
        last_nm=last_nm,
        wavelengths_monotonic=monotonic,
        wavelength_source=source,
    )


def _correction_report(
    raw_sample: np.ndarray,
    corr_sample: np.ndarray,
    sample_mask: np.ndarray,
) -> CorrectionReport:
    # Exclude -9999 (no-data) values from delta calculation
    # -9999 values can cause spurious large deltas (e.g., -9999 - 0.5 = -9999.5, or 0.5 - (-9999) = 9999.5)
    # Use a more lenient threshold to catch values close to -9999
    no_data_threshold = -9990.0  # Anything <= -9990 is considered no-data
    
    # Mask out pixels where either raw or corrected is no-data
    raw_is_nodata = raw_sample <= no_data_threshold
    corr_is_nodata = corr_sample <= no_data_threshold
    either_is_nodata = raw_is_nodata | corr_is_nodata
    
    # Combined mask: valid pixels AND not no-data
    combined_mask = sample_mask & ~either_is_nodata
    
    # Calculate difference only for valid, non-no-data pixels
    diff = np.where(combined_mask, corr_sample - raw_sample, np.nan)
    
    # CRITICAL: Exclude deltas that are suspiciously large (likely -9999 contamination)
    # If absolute delta is > 1000, it's almost certainly from a -9999 value
    # This catches cases like: -9999 - 76 = -10075 or 76 - (-9999) = 10075
    # Use a threshold of 1000 to be safe (reflectance deltas should be < 1 typically)
    suspicious_deltas = np.abs(diff) > 1000.0
    n_suspicious = np.sum(suspicious_deltas)
    if n_suspicious > 0:
        print(f"[QA] ⚠️  Excluding {n_suspicious:,} suspicious deltas (|delta| > 1000) likely from -9999 contamination")
    diff = np.where(suspicious_deltas, np.nan, diff)
    
    delta_median = np.nanmedian(diff, axis=1)
    q75 = np.nanpercentile(diff, 75, axis=1)
    q25 = np.nanpercentile(diff, 25, axis=1)
    delta_iqr = q75 - q25
    
    # Count how many bands had -9999 contamination
    nodata_bands = np.sum(either_is_nodata, axis=1)
    if np.any(nodata_bands > 0):
        max_contaminated = np.max(nodata_bands)
        print(f"[QA] ⚠️  Excluded -9999 values from delta calculation (max {max_contaminated} pixels per band)")
    
    order = np.argsort(np.abs(delta_median))[::-1]
    top = order[:3].tolist()
    return CorrectionReport(
        delta_median=delta_median.astype(float).tolist(),
        delta_iqr=delta_iqr.astype(float).tolist(),
        largest_delta_indices=[int(idx) for idx in top],
    )


def _spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.nansum(a * b, axis=0)
    norm_a = np.sqrt(np.nansum(a * a, axis=0))
    norm_b = np.sqrt(np.nansum(b * b, axis=0))
    denom = np.maximum(norm_a * norm_b, 1e-6)
    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    return float(np.nanmean(angles))


def _load_filtered_parquet_stats(flightline_dir: Path, prefix: str) -> dict | None:
    """Load statistics from filtered merged parquet file if available."""
    if pd is None:
        return None
    
    merged_parquet = flightline_dir / f"{prefix}_merged_pixel_extraction.parquet"
    if not merged_parquet.exists():
        return None
    
    try:
        # Read a sample to get statistics
        df = pd.read_parquet(merged_parquet)
        
        # Identify spectral columns (excluding raw_* and metadata)
        import re
        spectral_cols = [col for col in df.columns 
                        if re.search(r'_wl\d+nm|^wl\d+nm', col, re.IGNORECASE) 
                        and not col.lower().startswith('raw_')
                        and col.lower() not in ['pixel_id', 'row', 'col', 'x', 'y', 'lon', 'lat']]
        
        if not spectral_cols:
            return None
        
        # Convert to numeric
        spectral_df = df[spectral_cols].apply(pd.to_numeric, errors='coerce')
        
        # Calculate statistics
        negative_mask = (spectral_df < 0) & spectral_df.notna()
        overbright_mask = (spectral_df > 1.2) & spectral_df.notna()
        no_data_mask = (np.abs(spectral_df - (-9999.0)) < 0.01) & spectral_df.notna()
        
        total_cells = spectral_df.notna().sum().sum()
        negative_cells = negative_mask.sum().sum()
        overbright_cells = overbright_mask.sum().sum()
        no_data_cells = no_data_mask.sum().sum()
        valid_cells = total_cells - no_data_cells
        
        return {
            'n_rows': len(df),
            'n_spectral_cols': len(spectral_cols),
            'negatives_pct': (negative_cells / valid_cells * 100) if valid_cells > 0 else 0.0,
            'overbright_pct': (overbright_cells / valid_cells * 100) if valid_cells > 0 else 0.0,
            'no_data_pct': (no_data_cells / total_cells * 100) if total_cells > 0 else 0.0,
            'valid_pct': (valid_cells / total_cells * 100) if total_cells > 0 else 0.0,
        }
    except Exception as e:
        logger.warning(f"Failed to load filtered parquet stats: {e}")
        return None


def _safe_extract_geometry_value(value) -> float | None:
    """Safely extract geometry value, handling dict, tuple, or float."""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, dict):
        return value.get("mean") if "mean" in value else None
    elif isinstance(value, (list, tuple)):
        # If it's a tuple/list, try to get mean or first element
        if len(value) > 0:
            try:
                return float(value[0])
            except (ValueError, TypeError):
                return None
    return None


def _build_brightness_summary_table(
    brightness_map: dict[str, dict[int, float]]
) -> dict[str, list[dict[str, float | str]]]:
    summary: dict[str, list[dict[str, float | str]]] = {}

    if not brightness_map:
        default_coeffs = load_brightness_coefficients()
        summary["landsat_to_micasense"] = [
            {
                "band": band_idx,
                "config_coeff_pct": coeff,
                "applied_coeff_pct": coeff,
                "brightness_coeff_pct": coeff,
                "brightness_coeff_label": f"{coeff:+.3f}%",
            }
            for band_idx, coeff in sorted(default_coeffs.items())
        ]
        return summary

    for system_pair, applied in brightness_map.items():
        config_coeffs = load_brightness_coefficients(system_pair)
        rows: list[dict[str, float | str]] = []
        for band_idx, config_value in sorted(config_coeffs.items()):
            applied_value = applied.get(band_idx, config_value)
            rows.append(
                {
                    "band": band_idx,
                    "config_coeff_pct": config_value,
                    "applied_coeff_pct": applied_value,
                    "brightness_coeff_pct": applied_value,
                    "brightness_coeff_label": f"{applied_value:+.3f}%",
                }
            )
        summary[system_pair] = rows

    return summary


def _convolution_reports(
    base_dir: Path,
    prefix: str,
    corr_cube: np.ndarray,
    sample_mask: np.ndarray,
) -> tuple[
    list[ConvolutionReport],
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, dict[int, float]],
]:
    reports: list[ConvolutionReport] = []
    scatter_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    brightness_map: dict[str, dict[int, float]] = {}
    patterns = [
        f"{prefix}_*_convolved_envi.img",
        f"{prefix}_resampled_*_envi.img",
        f"{prefix}_resampled_*_envi_masked.img",
        f"{prefix}_*landsat*_envi.img",  # Add Landsat patterns
        f"{prefix}_*micasense*_envi.img",  # Add MicaSense patterns
        f"{prefix}_*etm*_envi.img",  # Add ETM+ patterns
        f"{prefix}_*oli*_envi.img",  # Add OLI patterns
        f"{prefix}_*tm*_envi.img",  # Add TM patterns
    ]
    candidates: set[Path] = set()
    for pattern in patterns:
        for match in base_dir.rglob(pattern):
            # Exclude raw and corrected ENVI files, but include convolved/resampled
            stem = match.stem.lower()
            if ("corrected" not in stem and 
                "brdfandtopo" not in stem and
                stem != f"{prefix.lower()}_envi" and
                ("convolved" in stem or "resampled" in stem or 
                 "landsat" in stem or "micasense" in stem or 
                 "etm" in stem or "oli" in stem or "tm" in stem)):
                candidates.add(match)

    for img_path in sorted(candidates):
        sensor = img_path.stem.replace(f"{prefix}_", "").replace("_convolved_envi", "")
        hdr = hdr_to_dict(img_path.with_suffix(".hdr"))
        cube = read_envi_cube(img_path, hdr)
        if cube.shape != corr_cube.shape:
            continue
        brightness_entry = hdr.get("brightness_coefficients")
        if isinstance(brightness_entry, str):
            try:
                parsed = json.loads(brightness_entry)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                for system_pair, coeffs in parsed.items():
                    if not isinstance(coeffs, dict):
                        continue
                    dest = brightness_map.setdefault(system_pair, {})
                    for key, value in coeffs.items():
                        try:
                            dest[int(key)] = float(value)
                        except (TypeError, ValueError):
                            continue
        flat_cube = cube.reshape(cube.shape[0], -1)
        flat_corr = corr_cube.reshape(corr_cube.shape[0], -1)
        rmse = np.sqrt(np.nanmean((flat_corr - flat_cube) ** 2, axis=1))
        sam = _spectral_angle(flat_corr, flat_cube)
        reports.append(
            ConvolutionReport(sensor=sensor, rmse=rmse.astype(float).tolist(), sam=sam)
        )
        valid = sample_mask.any(axis=0)
        if not np.any(valid):
            continue
        corrected_vals = flat_corr[:, valid].flatten()
        convolved_vals = flat_cube[:, valid].flatten()
        scatter_data[sensor] = (corrected_vals, convolved_vals)
    return reports, scatter_data, brightness_map


def _render_page1_envi_overview(
    pdf: PdfPages,
    flightline_dir: Path,
    prefix: str,
    rgb_targets: Sequence[float],
) -> None:
    """Page 1: one row with one panel per ENVI product, to show they exist & render."""

    envi_paths = _list_envi_products(flightline_dir, prefix)
    if not envi_paths:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No ENVI products found", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(f"ENVI overview – {prefix}")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    n = len(envi_paths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    fig.suptitle(f"ENVI product overview – {prefix}")
    axes_row = axes[0]

    for ax, img_path in zip(axes_row, envi_paths):
        hdr = hdr_to_dict(img_path.with_suffix(".hdr"))
        cube = read_envi_cube(img_path, hdr)
        if cube.ndim != 3:
            ax.text(0.5, 0.5, "Non-3D ENVI cube", ha="center", va="center")
            ax.axis("off")
            continue
        wavelengths, _ = wavelengths_from_hdr(hdr)
        rgb_image, _ = _rgb_preview(cube, wavelengths, rgb_targets)
        ax.imshow(np.clip(rgb_image, 0, 1))
        
        # Shorten product name for display - extract key identifier
        stem = img_path.stem
        # Remove long prefix if present
        if stem.startswith(prefix):
            short_name = stem[len(prefix):].lstrip('_')
        else:
            short_name = stem
        
        # Further shorten common suffixes
        short_name = short_name.replace('_brdfandtopo_corrected_envi', '_corrected')
        short_name = short_name.replace('_envi', '')
        short_name = short_name.replace('_convolved', '_conv')
        short_name = short_name.replace('_resampled', '_resamp')
        
        # Wrap long names
        if len(short_name) > 30:
            # Try to split at underscores
            parts = short_name.split('_')
            if len(parts) > 2:
                # Take first and last parts
                short_name = f"{parts[0]}_{parts[-1]}"
            else:
                short_name = short_name[:27] + "..."
        
        ax.set_title(short_name, fontsize=8, wrap=True)
        ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _load_correction_geometry_json(
    flightline_dir: Path,
    corr_path: Path,
) -> dict | None:
    """Try to load the BRDF/topo correction JSON for extra diagnostics (optional)."""

    json_path = corr_path.with_suffix(".json")
    if not json_path.exists():
        candidates = sorted(flightline_dir.glob("*brdfandtopo_corrected_envi.json"))
        json_path = candidates[0] if candidates else None
    if not json_path or not json_path.exists():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _render_page2_topo_brdf(
    pdf: PdfPages,
    prefix: str,
    wavelengths: np.ndarray,
    correction_report: CorrectionReport,
    raw_sample: np.ndarray,
    corr_sample: np.ndarray,
    sample_mask: np.ndarray,
    corr_path: Path,
    filtered_stats: dict | None = None,
) -> None:
    """Page 2: diagnostics specific to topo (row 1) and BRDF (row 2) corrections."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Topographic + BRDF diagnostics – {prefix}")

    ax_topo_hist = axes[0, 0]
    _render_hist(ax_topo_hist, raw_sample, corr_sample, sample_mask)
    ax_topo_hist.set_title("Topographic + BRDF: pre vs post histograms")

    ax_topo_delta = axes[0, 1]
    _render_delta(ax_topo_delta, wavelengths, correction_report)
    title = "Topographic + BRDF: Δ median vs λ (before filtering)"
    if filtered_stats:
        title += "\n(Note: -9999 values excluded from delta calculation)"
    ax_topo_delta.set_title(title, fontsize=9)

    params = _load_correction_geometry_json(corr_path.parent, corr_path) or {}
    geom = params.get("geometry", {}) if isinstance(params, dict) else {}
    topo_keys = ["slope", "aspect"]
    brdf_keys = ["solar_zn", "solar_az", "sensor_zn", "sensor_az"]

    ax_brdf_geom = axes[1, 0]
    if geom:
        topo_present = [k for k in topo_keys if k in geom]
        if topo_present:
            means = []
            valid_keys = []
            for k in topo_present:
                value = _safe_extract_geometry_value(geom[k])
                if value is not None:
                    means.append(value)
                    valid_keys.append(k)
            if valid_keys and means:
                ax_brdf_geom.bar(valid_keys, means)
                ax_brdf_geom.set_title("Topographic geometry (mean)")
                ax_brdf_geom.set_ylabel("Value (radians)")
                # Set reasonable y-axis limits
                y_range = max(means) - min(means) if len(means) > 1 else abs(means[0]) if means else 1
                if y_range > 0:
                    ax_brdf_geom.set_ylim(min(means) - 0.1 * y_range, max(means) + 0.1 * y_range)
            else:
                ax_brdf_geom.text(0.5, 0.5, "No valid topo geometry values", ha="center", va="center")
        else:
            ax_brdf_geom.text(
                0.5,
                0.5,
                "No topo geometry stats in JSON",
                ha="center",
                va="center",
            )
        ax_brdf_geom.grid(True, alpha=0.2)
    else:
        ax_brdf_geom.text(
            0.5, 0.5, "No BRDF/topo JSON geometry available", ha="center", va="center"
        )
        ax_brdf_geom.grid(False)

    ax_brdf_geom2 = axes[1, 1]
    if geom:
        brdf_present = [k for k in brdf_keys if k in geom]
        if brdf_present:
            means = []
            valid_keys = []
            for k in brdf_present:
                value = _safe_extract_geometry_value(geom[k])
                if value is not None:
                    means.append(value)
                    valid_keys.append(k)
            if valid_keys and means:
                ax_brdf_geom2.bar(valid_keys, means)
                ax_brdf_geom2.set_title("BRDF geometry (mean)")
                ax_brdf_geom2.set_ylabel("Value (radians)")
                # Set reasonable y-axis limits
                y_range = max(means) - min(means) if len(means) > 1 else abs(means[0]) if means else 1
                if y_range > 0:
                    ax_brdf_geom2.set_ylim(min(means) - 0.1 * y_range, max(means) + 0.1 * y_range)
            else:
                ax_brdf_geom2.text(0.5, 0.5, "No valid BRDF geometry values", ha="center", va="center")
        else:
            ax_brdf_geom2.text(
                0.5, 0.5, "No BRDF geometry stats in JSON", ha="center", va="center"
            )
    else:
        ax_brdf_geom2.text(
            0.5, 0.5, "No BRDF/topo JSON available", ha="center", va="center"
        )
    ax_brdf_geom2.grid(True, alpha=0.2)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_page3_remaining(
    pdf: PdfPages,
    prefix: str,
    metrics: QAMetrics,
    scatter_data: dict[str, tuple[np.ndarray, np.ndarray]],
    filtered_stats: dict | None = None,
) -> None:
    """Page 3: remaining QA diagnostics (convolution + header/mask/issue summary)."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Additional QA diagnostics – {prefix}")

    ax_scatter = axes[0, 0]
    _render_scatter(ax_scatter, scatter_data)
    title = "Convolved vs corrected (all sensors)"
    if not scatter_data:
        title += "\n⚠ No convolved sensors found - check ENVI files"
    ax_scatter.set_title(title, fontsize=9)

    ax_header = axes[0, 1]
    header = metrics.header
    lines = [
        f"Header keys present: {', '.join(header.keys_present) or 'none'}",
        f"Header keys missing: {', '.join(header.keys_missing) or 'none'}",
        f"Wavelength unit: {header.wavelength_unit or 'unknown'}",
        f"n_bands: {header.n_bands}",
        f"finite wavelengths: {header.n_wavelengths_finite}",
        f"first_nm: {header.first_nm}",
        f"last_nm: {header.last_nm}",
        f"monotonic: {header.wavelengths_monotonic}",
        f"source: {header.wavelength_source}",
    ]
    ax_header.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9)
    ax_header.axis("off")
    ax_header.set_title("Header / wavelength integrity")

    ax_mask = axes[1, 0]
    mask = metrics.mask
    negatives_pct = metrics.negatives_pct
    overbright_pct = metrics.overbright_pct
    lines = [
        "=== ENVI Cube (Before Filtering) ===",
        f"Total pixels: {mask.n_total:,}",
        f"Valid pixels: {mask.n_valid:,}",
        f"Valid %: {mask.valid_pct:.2f}%",
        f"Negatives %: {negatives_pct:.2f}%",
        f">1.2 reflectance %: {overbright_pct:.2f}%",
    ]
    if filtered_stats:
        lines.extend([
            "",
            "=== Filtered Parquet (After Filtering) ===",
            f"Rows: {filtered_stats['n_rows']:,}",
            f"Spectral columns: {filtered_stats['n_spectral_cols']}",
            f"Valid %: {filtered_stats['valid_pct']:.2f}%",
            f"Negatives %: {filtered_stats['negatives_pct']:.2f}%",
            f">1.2 reflectance %: {filtered_stats['overbright_pct']:.2f}%",
            f"No-data %: {filtered_stats['no_data_pct']:.2f}%",
        ])
    else:
        lines.append("\n(Filtered parquet stats not available)")
    ax_mask.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=8, family='monospace')
    ax_mask.axis("off")
    ax_mask.set_title("Mask coverage & negatives")

    ax_issues = axes[1, 1]
    lines = []
    if metrics.issues:
        lines.extend(f"⚠ {issue}" for issue in metrics.issues)
    else:
        lines.append("No general QA issues flagged.")

    if metrics.brightness_coefficients:
        lines.append("")
        lines.append("Brightness coefficients (percent):")
        for system_pair, coeffs in metrics.brightness_coefficients.items():
            lines.append(f"{system_pair}:")
            for band_idx in sorted(coeffs):
                lines.append(f"  Band {band_idx}: {coeffs[band_idx]:.3f}%")
    else:
        lines.append("")
        lines.append("No brightness adjustment applied.")

    ax_issues.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9)
    ax_issues.axis("off")
    ax_issues.set_title("Issues & brightness adjustments")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_page4_parquet_merge_quality(
    pdf: PdfPages,
    flightline_dir: Path,
    prefix: str,
) -> None:
    """Page 4: Parquet and merge quality analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Parquet & Merge Quality Analysis – {prefix}")
    
    # Top-left: Parquet file inventory
    ax_parquet = axes[0, 0]
    parquet_files = sorted(flightline_dir.glob("*.parquet"))
    parquet_info = []
    total_size = 0
    
    for pq in parquet_files:
        size_mb = pq.stat().st_size / (1024 * 1024)
        total_size += size_mb
        parquet_info.append(f"{pq.name[:50]}: {size_mb:.1f} MB")
    
    if parquet_info:
        text = "=== Parquet Files ===\n" + "\n".join(parquet_info[:15])  # Limit to 15 files
        if len(parquet_info) > 15:
            text += f"\n... and {len(parquet_info) - 15} more files"
        text += f"\n\nTotal: {len(parquet_files)} files, {total_size:.1f} MB"
    else:
        text = "No parquet files found"
    
    ax_parquet.text(0.01, 0.99, text, va="top", ha="left", fontsize=8, family='monospace')
    ax_parquet.axis("off")
    ax_parquet.set_title("Parquet File Inventory")
    
    # Top-right: Merge status
    ax_merge = axes[0, 1]
    merged_parquet = flightline_dir / f"{prefix}_merged_pixel_extraction.parquet"
    
    if merged_parquet.exists():
        try:
            if pd is not None:
                df = pd.read_parquet(merged_parquet)
                n_rows = len(df)
                n_cols = len(df.columns)
                
                # Count expected parquet types
                original_parquets = list(flightline_dir.glob("*_envi.parquet"))
                corrected_parquets = list(flightline_dir.glob("*_corrected*.parquet"))
                resampled_parquets = [p for p in flightline_dir.glob("*.parquet") 
                                    if "merged" not in p.name and "corrected" not in p.name 
                                    and "_envi.parquet" not in p.name]
                
                lines = [
                    "=== Merge Status ===",
                    "✅ Merged file exists",
                    f"Rows: {n_rows:,}",
                    f"Columns: {n_cols}",
                    "",
                    "=== Input Files ===",
                    f"Original: {len(original_parquets)}",
                    f"Corrected: {len(corrected_parquets)}",
                    f"Resampled: {len(resampled_parquets)}",
                    f"Total inputs: {len(original_parquets) + len(corrected_parquets) + len(resampled_parquets)}",
                ]
            else:
                lines = [
                    "=== Merge Status ===",
                    "✅ Merged file exists",
                    f"Size: {merged_parquet.stat().st_size / (1024*1024):.1f} MB",
                    "(pandas not available for detailed stats)",
                ]
        except Exception as e:
            lines = [
                "=== Merge Status ===",
                f"⚠️ Error reading merged file: {e}",
            ]
    else:
        lines = [
            "=== Merge Status ===",
            "❌ Merged file not found",
            f"Expected: {merged_parquet.name}",
        ]
    
    ax_merge.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9, family='monospace')
    ax_merge.axis("off")
    ax_merge.set_title("Merge Status")
    
    # Bottom-left: Column summary
    ax_cols = axes[1, 0]
    if merged_parquet.exists() and pd is not None:
        try:
            df = pd.read_parquet(merged_parquet)
            import re
            
            # Categorize columns
            spectral_cols = [c for c in df.columns if re.search(r'_wl\d+nm|^wl\d+nm', c, re.IGNORECASE)]
            raw_cols = [c for c in spectral_cols if c.lower().startswith('raw_')]
            corr_cols = [c for c in spectral_cols if 'corr' in c.lower() and not c.lower().startswith('raw_')]
            conv_cols = [c for c in spectral_cols if any(x in c.lower() for x in ['etm', 'oli', 'tm', 'micasense']) and not c.lower().startswith('raw_')]
            meta_cols = [c for c in df.columns if c not in spectral_cols]
            
            lines = [
                "=== Column Summary ===",
                f"Total columns: {len(df.columns)}",
                "",
                "Spectral columns:",
                f"  Total: {len(spectral_cols)}",
                f"  Raw: {len(raw_cols)}",
                f"  Corrected: {len(corr_cols)}",
                f"  Convolved: {len(conv_cols)}",
                "",
                f"Metadata columns: {len(meta_cols)}",
            ]
        except Exception as e:
            lines = [f"Error analyzing columns: {e}"]
    else:
        lines = ["Column analysis requires merged parquet file"]
    
    ax_cols.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9, family='monospace')
    ax_cols.axis("off")
    ax_cols.set_title("Column Summary")
    
    # Bottom-right: Quality checks
    ax_quality = axes[1, 1]
    quality_checks = []
    
    # Check 1: Merged file exists
    if merged_parquet.exists():
        quality_checks.append("✅ Merged parquet exists")
    else:
        quality_checks.append("❌ Merged parquet missing")
    
    # Check 2: Expected parquet files
    expected_types = {
        "Original ENVI": list(flightline_dir.glob("*_envi.parquet")),
        "Corrected": list(flightline_dir.glob("*_corrected*.parquet")),
    }
    for name, files in expected_types.items():
        if files:
            quality_checks.append(f"✅ {name}: {len(files)} file(s)")
        else:
            quality_checks.append(f"⚠️ {name}: 0 files")
    
    # Check 3: Resampled products
    resampled = [p for p in flightline_dir.glob("*.parquet") 
                if "merged" not in p.name and "corrected" not in p.name 
                and "_envi.parquet" not in p.name and "polygon" not in p.name]
    if resampled:
        quality_checks.append(f"✅ Resampled products: {len(resampled)} file(s)")
    else:
        quality_checks.append("⚠️ No resampled products found")
    
    ax_quality.text(0.01, 0.99, "\n".join(quality_checks), va="top", ha="left", fontsize=9, family='monospace')
    ax_quality.axis("off")
    ax_quality.set_title("Quality Checks")
    
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _package_version() -> str:
    try:
        return version("cross-sensor-cal")
    except PackageNotFoundError:
        from . import __version__

        return __version__


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
    except Exception:
        return "unknown"
    return out.decode("utf-8").strip()


def _provenance(prefix: str, inputs: Iterable[Path]) -> Provenance:
    hashes = {path.name: _hash_file(path) for path in inputs if path.exists()}
    return Provenance(
        flightline_id=prefix,
        created_utc=_dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        package_version=_package_version(),
        git_sha=_git_sha(),
        input_hashes=hashes,
    )


def _render_hist(ax: Axes, raw: np.ndarray, corr: np.ndarray, mask: np.ndarray) -> None:
    ax.set_title("Pre vs Post Histograms")
    masked_raw = np.where(mask, raw, np.nan)
    masked_corr = np.where(mask, corr, np.nan)
    bins = np.linspace(0, 1.2, 40)
    ax.hist(masked_raw.flatten(), bins=bins, alpha=0.4, label="Raw", color="#1f77b4")
    ax.hist(masked_corr.flatten(), bins=bins, alpha=0.4, label="Corrected", color="#ff7f0e")
    ax.set_xlabel("Reflectance")
    ax.legend(loc="upper right")


def _render_delta(ax: Axes, wavelengths: np.ndarray, report: CorrectionReport) -> None:
    xs = wavelengths if wavelengths.size == len(report.delta_median) else np.arange(len(report.delta_median))
    ax.set_title("Δ Median vs λ")
    ax.plot(xs, report.delta_median, label="Δ median")
    ax.fill_between(
        xs,
        np.array(report.delta_median) - np.array(report.delta_iqr) / 2,
        np.array(report.delta_median) + np.array(report.delta_iqr) / 2,
        alpha=0.2,
        label="IQR",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Wavelength (nm)" if wavelengths.size == len(report.delta_median) else "Band index")
    ax.set_ylabel("Reflectance Δ")
    ax.legend(loc="upper right")


def _render_scatter(ax: Axes, data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    ax.set_title("Convolved vs Corrected")
    if not data:
        ax.text(0.5, 0.5, "No convolved sensors", ha="center", va="center")
        return
    for sensor, pair in data.items():
        corrected, convolved = pair
        ax.scatter(corrected, convolved, s=3, alpha=0.3, label=sensor)
    limits_x = ax.get_xlim()
    limits_y = ax.get_ylim()
    low = min(limits_x[0], limits_y[0])
    high = max(limits_x[1], limits_y[1])
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Corrected reflectance")
    ax.set_ylabel("Convolved reflectance")
    ax.legend(loc="upper left", fontsize="small")


def _render_footer(fig: Figure, metrics: QAMetrics) -> None:
    prov = metrics.provenance
    hashes = ", ".join(f"{k}:{v[:8]}" for k, v in prov.input_hashes.items())
    footer = (
        f"Flightline: {prov.flightline_id} | UTC: {prov.created_utc} | "
        f"Package: {prov.package_version} | Git: {prov.git_sha}\nInputs: {hashes}"
    )
    fig.text(0.01, 0.02, footer, fontsize=8, ha="left", va="bottom")


def _render_issues(ax: Axes, issues: list[str]) -> None:
    if not issues:
        return
    message = "\n".join(f"⚠ {issue}" for issue in issues[:4])
    ax.text(
        0.01,
        0.99,
        message,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round", facecolor="#c1121f", alpha=0.8, edgecolor="none"),
    )


def render_flightline_panel(
    flightline_dir: Path,
    quick: bool = False,
    save_json: bool = True,
    n_sample: int = 100_000,
    rgb_bands: str | None = None,
) -> tuple[Path, dict]:
    """Return (png_path, metrics_dict); also writes _qa.json/_qa.pdf when requested."""
    
    # Debug: Print which file is being used and version identifier
    import inspect
    import os
    try:
        current_file = inspect.getfile(render_flightline_panel)
        print(f"[QA] 📍 QA plots module file: {current_file}")
        print("[QA] ✅ QA plots version: 2025-01-20-v2 (includes Page 4: Parquet & Merge Quality)")
        
        # Verify file modification time
        if os.path.exists(current_file):
            mtime = os.path.getmtime(current_file)
            import datetime
            mod_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[QA] 📅 File modification time: {mod_time}")
        
        # Verify Page 4 function exists
        if '_render_page4_parquet_merge_quality' in globals():
            print("[QA] ✅ Page 4 function found in module")
        else:
            print("[QA] ❌ WARNING: Page 4 function NOT found in module!")
    except Exception as e:
        print(f"[QA] ⚠️  Could not inspect module file: {e}")
    
    flightline_dir = Path(flightline_dir)
    raw_path, corr_path = _discover_primary_cube(flightline_dir)
    prefix = _flightline_prefix(raw_path)
    
    # Check if header files exist before trying to read them
    raw_hdr_path = raw_path.with_suffix(".hdr")
    corr_hdr_path = corr_path.with_suffix(".hdr")
    
    if not raw_hdr_path.exists():
        raise FileNotFoundError(
            f"Missing raw ENVI header: {raw_hdr_path}\n"
            f"Raw ENVI image exists at: {raw_path}\n"
            f"Flightline directory: {flightline_dir}"
        )
    if not corr_hdr_path.exists():
        raise FileNotFoundError(
            f"Missing corrected ENVI header: {corr_hdr_path}\n"
            f"Corrected ENVI image exists at: {corr_path}\n"
            f"Flightline directory: {flightline_dir}"
        )
    
    raw_hdr = hdr_to_dict(raw_hdr_path)
    corr_hdr = hdr_to_dict(corr_hdr_path)
    raw_cube = read_envi_cube(raw_path, raw_hdr)
    corr_cube = read_envi_cube(corr_path, corr_hdr)
    if raw_cube.shape != corr_cube.shape:
        raise ValueError("Raw and corrected cubes must share shape for QA panel")

    rgb_targets = _rgb_targets_from_arg(rgb_bands)

    wavelengths, wavelength_source = wavelengths_from_hdr(corr_hdr)
    header_report = _header_report(corr_hdr, wavelengths, wavelength_source)

    valid_mask = np.isfinite(corr_cube)
    mask_report = MaskReport(
        n_total=int(valid_mask.size),
        n_valid=int(np.count_nonzero(valid_mask)),
        valid_pct=float(100.0 * np.count_nonzero(valid_mask) / valid_mask.size if valid_mask.size else 0.0),
    )

    max_samples = min(n_sample, 25_000 if quick else n_sample)
    corr_sample, sample_mask = _deterministic_sample(corr_cube, valid_mask, max_samples)
    raw_sample, _ = _deterministic_sample(raw_cube, valid_mask, max_samples)

    correction_report = _correction_report(raw_sample, corr_sample, sample_mask)

    negatives_pct = float(
        100.0
        * np.count_nonzero((corr_cube < 0) & valid_mask)
        / mask_report.n_valid
        if mask_report.n_valid
        else 0.0
    )

    issues: list[str] = []
    if header_report.keys_missing:
        issues.append(f"Header missing: {', '.join(header_report.keys_missing)}")
    if header_report.wavelengths_monotonic is False:
        issues.append("Wavelengths not strictly increasing")
    if wavelength_source == "sensor_default":
        issues.append("Wavelengths missing from header – using sensor defaults")
    elif wavelength_source == "absent":
        issues.append("Wavelength metadata absent from header")
    if negatives_pct > _NEGATIVE_WARN_THRESHOLD:
        issues.append(f"{negatives_pct:.2f}% negative pixels")
    overbright_pct = float(
        100.0
        * np.count_nonzero((corr_cube > 1.2) & valid_mask)
        / mask_report.n_valid
        if mask_report.n_valid
        else 0.0
    )
    if overbright_pct > _OVERBRIGHT_WARN_THRESHOLD:
        issues.append(f"{overbright_pct:.2f}% pixels > 1.2 reflectance")
    if any(abs(val) > _DELTA_WARN_THRESHOLD for val in correction_report.delta_median):
        issues.append("Large correction deltas detected")

    conv_reports, scatter_data, brightness_map = _convolution_reports(
        flightline_dir, prefix, corr_cube, sample_mask
    )

    brightness_summary = _build_brightness_summary_table(brightness_map)

    provenance = _provenance(prefix, [raw_path, corr_path])

    metrics = QAMetrics(
        provenance=provenance,
        header=header_report,
        mask=mask_report,
        correction=correction_report,
        convolution=conv_reports,
        negatives_pct=negatives_pct,
        overbright_pct=overbright_pct,
        issues=issues,
        brightness_coefficients=brightness_map,
        brightness_summary=brightness_summary,
    )

    rgb_image, rgb_indices = _rgb_preview(corr_cube, wavelengths, rgb_targets)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"QA panel – {prefix}")

    ax_rgb = axes[0, 0]
    ax_rgb.imshow(np.clip(rgb_image, 0, 1))
    ax_rgb.set_title(
        f"RGB preview (bands {rgb_indices[0]+1}/{rgb_indices[1]+1}/{rgb_indices[2]+1})"
    )
    ax_rgb.axis("off")
    _render_issues(ax_rgb, issues)

    _render_hist(axes[0, 1], raw_sample, corr_sample, sample_mask)
    _render_delta(axes[1, 0], wavelengths, correction_report)
    _render_scatter(axes[1, 1], scatter_data)

    for ax in axes.flat:
        if ax is not ax_rgb:
            ax.grid(True, alpha=0.2)

    _render_footer(fig, metrics)

    png_path = flightline_dir / f"{prefix}_qa.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    metrics_dict_full = json.loads(metrics.to_json())
    core_keys = [
        "provenance",
        "header",
        "mask",
        "correction",
        "convolution",
        "negatives_pct",
        "overbright_pct",
        "issues",
    ]
    metrics_dict = {key: metrics_dict_full[key] for key in core_keys}

    if save_json:
        json_path = flightline_dir / f"{prefix}_qa.json"
        write_json(metrics, json_path)

    pdf_path = flightline_dir / f"{prefix}_qa.pdf"
    print(f"[QA] 📄 Generating QA PDF with 4 pages: {pdf_path.name}")
    with PdfPages(pdf_path) as pdf:
        print("[QA]   📑 Rendering Page 1: ENVI overview...")
        _render_page1_envi_overview(pdf, flightline_dir, prefix, rgb_targets)
        
        # Load filtered parquet statistics if available
        print("[QA]   📊 Loading filtered parquet statistics...")
        filtered_stats = _load_filtered_parquet_stats(flightline_dir, prefix)
        if filtered_stats:
            print(f"[QA]     ✅ Found filtered stats: {filtered_stats['n_rows']:,} rows")
        else:
            print("[QA]     ⚠️  No filtered parquet stats available")
        
        print("[QA]   📑 Rendering Page 2: Topographic + BRDF diagnostics...")
        _render_page2_topo_brdf(
            pdf=pdf,
            prefix=prefix,
            wavelengths=wavelengths,
            correction_report=correction_report,
            raw_sample=raw_sample,
            corr_sample=corr_sample,
            sample_mask=sample_mask,
            corr_path=corr_path,
            filtered_stats=filtered_stats,
        )
        print("[QA]   📑 Rendering Page 3: Additional QA diagnostics...")
        _render_page3_remaining(
            pdf=pdf,
            prefix=prefix,
            metrics=metrics,
            scatter_data=scatter_data,
            filtered_stats=filtered_stats,
        )
        
        # Add Page 4: Parquet and merge quality
        print("[QA]   📑 Rendering Page 4: Parquet & Merge Quality Analysis...")
        try:
            _render_page4_parquet_merge_quality(
                pdf=pdf,
                flightline_dir=flightline_dir,
                prefix=prefix,
            )
            print("[QA]   ✅ Page 4 rendered successfully")
        except Exception as e:
            print(f"[QA]   ❌ ERROR rendering Page 4: {e}")
            import traceback
            traceback.print_exc()
            # Continue anyway - 3 pages is still useful
    
    # Verify PDF was created and count pages
    try:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            print(f"[QA] ✅ QA PDF complete: {pdf_path.name} ({num_pages} pages)")
            if num_pages < 4:
                print(f"[QA] ⚠️  WARNING: Expected 4 pages but PDF has {num_pages} pages!")
                print("[QA]     This may indicate the old code is still being used.")
            elif num_pages == 4:
                print("[QA] ✅ PDF has 4 pages as expected!")
        except ImportError:
            print(f"[QA] ✅ QA PDF complete: {pdf_path.name} (PyPDF2 not available to verify page count)")
    except Exception as e:
        print(f"[QA] ✅ QA PDF complete: {pdf_path.name} (could not verify page count: {e})")

    return png_path, metrics_dict


__all__ = ["render_flightline_panel"]
