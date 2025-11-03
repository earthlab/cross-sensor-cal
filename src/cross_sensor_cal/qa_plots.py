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
_DELTA_WARN_THRESHOLD = 0.05


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_primary_cube(flightline_dir: Path) -> tuple[Path, Path]:
    raw_candidates = sorted(
        p
        for p in flightline_dir.glob("*_envi.img")
        if "corrected" not in p.stem and "qa" not in p.stem
    )
    if not raw_candidates:
        raise FileNotFoundError(f"No ENVI cube found in {flightline_dir}")
    raw_path = raw_candidates[0]
    prefix = raw_path.stem.rsplit("_envi", 1)[0]
    corrected = flightline_dir / f"{prefix}_brdfandtopo_corrected_envi.img"
    if not corrected.exists():
        raise FileNotFoundError(f"Missing corrected cube: {corrected}")
    return raw_path, corrected


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
        indices = [int(np.nanargmin(np.abs(wavelengths - target))) for target in rgb_targets]
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
    diff = np.where(sample_mask, corr_sample - raw_sample, np.nan)
    delta_median = np.nanmedian(diff, axis=1)
    q75 = np.nanpercentile(diff, 75, axis=1)
    q25 = np.nanpercentile(diff, 25, axis=1)
    delta_iqr = q75 - q25
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
    for img_path in sorted(base_dir.glob(f"{prefix}_*_convolved_envi.img")):
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
        ax.set_title(img_path.stem, fontsize=8)
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
) -> None:
    """Page 2: diagnostics specific to topo (row 1) and BRDF (row 2) corrections."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Topographic + BRDF diagnostics – {prefix}")

    ax_topo_hist = axes[0, 0]
    _render_hist(ax_topo_hist, raw_sample, corr_sample, sample_mask)
    ax_topo_hist.set_title("Topographic + BRDF: pre vs post histograms")

    ax_topo_delta = axes[0, 1]
    _render_delta(ax_topo_delta, wavelengths, correction_report)
    ax_topo_delta.set_title("Topographic + BRDF: Δ median vs λ")

    params = _load_correction_geometry_json(corr_path.parent, corr_path) or {}
    geom = params.get("geometry", {}) if isinstance(params, dict) else {}
    topo_keys = ["slope", "aspect"]
    brdf_keys = ["solar_zn", "solar_az", "sensor_zn", "sensor_az"]

    ax_brdf_geom = axes[1, 0]
    if geom:
        topo_present = [k for k in topo_keys if k in geom]
        if topo_present:
            means = []
            for k in topo_present:
                value = geom[k]
                if isinstance(value, dict):
                    means.append(value.get("mean"))
                else:
                    means.append(value)
            ax_brdf_geom.bar(topo_present, means)
            ax_brdf_geom.set_title("Topographic geometry (mean)")
            ax_brdf_geom.set_ylabel("Value (radians)")
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
            for k in brdf_present:
                value = geom[k]
                if isinstance(value, dict):
                    means.append(value.get("mean"))
                else:
                    means.append(value)
            ax_brdf_geom2.bar(brdf_present, means)
            ax_brdf_geom2.set_title("BRDF geometry (mean)")
            ax_brdf_geom2.set_ylabel("Value (radians)")
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
) -> None:
    """Page 3: remaining QA diagnostics (convolution + header/mask/issue summary)."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Additional QA diagnostics – {prefix}")

    ax_scatter = axes[0, 0]
    _render_scatter(ax_scatter, scatter_data)
    ax_scatter.set_title("Convolved vs corrected (all sensors)")

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
    lines = [
        f"Total pixels: {mask.n_total}",
        f"Valid pixels: {mask.n_valid}",
        f"Valid %: {mask.valid_pct:.2f}%",
        f"Negatives %: {negatives_pct:.2f}%",
    ]
    ax_mask.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9)
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

    flightline_dir = Path(flightline_dir)
    raw_path, corr_path = _discover_primary_cube(flightline_dir)
    prefix = _flightline_prefix(raw_path)
    raw_hdr = hdr_to_dict(raw_path.with_suffix(".hdr"))
    corr_hdr = hdr_to_dict(corr_path.with_suffix(".hdr"))
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
    if negatives_pct > _NEGATIVE_WARN_THRESHOLD:
        issues.append(f"{negatives_pct:.2f}% negative pixels")
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
        "issues",
    ]
    metrics_dict = {key: metrics_dict_full[key] for key in core_keys}

    if save_json:
        json_path = flightline_dir / f"{prefix}_qa.json"
        write_json(metrics, json_path)

    pdf_path = flightline_dir / f"{prefix}_qa.pdf"
    with PdfPages(pdf_path) as pdf:
        _render_page1_envi_overview(pdf, flightline_dir, prefix, rgb_targets)
        _render_page2_topo_brdf(
            pdf=pdf,
            prefix=prefix,
            wavelengths=wavelengths,
            correction_report=correction_report,
            raw_sample=raw_sample,
            corr_sample=corr_sample,
            sample_mask=sample_mask,
            corr_path=corr_path,
        )
        _render_page3_remaining(
            pdf=pdf,
            prefix=prefix,
            metrics=metrics,
            scatter_data=scatter_data,
        )

    return png_path, metrics_dict


__all__ = ["render_flightline_panel"]
