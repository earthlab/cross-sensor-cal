"""QA panel rendering with deterministic sampling and metrics JSON sidecars."""

from __future__ import annotations

import hashlib
import logging
import math
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import __version__ as PACKAGE_VERSION
from .envi import hdr_to_dict, memmap_bsq, to_unitless_reflectance
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

RGB_TARGETS = {"R": 660.0, "G": 560.0, "B": 490.0}
SENSOR_RGB_INDEX = {
    "landsat_tm": (2, 1, 0),
    "landsat_etm+": (2, 1, 0),
    "landsat_oli": (3, 2, 1),
    "landsat_oli2": (3, 2, 1),
    "micasense": (2, 1, 0),
}


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:  # pragma: no cover - git metadata optional
        return "unknown"


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_inputs(flightline_dir: Path) -> Dict[str, Path]:
    corrected = sorted(flightline_dir.glob("*_brdfandtopo_corrected_envi.img"))
    raw = [
        p
        for p in sorted(flightline_dir.glob("*_envi.img"))
        if "brdfandtopo_corrected" not in p.name
    ]
    masks = sorted(flightline_dir.glob("*_mask*.img"))
    sensors: list[Path] = []
    for pattern in ("*_landsat_*_envi.img", "*_micasense*_envi.img"):
        sensors.extend(sorted(flightline_dir.glob(pattern)))
    parquet = sorted(flightline_dir.glob("*_merged_pixel_extraction.parquet"))
    chosen: Dict[str, Path] = {}
    if raw:
        chosen["raw_img"] = raw[0]
        raw_hdr = raw[0].with_suffix(".hdr")
        if raw_hdr.exists():
            chosen["raw_hdr"] = raw_hdr
    if corrected:
        chosen["corrected_img"] = corrected[0]
        corr_hdr = corrected[0].with_suffix(".hdr")
        if corr_hdr.exists():
            chosen["corrected_hdr"] = corr_hdr
    if masks:
        chosen["mask_img"] = masks[0]
        mask_hdr = masks[0].with_suffix(".hdr")
        if mask_hdr.exists():
            chosen["mask_hdr"] = mask_hdr
    if sensors:
        chosen["sensor_img"] = sensors[0]
        sensor_hdr = sensors[0].with_suffix(".hdr")
        if sensor_hdr.exists():
            chosen["sensor_hdr"] = sensor_hdr
    if parquet:
        chosen["parquet"] = parquet[0]
    return chosen


def _grid_coordinates(lines: int, samples: int, max_points: int) -> List[Tuple[int, int]]:
    if max_points <= 0:
        return []
    target = min(max_points, lines * samples)
    grid_side = max(1, int(math.sqrt(target)))
    step_y = max(1, lines // grid_side)
    step_x = max(1, samples // grid_side)
    coords: list[Tuple[int, int]] = []
    for y0 in range(0, lines, step_y):
        for x0 in range(0, samples, step_x):
            y = min(lines - 1, y0 + step_y // 2)
            x = min(samples - 1, x0 + step_x // 2)
            coords.append((y, x))
            if len(coords) >= target:
                return coords
    return coords


def _sample_cube(img_path: Path, header: dict, coords: Sequence[Tuple[int, int]]) -> np.ndarray:
    cube = memmap_bsq(img_path, header)
    data = np.stack([cube[:, y, x] for y, x in coords], axis=0)
    return to_unitless_reflectance(np.asarray(data, dtype=np.float32))


def _select_rgb_indices(
    wavelengths: np.ndarray,
    n_bands: int,
    rgb_override: Optional[str],
    sensor_hint: str,
) -> Tuple[List[int], str]:
    if rgb_override:
        parts = [int(p.strip()) for p in rgb_override.split(",") if p.strip()]
        if len(parts) == 3:
            indices = [max(0, min(n_bands - 1, p)) for p in parts]
            return indices, "override"
    if wavelengths.size >= 3:
        indices = []
        for key in ("R", "G", "B"):
            tgt = RGB_TARGETS[key]
            idx = int(np.nanargmin(np.abs(wavelengths - tgt)))
            indices.append(idx)
        return indices, "header"
    if sensor_hint in SENSOR_RGB_INDEX:
        idx = [min(n_bands - 1, v) for v in SENSOR_RGB_INDEX[sensor_hint]]
        return idx, "sensor default"
    return [min(n_bands - 1, i) for i in (0, 1, 2)], "fallback"


def _rgb_quicklook(cube: np.ndarray) -> np.ndarray:
    if cube.ndim != 3 or cube.shape[2] != 3:
        raise ValueError("RGB cube expected with shape (lines, samples, 3)")
    arr = np.clip(cube, -0.05, 1.2)
    arr = (arr - arr.min()) / max(1e-6, arr.max() - arr.min())
    return np.clip(arr, 0.0, 1.0)


def _infer_sensor_key(path: Path) -> str:
    name = path.name.lower()
    for key in SENSOR_RGB_INDEX:
        if key in name:
            return key
    return "landsat_oli"


def _compute_header_report(
    header: dict,
    wavelengths: np.ndarray,
    wavelength_source: str,
) -> HeaderReport:
    required = ["samples", "lines", "bands", "wavelength", "fwhm", "band names"]
    keys_present = [k for k in required if k in {key.lower(): key for key in header}.values()]
    keys_missing = [k for k in required if k not in keys_present]
    unit = None
    for key in ("wavelength units", "wavelength_unit", "units"):
        if key in header:
            unit = str(header[key])
            break
    finite = wavelengths[np.isfinite(wavelengths)] if wavelengths.size else np.array([])
    n_wavelengths_finite = int(finite.size)
    monotonic: Optional[bool]
    first_nm: Optional[float]
    last_nm: Optional[float]
    if finite.size >= 2:
        diffs = np.diff(finite)
        monotonic = bool(np.all(diffs > 0))
        first_nm = float(finite[0])
        last_nm = float(finite[-1])
    elif finite.size == 1:
        monotonic = None
        first_nm = last_nm = float(finite[0])
    else:
        monotonic = None
        first_nm = last_nm = None
    return HeaderReport(
        keys_present=sorted(keys_present),
        keys_missing=sorted(keys_missing),
        wavelength_unit=unit,
        n_bands=int(header.get("bands", 0) or 0),
        n_wavelengths_finite=n_wavelengths_finite,
        first_nm=first_nm,
        last_nm=last_nm,
        wavelengths_monotonic=monotonic,
        wavelength_source=wavelength_source,
    )


def _issues_from_metrics(metrics: QAMetrics) -> List[str]:
    issues: list[str] = []
    if metrics.header.n_wavelengths_finite == 0:
        issues.append("missing_wavelengths")
    if metrics.header.wavelength_source == "absent":
        issues.append("wavelengths_absent")
    if metrics.header.wavelength_unit is None:
        issues.append("missing_wavelength_unit")
    if metrics.header.wavelengths_monotonic is False:
        issues.append("non_monotonic_wavelengths")
    if metrics.mask.valid_pct < 0.5:
        issues.append("low_mask_valid_pct")
    if metrics.negatives_pct > 0.05:
        issues.append("high_negatives")
    if metrics.correction and any(abs(v) > 0.1 for v in metrics.correction.delta_median):
        issues.append("large_correction_delta")
    return sorted(set(metrics.issues + issues))


def render_flightline_panel(
    flightline_dir: Path,
    quick: bool = False,
    save_json: bool = True,
    n_sample: int = 100_000,
    rgb_bands: str | None = None,
) -> tuple[Path, dict]:
    flightline_dir = Path(flightline_dir)
    if not flightline_dir.exists():
        raise FileNotFoundError(f"Flightline directory missing: {flightline_dir}")

    inputs = _discover_inputs(flightline_dir)
    if "corrected_img" not in inputs:
        raise FileNotFoundError("Corrected ENVI cube not found for QA panel")

    corrected_img = inputs["corrected_img"]
    corrected_stem = corrected_img.stem
    if corrected_stem.endswith("_brdfandtopo_corrected_envi"):
        prefix = corrected_stem[: -len("_brdfandtopo_corrected_envi")]
    elif corrected_stem.endswith("_envi"):
        prefix = corrected_stem[: -len("_envi")]
    else:
        prefix = corrected_stem
    corrected_hdr = hdr_to_dict(corrected_img.with_suffix(".hdr"))
    raw_img = inputs.get("raw_img")
    raw_hdr = hdr_to_dict(raw_img.with_suffix(".hdr")) if raw_img else None

    sample_limit = min(n_sample, 100_000)
    if quick:
        sample_limit = min(sample_limit, 10_000)

    cube = memmap_bsq(corrected_img, corrected_hdr)
    bands, lines, samples = cube.shape
    coords = _grid_coordinates(lines, samples, sample_limit)
    if not coords:
        raise RuntimeError("No sample coordinates derived for QA panel")

    corrected_samples = _sample_cube(corrected_img, corrected_hdr, coords)
    raw_samples = (
        _sample_cube(raw_img, raw_hdr, coords) if raw_img and raw_hdr else None
    )

    pixel_mask = np.isfinite(corrected_samples).all(axis=1)
    valid_samples = corrected_samples[pixel_mask]
    if raw_samples is not None:
        raw_samples = raw_samples[pixel_mask]

    mask_report = MaskReport(
        n_total=len(coords),
        n_valid=int(pixel_mask.sum()),
        valid_pct=float(pixel_mask.mean() if coords else 0.0),
    )

    negatives_pct = float(
        np.mean(valid_samples < 0.0) if valid_samples.size else 0.0
    )

    sensor_hint = _infer_sensor_key(corrected_img)
    wavelengths, source = wavelengths_from_hdr(corrected_hdr)
    header_report = _compute_header_report(corrected_hdr, wavelengths, source)

    correction_report: Optional[CorrectionReport]
    delta_med: list[float] = []
    delta_iqr: list[float] = []
    largest: list[int] = []
    if raw_samples is not None and raw_samples.size:
        delta = valid_samples - raw_samples
        delta_med = np.nanmedian(delta, axis=0).tolist()
        q75 = np.nanpercentile(delta, 75, axis=0)
        q25 = np.nanpercentile(delta, 25, axis=0)
        delta_iqr = (q75 - q25).tolist()
        if delta.shape[1]:
            idx = np.argsort(np.abs(np.nanmedian(delta, axis=0)))
            largest = [int(i) for i in idx[::-1][:3]]
        correction_report = CorrectionReport(
            delta_median=[float(x) for x in delta_med],
            delta_iqr=[float(x) for x in delta_iqr],
            largest_delta_indices=largest,
        )
    else:
        correction_report = None

    # Convolution reports (first sensor product only for metrics/plotting)
    convolution_reports: list[ConvolutionReport] = []
    conv_plot_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, Optional[float]]]
    conv_plot_data = []
    sensor_img = inputs.get("sensor_img")
    if sensor_img is not None:
        sensor_hdr = hdr_to_dict(sensor_img.with_suffix(".hdr"))
        sensor_samples = _sample_cube(sensor_img, sensor_hdr, coords)
        sensor_samples = sensor_samples[pixel_mask]
        sensor_wl, _ = wavelengths_from_hdr(sensor_hdr)
        if wavelengths.size and sensor_wl.size:
            mapping = [int(np.nanargmin(np.abs(wavelengths - w))) for w in sensor_wl]
            expected = valid_samples[:, mapping]
        else:
            expected = valid_samples[:, : sensor_samples.shape[1]]
        diff = sensor_samples - expected
        rmse = np.sqrt(np.nanmean(diff**2, axis=0))
        sam: Optional[float]
        norms = np.linalg.norm(expected, axis=1) * np.linalg.norm(sensor_samples, axis=1)
        mask = norms > 0
        if np.any(mask):
            cosang = np.clip(
                np.sum(expected[mask] * sensor_samples[mask], axis=1) / norms[mask],
                -1.0,
                1.0,
            )
            sam = float(np.degrees(np.nanmean(np.arccos(cosang))))
        else:
            sam = None
        convolution_reports.append(
            ConvolutionReport(
                sensor=_infer_sensor_key(sensor_img),
                rmse=rmse.tolist(),
                sam=sam,
            )
        )
        conv_plot_data.append((sensor_img.name, expected, sensor_samples, rmse, sam))

    provenance = Provenance(
        flightline_id=flightline_dir.name,
        created_utc=datetime.now(timezone.utc).isoformat(),
        package_version=PACKAGE_VERSION,
        git_sha=_git_sha(),
        input_hashes={
            key: _sha1(path)
            for key, path in inputs.items()
            if path.exists() and path.is_file()
        },
    )

    metrics = QAMetrics(
        provenance=provenance,
        header=header_report,
        mask=mask_report,
        correction=correction_report,
        convolution=convolution_reports,
        negatives_pct=negatives_pct,
        issues=[],
    )
    metrics.issues = _issues_from_metrics(metrics)

    rgb_indices, rgb_source = _select_rgb_indices(
        wavelengths,
        header_report.n_bands,
        rgb_bands,
        sensor_hint,
    )
    rgb_cube = np.stack([cube[idx, :, :] for idx in rgb_indices], axis=-1)
    rgb_image = _rgb_quicklook(to_unitless_reflectance(rgb_cube))

    x_axis = wavelengths if wavelengths.size else np.arange(valid_samples.shape[1])
    title_axis = "Wavelength (nm)" if wavelengths.size else "Band index"

    png_path = flightline_dir / f"{prefix}_qa.png"

    fig = plt.figure(figsize=(12, 8), dpi=150)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], height_ratios=[1.1, 1.0, 1.0])
    ax_rgb = fig.add_subplot(gs[:, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_delta = fig.add_subplot(gs[1, 1])
    ax_conv = fig.add_subplot(gs[2, 1])

    ax_rgb.imshow(rgb_image)
    ax_rgb.set_title("RGB quicklook")
    ann = ", ".join(f"{band}@{idx}" for band, idx in zip("RGB", rgb_indices))
    ax_rgb.text(
        0.01,
        0.99,
        f"Bands: {ann} ({rgb_source})",
        ha="left",
        va="top",
        transform=ax_rgb.transAxes,
        fontsize=9,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
    )
    ax_rgb.axis("off")

    flat_corr = valid_samples.flatten()
    flat_raw = raw_samples.flatten() if raw_samples is not None else None
    bins = np.linspace(-0.05, 1.2, 60)
    ax_hist.hist(flat_corr, bins=bins, alpha=0.6, label="Corrected")
    if flat_raw is not None:
        ax_hist.hist(flat_raw, bins=bins, alpha=0.6, label="Raw")
    ax_hist.set_xlabel("Reflectance")
    ax_hist.set_ylabel("Count")
    ax_hist.legend()
    med_corr = float(np.nanmedian(flat_corr)) if flat_corr.size else float("nan")
    med_raw = (
        float(np.nanmedian(flat_raw))
        if flat_raw is not None and flat_raw.size
        else float("nan")
    )
    delta_median = med_corr - med_raw if not math.isnan(med_raw) else float("nan")
    ax_hist.text(
        0.02,
        0.95,
        f"Δmedian={delta_median:.3f}\n%<0={metrics.negatives_pct*100:.2f}",
        transform=ax_hist.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    if delta_med:
        med = np.array(delta_med)
        iqr = np.array(delta_iqr)
        ax_delta.plot(x_axis[: med.size], med, color="tab:orange", label="Δ median")
        ax_delta.fill_between(
            x_axis[: med.size],
            med - 0.5 * iqr,
            med + 0.5 * iqr,
            color="tab:orange",
            alpha=0.3,
            label="IQR",
        )
    ax_delta.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax_delta.set_xlabel(title_axis)
    ax_delta.set_ylabel("Δ reflectance")
    ax_delta.set_title("Correction deltas")

    if conv_plot_data:
        for name, expected, observed, rmse, sam in conv_plot_data:
            mean_expected = np.nanmean(expected, axis=0)
            mean_observed = np.nanmean(observed, axis=0)
            ax_conv.scatter(mean_expected, mean_observed, label=name)
            ax_conv.plot(
                [0, 1],
                [0, 1],
                color="black",
                linestyle=":",
                linewidth=0.8,
            )
            ax_conv.text(
                0.02,
                0.95,
                f"RMSE={np.nanmean(rmse):.3f}\nSAM={(sam or float('nan')):.2f}",
                transform=ax_conv.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
            )
        ax_conv.set_xlabel("Expected")
        ax_conv.set_ylabel("Observed")
        ax_conv.set_title("Convolution check")
        ax_conv.legend(loc="lower right", fontsize=8)
    else:
        ax_conv.axis("off")
        ax_conv.text(0.5, 0.5, "No convolved products", ha="center", va="center")

    footer_hash = ", ".join(
        f"{k}:{v[:8]}" for k, v in provenance.input_hashes.items()
    )
    footer = (
        f"{provenance.flightline_id} | {provenance.created_utc} | "
        f"cross-sensor-cal {provenance.package_version} | git {provenance.git_sha}\n"
        f"Inputs: {footer_hash}"
    )
    fig.text(0.01, 0.01, footer, fontsize=8, ha="left", va="bottom")

    if metrics.issues:
        issue_text = "\n".join(f"⚠ {issue}" for issue in metrics.issues)
        ax_rgb.text(
            0.01,
            0.01,
            f"{issue_text}\nSee JSON for details",
            transform=ax_rgb.transAxes,
            ha="left",
            va="bottom",
            color="red",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="red", boxstyle="round,pad=0.3"),
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    json_path = png_path.with_suffix(".json")
    if save_json:
        write_json(metrics, json_path)

    metrics_dict = asdict(metrics)
    return png_path, metrics_dict


def make_flightline_panel(
    flightline_dir: Path,
    panel_png: Path,
    *,
    quick: bool = False,
    save_json: bool = True,
    n_sample: int = 100_000,
    rgb_bands: str | None = None,
) -> Optional[Path]:
    png_path, metrics = render_flightline_panel(
        flightline_dir,
        quick=quick,
        save_json=save_json,
        n_sample=n_sample,
        rgb_bands=rgb_bands,
    )
    panel_png = Path(panel_png)
    panel_png.parent.mkdir(parents=True, exist_ok=True)
    json_src = png_path.with_suffix(".json") if save_json else None
    if panel_png != png_path:
        Path(png_path).replace(panel_png)
        if save_json and json_src and Path(json_src).exists():
            json_dst = panel_png.with_suffix(".json")
            Path(json_src).replace(json_dst)
    return panel_png


def summarize_flightline_outputs(
    base_folder: Path,
    flight_stem: str,
    out_png: Path | None = None,
    *,
    quick: bool = False,
    save_json: bool = True,
    n_sample: int = 100_000,
    rgb_bands: str | None = None,
):
    base_folder = Path(base_folder)
    flight_dir = base_folder / flight_stem
    if out_png is None:
        out_png = flight_dir / f"{flight_stem}_qa.png"
    png_path, metrics = render_flightline_panel(
        flight_dir,
        quick=quick,
        save_json=save_json,
        n_sample=n_sample,
        rgb_bands=rgb_bands,
    )
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png != png_path:
        Path(png_path).replace(out_png)
        if save_json:
            json_src = png_path.with_suffix(".json")
            if json_src.exists():
                json_dst = out_png.with_suffix(".json")
                json_src.replace(json_dst)
    return out_png


def summarize_all_flightlines(
    base_folder: Path,
    out_dir: Path | None = None,
    *,
    quick: bool = False,
    save_json: bool = True,
    n_sample: int = 100_000,
    rgb_bands: str | None = None,
):
    base_folder = Path(base_folder)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for child in sorted(base_folder.iterdir()):
        if not child.is_dir():
            continue
        target_png = (
            out_dir / f"{child.name}_qa.png" if out_dir is not None else child / f"{child.name}_qa.png"
        )
        try:
            png_path, metrics = render_flightline_panel(
                child,
                quick=quick,
                save_json=save_json,
                n_sample=n_sample,
                rgb_bands=rgb_bands,
            )
            if out_dir is not None:
                target_png.parent.mkdir(parents=True, exist_ok=True)
                if png_path != target_png:
                    Path(png_path).replace(target_png)
                    if save_json:
                        json_src = png_path.with_suffix(".json")
                        if json_src.exists():
                            json_dst = target_png.with_suffix(".json")
                            json_src.replace(json_dst)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to summarize %s: %s", child.name, exc)


__all__ = [
    "render_flightline_panel",
    "make_flightline_panel",
    "summarize_flightline_outputs",
    "summarize_all_flightlines",
]

