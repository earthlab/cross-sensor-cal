"""Generate quicklook QA plots for processed flightline outputs."""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .envi import (
    band_axis_from_header,
    hdr_to_dict,
    memmap_bsq,
    wavelength_array,
)


logger = logging.getLogger(__name__)


def _patch_mean_spectrum(
    arr: np.ndarray,
    hdr: dict,
    patch: tuple[slice, slice] | None = None,
) -> np.ndarray:
    """
    Compute mean spectrum over a spatial patch, averaging **only** spatial axes.

    Works for arrays ordered ``(bands, rows, cols)`` or ``(rows, cols, bands)``.
    """

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    band_axis = band_axis_from_header(arr, hdr)
    spatial_axes = tuple(sorted({0, 1, 2} - {band_axis}))

    selected = arr
    if patch is not None:
        sel = [slice(None)] * 3
        spatial_list = list(spatial_axes)
        sel[spatial_list[0]] = patch[0]
        sel[spatial_list[1]] = patch[1]
        selected = selected[tuple(sel)]

    selected = np.asarray(selected, dtype=np.float32)
    selected = np.where(
        np.isfinite(selected) & (selected != 0),
        selected,
        np.nan,
    )

    spectrum = np.nanmean(selected, axis=spatial_axes)

    if np.nanstd(spectrum) < 1e-4 and band_axis != 2:
        alt_band_axis = 2
        alt_spatial_axes = tuple(sorted({0, 1, 2} - {alt_band_axis}))
        spectrum = np.nanmean(selected, axis=alt_spatial_axes)

    return spectrum


def _to_unitless_reflectance(spec: np.ndarray) -> np.ndarray:
    """
    Convert a 1D spectrum to unitless reflectance (0–1) if it appears scaled.
    """

    if spec.ndim != 1:
        raise ValueError(f"Expected 1D spectrum, got shape {spec.shape}")
    med = float(np.nanmedian(spec))
    if np.isnan(med):
        return spec
    return spec / 10000.0 if med > 1.5 else spec


def _load_envi_header(hdr_path: Path) -> dict:
    return hdr_to_dict(hdr_path)


def _unitless(arr: np.ndarray) -> np.ndarray:
    """Convert scaled reflectance values to the unitless 0–1 range heuristically."""

    med = float(np.nanmedian(arr))
    if np.isnan(med):
        return arr
    # If median > 1.5, treat as scaled-int reflectance (0–10000)
    return arr / 10000.0 if med > 1.5 else arr


def _pick_rgb_bands_for_raw(header_dict: dict) -> tuple[int, int, int]:
    target_wavs = [650, 560, 470]
    wavs_arr = wavelength_array(header_dict)
    if wavs_arr is None:
        return (0, 0, 0)
    idxs = [int(np.argmin(np.abs(wavs_arr - tw))) for tw in target_wavs]
    return tuple(idxs)  # type: ignore[return-value]


def _pick_preview_band_for_corrected(header_dict: dict) -> int:
    target = 800
    wavs_arr = wavelength_array(header_dict)
    if wavs_arr is None:
        return 0
    idx = int(np.argmin(np.abs(wavs_arr - target)))
    return idx


def _percentile_stretch(arr: np.ndarray, p_lo: float = 2, p_hi: float = 98) -> np.ndarray:
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = np.nanpercentile(arr, p_lo)
    hi = np.nanpercentile(arr, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def _format_size_mb(path: Path) -> str:
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{size_mb:.1f} MB"


def _make_rgb_composite(mm: np.memmap, band_indices: Sequence[int]) -> np.ndarray:
    bands = []
    for idx in band_indices:
        idx = int(max(0, min(mm.shape[0] - 1, idx)))
        band = np.asarray(mm[idx], dtype=np.float32)
        bands.append(_percentile_stretch(band))
    while len(bands) < 3:
        bands.append(bands[-1])
    rgb = np.stack(bands[:3], axis=-1)
    return rgb


def _safe_band_indices(header_dict: dict, desired: Sequence[int] = (0, 1, 2)) -> tuple[int, int, int]:
    nb_raw = header_dict.get("bands", 0)
    try:
        nb = int(nb_raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        nb = 0

    if nb >= 3:
        return tuple(int(i) for i in desired[:3])  # type: ignore[return-value]
    if nb == 2:
        return (0, 1, 1)
    if nb == 1:
        return (0, 0, 0)
    raise ValueError("ENVI header reports zero bands.")


def _sensor_label(img_path: Path, flight_stem: str) -> str:
    name = img_path.stem
    prefix = f"{flight_stem}_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    return name


def _gather_sensor_products(work_dir: Path, flight_stem: str) -> list[Path]:
    all_imgs = sorted(work_dir.glob("*_envi.img"))
    keep: list[Path] = []
    raw_name = f"{flight_stem}_envi.img"
    corrected_name = f"{flight_stem}_brdfandtopo_corrected_envi.img"
    for img in all_imgs:
        if img.name in {raw_name, corrected_name}:
            continue
        keep.append(img)
    return keep


def _downsample_rgb(rgb: np.ndarray, max_size: int = 200) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB array must have shape (lines, samples, 3)")
    max_dim = max(rgb.shape[0], rgb.shape[1])
    if max_dim <= max_size:
        return rgb
    step = max(1, int(math.ceil(max_dim / max_size)))
    return rgb[::step, ::step, :]


def _collect_parquet_summaries(files: Iterable[Path]) -> list[str]:
    summaries: list[str] = []
    for pq in sorted(files):
        try:
            size = _format_size_mb(pq)
        except OSError:
            size = "(size unavailable)"
        summaries.append(f"{pq.name} ({size})")
    return summaries


def summarize_flightline_outputs(
    base_folder: Path,
    flight_stem: str,
    out_png: Path | None = None,
    *,
    shaded_regions: bool = True,
    overwrite: bool = True,
) -> Figure:
    base_folder = Path(base_folder)
    work_dir = base_folder / flight_stem
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Flightline folder missing: {work_dir}")

    raw_img = work_dir / f"{flight_stem}_envi.img"
    raw_hdr = raw_img.with_suffix(".hdr")
    corrected_img = work_dir / f"{flight_stem}_brdfandtopo_corrected_envi.img"
    corrected_hdr = corrected_img.with_suffix(".hdr")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        height_ratios=[1, 1, 1],
        width_ratios=[2, 1, 1],
        hspace=0.4,
        wspace=0.4,
    )

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_spectra = fig.add_subplot(gs[0, 1:])
    corrected_spec = gs[1, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
    ax_corrected = fig.add_subplot(corrected_spec[0])
    ax_ratio_map = fig.add_subplot(corrected_spec[1])
    ax_parquet = fig.add_subplot(gs[2, 0])

    thumb_spec = gs[1:, 1:].subgridspec(1, 1)
    thumb_axes: list = []

    def _fail_axis(ax, title: str, message: str) -> None:
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)

    def _skip_spectra_panel(message: str) -> None:
        warnings.warn(message)
        ax_spectra.clear()
        ax_spectra.axis("off")
        ax_spectra.set_title("Patch spectra (skipped)")
        ax_spectra.text(0.5, 0.5, message, ha="center", va="center", wrap=True)

    # Panel A: Raw RGB
    if raw_img.exists() and raw_hdr.exists():
        try:
            raw_header = _load_envi_header(raw_hdr)
            raw_mm = memmap_bsq(raw_img, raw_header)
            rgb_indices = _pick_rgb_bands_for_raw(raw_header)
            rgb = _make_rgb_composite(raw_mm, rgb_indices)
            ax_raw.imshow(rgb)
            ax_raw.set_title("Raw ENVI export RGB (uncorrected)")
            ax_raw.axis("off")
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Failed to plot raw RGB for {flight_stem}: {exc}")
            _fail_axis(ax_raw, "Raw ENVI export RGB (uncorrected)", "Unavailable")
            raw_header = None
            raw_mm = None
    else:
        _fail_axis(
            ax_raw,
            "Raw ENVI export RGB (uncorrected)",
            "Missing raw ENVI export",
        )
        raw_header = None
        raw_mm = None

    # Panel C requires corrected header/memmap for patch selection.
    if corrected_img.exists() and corrected_hdr.exists():
        try:
            corrected_header = _load_envi_header(corrected_hdr)
            corrected_mm = memmap_bsq(corrected_img, corrected_header)
            nir_idx = _pick_preview_band_for_corrected(corrected_header)
            nir_band = np.asarray(corrected_mm[nir_idx], dtype=np.float32)
            nir_preview = _percentile_stretch(nir_band)
            ax_corrected.imshow(nir_preview, cmap="gray")
            ax_corrected.set_title("Corrected cube (BRDF+topo) — NIR band preview")
            ax_corrected.axis("off")

            if raw_mm is not None and raw_mm.shape[0] > 0:
                ratio_idx = int(min(raw_mm.shape[0] - 1, nir_idx))
                raw_ratio_band = np.asarray(raw_mm[ratio_idx], dtype=np.float32)
                corr_unit_band = _unitless(nir_band.astype(np.float32, copy=False))
                raw_unit_band = _unitless(raw_ratio_band.astype(np.float32, copy=False))
                ratio_map = np.full_like(corr_unit_band, np.nan, dtype=np.float32)
                valid_mask = (
                    np.isfinite(corr_unit_band)
                    & np.isfinite(raw_unit_band)
                    & (raw_unit_band != 0)
                )
                ratio_map[valid_mask] = corr_unit_band[valid_mask] / raw_unit_band[valid_mask]

                clipped_ratio = np.clip(ratio_map, 0.5, 1.5)
                ax_ratio_map.imshow(clipped_ratio, cmap="coolwarm", vmin=0.5, vmax=1.5)
                ax_ratio_map.set_title("Corrected/raw ratio (NIR)")
                ax_ratio_map.axis("off")

                valid_count = int(np.count_nonzero(~np.isnan(ratio_map)))
                if valid_count > 0:
                    outside = int(
                        np.count_nonzero((ratio_map < 0.2) | (ratio_map > 2.0))
                    )
                    if outside / valid_count > 0.10:
                        ax_ratio_map.text(
                            0.5,
                            0.08,
                            "Wide ratio spread — check geometry/offset",
                            transform=ax_ratio_map.transAxes,
                            ha="center",
                            color="crimson",
                            fontsize=9,
                            bbox=dict(facecolor="white", alpha=0.7),
                        )
            else:
                _fail_axis(
                    ax_ratio_map,
                    "Corrected/raw ratio (NIR)",
                    "Raw preview unavailable",
                )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Failed to plot corrected preview for {flight_stem}: {exc}")
            _fail_axis(
                ax_corrected,
                "Corrected cube (BRDF+topo) — NIR band preview",
                "Unavailable",
            )
            _fail_axis(
                ax_ratio_map,
                "Corrected/raw ratio (NIR)",
                "Unavailable",
            )
            corrected_header = None
            corrected_mm = None
            nir_band = None
    else:
        _fail_axis(
            ax_corrected,
            "Corrected cube (BRDF+topo) — NIR band preview",
            "Missing corrected ENVI cube",
        )
        _fail_axis(
            ax_ratio_map,
            "Corrected/raw ratio (NIR)",
            "Missing corrected ENVI cube",
        )
        corrected_header = None
        corrected_mm = None
        nir_band = None

    # Panel B: Spectral correction effect
    if (
        raw_header is not None
        and corrected_header is not None
        and raw_mm is not None
        and corrected_mm is not None
    ):
        raw_wavs_arr = wavelength_array(raw_header)
        corr_wavs_arr = wavelength_array(corrected_header)
        bands = 0
        raw_band_axis: int | None = None
        corr_band_axis: int | None = None
        try:
            raw_band_axis = band_axis_from_header(raw_mm, raw_header)
            corr_band_axis = band_axis_from_header(corrected_mm, corrected_header)
        except ValueError as exc:
            _skip_spectra_panel(f"Skipping spectra panel: {exc}")
            raw_band_axis = corr_band_axis = None
        if raw_band_axis is not None and corr_band_axis is not None:
            bands = min(
                raw_mm.shape[raw_band_axis],
                corrected_mm.shape[corr_band_axis],
            )
            if raw_wavs_arr is not None:
                bands = min(bands, raw_wavs_arr.shape[0])
            if corr_wavs_arr is not None:
                bands = min(bands, corr_wavs_arr.shape[0])

        if raw_band_axis is None or corr_band_axis is None or bands == 0:
            _skip_spectra_panel(
                "Skipping spectra panel: no overlapping bands between raw and corrected cubes."
            )
        else:
            if nir_band is not None:
                h, w = nir_band.shape
            else:
                spatial_axes_raw = tuple(sorted({0, 1, 2} - {raw_band_axis}))
                h = raw_mm.shape[spatial_axes_raw[0]]
                w = raw_mm.shape[spatial_axes_raw[1]]

            cy, cx = h // 2, w // 2
            half = 12
            min_valid_pixels = 200
            mask_band_idx = int(
                min(
                    max(0, bands - 1),
                    nir_idx if nir_band is not None else 0,
                )
            )

            def _bounds(center_y: int, center_x: int) -> tuple[int, int, int, int]:
                y0_i = max(0, center_y - half)
                y1_i = min(h, center_y + half)
                x0_i = max(0, center_x - half)
                x1_i = min(w, center_x + half)
                return y0_i, y1_i, x0_i, x1_i

            def _valid_counts_for_patch(y0_i: int, y1_i: int, x0_i: int, x1_i: int) -> tuple[int, int]:
                raw_band_idx = int(
                    min(raw_mm.shape[raw_band_axis] - 1, mask_band_idx)
                )
                corr_band_idx = int(
                    min(corrected_mm.shape[corr_band_axis] - 1, mask_band_idx)
                )
                raw_slice = np.take(raw_mm, raw_band_idx, axis=raw_band_axis)
                corr_slice = np.take(corrected_mm, corr_band_idx, axis=corr_band_axis)
                raw_patch_band = np.asarray(raw_slice[y0_i:y1_i, x0_i:x1_i], dtype=np.float32)
                corr_patch_band = np.asarray(
                    corr_slice[y0_i:y1_i, x0_i:x1_i], dtype=np.float32
                )
                raw_mask = np.isfinite(raw_patch_band) & (raw_patch_band != 0)
                corr_mask = np.isfinite(corr_patch_band) & (corr_patch_band != 0)
                return int(raw_mask.sum()), int(corr_mask.sum())

            grid_fracs = np.linspace(0.25, 0.75, num=3)
            candidate_centers: list[tuple[int, int]] = [(cy, cx)]
            for frac_y in grid_fracs:
                for frac_x in grid_fracs:
                    y_cand = int(np.clip(round(frac_y * (h - 1)), 0, h - 1))
                    x_cand = int(np.clip(round(frac_x * (w - 1)), 0, w - 1))
                    cand = (y_cand, x_cand)
                    if cand not in candidate_centers:
                        candidate_centers.append(cand)

            selected_bounds: tuple[int, int, int, int] | None = None
            selected_counts: tuple[int, int] = (0, 0)
            best_candidate: tuple[int, int, int, int] | None = None
            best_counts: tuple[int, int] = (0, 0)
            best_score = -1

            for center_y, center_x in candidate_centers:
                y0_i, y1_i, x0_i, x1_i = _bounds(center_y, center_x)
                raw_valid, corr_valid = _valid_counts_for_patch(y0_i, y1_i, x0_i, x1_i)
                score = min(raw_valid, corr_valid)
                if score > best_score:
                    best_score = score
                    best_candidate = (y0_i, y1_i, x0_i, x1_i)
                    best_counts = (raw_valid, corr_valid)
                if raw_valid >= min_valid_pixels and corr_valid >= min_valid_pixels:
                    selected_bounds = (y0_i, y1_i, x0_i, x1_i)
                    selected_counts = (raw_valid, corr_valid)
                    break

            if selected_bounds is None:
                if best_candidate is not None:
                    selected_bounds = best_candidate
                    selected_counts = best_counts
                else:
                    selected_bounds = _bounds(cy, cx)
                    selected_counts = (0, 0)

            y0, y1, x0, x1 = selected_bounds
            raw_valid_count, corr_valid_count = selected_counts
            insufficient_pixels = (
                raw_valid_count < min_valid_pixels or corr_valid_count < min_valid_pixels
            )

            if insufficient_pixels:
                logger.warning(
                    "Spectra patch had limited valid pixels (raw=%d, corrected=%d; threshold=%d)",
                    raw_valid_count,
                    corr_valid_count,
                    min_valid_pixels,
                )

            logger.info(
                "Spectra patch valid pixels (>= %d required): raw=%d corrected=%d",
                min_valid_pixels,
                raw_valid_count,
                corr_valid_count,
            )

            patch = (slice(y0, y1), slice(x0, x1))

            try:
                raw_spec = _patch_mean_spectrum(raw_mm, raw_header, patch=patch)
                corr_spec = _patch_mean_spectrum(corrected_mm, corrected_header, patch=patch)

                raw_spec = np.asarray(raw_spec[:bands], dtype=np.float32)
                corr_spec = np.asarray(corr_spec[:bands], dtype=np.float32)

                raw_s = _to_unitless_reflectance(raw_spec)
                corr_s = _to_unitless_reflectance(corr_spec)

                diff_arr = corr_s - raw_s
                ratio_arr = np.divide(
                    corr_s,
                    np.maximum(raw_s, 1e-6),
                    out=np.full_like(corr_s, np.nan),
                    where=~np.isnan(raw_s) & ~np.isnan(corr_s),
                )

                raw_med = float(np.nanmedian(raw_s)) if raw_s.size else float("nan")
                corr_med = float(np.nanmedian(corr_s)) if corr_s.size else float("nan")
                ratio_med = float(np.nanmedian(ratio_arr)) if ratio_arr.size else float("nan")
                corr_std = float(np.nanstd(corr_s)) if corr_s.size else float("nan")

                logger.info(
                    "Spectra medians (unitless): raw=%.3f corrected=%.3f (std corrected=%.4f)",
                    raw_med,
                    corr_med,
                    corr_std,
                )
                if np.isfinite(ratio_med):
                    logger.info("Spectra corrected/raw median ratio: %.3f", ratio_med)

                use_wavelengths = (
                    raw_wavs_arr is not None
                    and corr_wavs_arr is not None
                    and raw_wavs_arr.shape == corr_wavs_arr.shape
                    and raw_wavs_arr.shape[0] >= bands
                )

                if use_wavelengths:
                    x_vals = raw_wavs_arr[:bands]
                    ax_spectra.set_xlabel("Wavelength (nm)")
                else:
                    x_vals = np.arange(bands)
                    ax_spectra.set_xlabel("Band index")

                ax_spectra.set_title(
                    "Patch-mean spectrum before vs after BRDF+topo correction"
                )
                ax_spectra.set_ylabel("Reflectance (unitless, 0–1)")
                line_raw, = ax_spectra.plot(
                    x_vals,
                    raw_s,
                    label="raw export (uncorrected)",
                )
                line_corr, = ax_spectra.plot(
                    x_vals,
                    corr_s,
                    label="corrected (BRDF+topo)",
                )

                if shaded_regions and use_wavelengths:
                    for start, end in ((400, 700), (700, 1300), (1300, 2500)):
                        ax_spectra.axvspan(start, end, alpha=0.08, color="tab:blue", zorder=0)

                ax_ratio = ax_spectra.twinx()
                line_ratio, = ax_ratio.plot(
                    x_vals,
                    ratio_arr,
                    color="lightgray",
                    linewidth=1.2,
                    label="corrected/raw ratio",
                )
                ax_ratio.set_ylabel("Corrected/Raw ratio")
                ax_ratio.set_ylim(0.5, 1.5)
                ax_ratio.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
                ax_ratio.spines["right"].set_color("gray")
                ax_ratio.tick_params(axis="y", colors="dimgray")
                ax_ratio.yaxis.label.set_color("dimgray")

                ax_diff = ax_spectra.twinx()
                ax_diff.spines["right"].set_position(("axes", 1.12))
                ax_diff.spines["right"].set_visible(True)
                ax_diff.patch.set_visible(False)
                line_diff, = ax_diff.plot(
                    x_vals,
                    diff_arr,
                    color="tab:gray",
                    linestyle="--",
                    label="difference (corrected − raw)",
                )
                ax_diff.set_ylabel("Difference (unitless)")
                ax_diff.tick_params(axis="y", colors="tab:gray")
                ax_diff.yaxis.label.set_color("tab:gray")

                handles = [line_raw, line_corr, line_diff, line_ratio]
                ax_spectra.legend(
                    handles, [h.get_label() for h in handles], loc="upper right"
                )

                note = (
                    "Correction reduces angular/illumination bias;\n"
                    "smoother, lower curve expected."
                )
                ax_spectra.text(
                    0.02, 0.95, note, transform=ax_spectra.transAxes, fontsize=9, va="top"
                )

                if (
                    (np.isfinite(raw_med) and raw_med < 1e-6)
                    or (np.isfinite(corr_med) and corr_med < 1e-6)
                    or (
                        np.isfinite(raw_med)
                        and np.isfinite(corr_med)
                        and raw_med < 0.005
                        and corr_med < 0.005
                    )
                ):
                    ax_spectra.text(
                        0.5,
                        0.98,
                        "WARNING: spectra nearly zero — check units / double-apply / offset",
                        transform=ax_spectra.transAxes,
                        ha="center",
                        color="crimson",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7),
                    )

                if np.isfinite(corr_std) and corr_std < 1e-3:
                    ax_spectra.text(
                        0.5,
                        0.92,
                        "WARNING: corrected spectrum nearly flat — check axes/units/double-apply",
                        transform=ax_spectra.transAxes,
                        ha="center",
                        fontsize=9,
                        color="crimson",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                    )

                if np.isfinite(ratio_med) and (ratio_med < 0.2 or ratio_med > 1.8):
                    ax_spectra.text(
                        0.5,
                        0.84,
                        f"Suspicious correction magnitude (median ratio={ratio_med:.2f})",
                        transform=ax_spectra.transAxes,
                        ha="center",
                        color="crimson",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7),
                    )

                if insufficient_pixels:
                    ax_spectra.text(
                        0.5,
                        0.1,
                        "Insufficient valid pixels for representative patch.",
                        transform=ax_spectra.transAxes,
                        ha="center",
                        color="darkorange",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7),
                    )

                mad = float(np.nanmean(np.abs(diff_arr)))
                ax_spectra.text(
                    0.01,
                    0.02,
                    f"Mean |Δ| = {mad:.3f}",
                    transform=ax_spectra.transAxes,
                    fontsize=9,
                    va="bottom",
                )
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"Failed to render spectra panel: {exc}")
                ax_spectra.axis("off")
                ax_spectra.set_title("Patch spectra (failed)")
                ax_spectra.text(
                    0.5,
                    0.5,
                    "Rendering error — see logs",
                    ha="center",
                    va="center",
                )
            
    else:
        ax_spectra.axis("off")
        ax_spectra.set_title("Patch spectra (skipped: data unavailable)")
        ax_spectra.text(
            0.5,
            0.5,
            "Raw or corrected data unavailable",
            ha="center",
            va="center",
        )

    # Panel D: Thumbnails of sensor products
    sensor_imgs = _gather_sensor_products(work_dir, flight_stem)
    if sensor_imgs:
        cols = min(3, max(1, int(math.ceil(math.sqrt(len(sensor_imgs))))))
        rows = int(math.ceil(len(sensor_imgs) / cols))
        thumb_spec = gs[1:, 1:].subgridspec(rows, cols, hspace=0.3, wspace=0.3)
        thumb_axes = [fig.add_subplot(thumb_spec[idx]) for idx in range(rows * cols)]
        for ax in thumb_axes:
            ax.axis("off")
        for img_path, ax in zip(sensor_imgs, thumb_axes):
            hdr_path = img_path.with_suffix(".hdr")
            if not hdr_path.exists():
                ax.text(0.5, 0.5, "Missing HDR", ha="center", va="center")
                ax.set_title(_sensor_label(img_path, flight_stem), fontsize=9)
                continue
            try:
                header = _load_envi_header(hdr_path)
                mm = memmap_bsq(img_path, header)
                idxs = _safe_band_indices(header)
                rgb = _make_rgb_composite(mm, idxs)
                rgb_small = _downsample_rgb(rgb)
                ax.imshow(rgb_small)
                ax.set_title(_sensor_label(img_path, flight_stem), fontsize=9)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"Thumbnail failed for {img_path.name}: {exc}")
                ax.text(0.5, 0.5, "Unavailable", ha="center", va="center")
                ax.set_title(_sensor_label(img_path, flight_stem), fontsize=9)
    else:
        thumb_spec = gs[1:, 1:].subgridspec(1, 1)
        ax_thumb = fig.add_subplot(thumb_spec[0])
        ax_thumb.axis("off")
        ax_thumb.text(0.5, 0.5, "No sensor products found", ha="center", va="center")

    # Panel E: Parquet summary
    ax_parquet.set_title("Parquet outputs present")
    ax_parquet.axis("off")
    parquet_files = list(work_dir.glob("*.parquet"))
    summaries = _collect_parquet_summaries(parquet_files)
    if summaries:
        text = "\n".join(summaries)
    else:
        text = "No Parquet outputs found"
    ax_parquet.text(0, 1, text, va="top")

    fig.suptitle(
        (
            f"QA Summary: {flight_stem}\n"
            "(Reflectance scaled to 0–1; BRDF+topo reduces angular bias)"
        ),
        fontsize=14,
    )

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        if out_png.exists():
            logger.info("🖼️  Overwriting QA panel -> %s", out_png)
        else:
            logger.info("🖼️  Writing QA panel -> %s", out_png)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")

    return fig


def summarize_all_flightlines(
    base_folder: Path,
    out_dir: Path | None = None,
    *,
    shaded_regions: bool = True,
    overwrite: bool = True,
):
    base_folder = Path(base_folder)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for child in sorted(base_folder.iterdir()):
        if not child.is_dir():
            continue
        flight_stem = child.name
        raw_img = child / f"{flight_stem}_envi.img"
        if not raw_img.exists():
            continue
        if out_dir is not None:
            out_png = out_dir / f"{flight_stem}_qa.png"
        else:
            out_png = child / f"{flight_stem}_qa.png"
        try:
            summarize_flightline_outputs(
                base_folder,
                flight_stem,
                out_png=out_png,
                shaded_regions=shaded_regions,
                overwrite=overwrite,
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"Failed to summarize {flight_stem}: {exc}",
                RuntimeWarning,
            )


__all__ = ["summarize_flightline_outputs", "summarize_all_flightlines"]

