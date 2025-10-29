"""Generate quicklook QA plots for processed flightline outputs."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .pipelines.pipeline import _parse_envi_header


_ENVI_DTYPE_MAP: dict[int, np.dtype] = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
}


def _load_envi_header(hdr_path: Path) -> dict:
    header = _parse_envi_header(hdr_path)
    return header


def _dtype_from_header(header: dict, hdr_path: Path) -> np.dtype:
    try:
        code = int(header["data type"])
    except KeyError as exc:  # pragma: no cover - malformed header unexpected
        raise RuntimeError(f"ENVI header missing 'data type': {hdr_path}") from exc
    try:
        return np.dtype(_ENVI_DTYPE_MAP[code])
    except KeyError as exc:  # pragma: no cover - unsupported dtype unexpected
        raise RuntimeError(f"Unsupported ENVI data type {code} for {hdr_path}") from exc


def _memmap_bsq(img_path: Path, header: dict) -> np.memmap:
    try:
        samples = int(header["samples"])
        lines = int(header["lines"])
        bands = int(header["bands"])
    except KeyError as exc:  # pragma: no cover - malformed header unexpected
        raise RuntimeError(f"ENVI header missing dimension {exc.args[0]!r}: {img_path}") from exc

    interleave = str(header.get("interleave", "")).lower()
    if interleave != "bsq":
        raise RuntimeError(f"Only BSQ interleave supported for QA plots: {img_path}")

    dtype = _dtype_from_header(header, img_path.with_suffix(".hdr"))
    shape = (bands, lines, samples)
    return np.memmap(img_path, mode="r", dtype=dtype, shape=shape, order="C")


def _pick_rgb_bands_for_raw(header_dict: dict) -> tuple[int, int, int]:
    target_wavs = [650, 560, 470]
    wavs = np.asarray([float(w) for w in header_dict.get("wavelength", [])], dtype=float)
    if wavs.size == 0:
        return (0, 0, 0)
    idxs = [int(np.argmin(np.abs(wavs - tw))) for tw in target_wavs]
    return tuple(idxs)  # type: ignore[return-value]


def _pick_preview_band_for_corrected(header_dict: dict) -> int:
    target = 800
    wavs = np.asarray([float(w) for w in header_dict.get("wavelength", [])], dtype=float)
    if wavs.size == 0:
        return 0
    idx = int(np.argmin(np.abs(wavs - target)))
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
    ax_corrected = fig.add_subplot(gs[1, 0])
    ax_parquet = fig.add_subplot(gs[2, 0])

    thumb_spec = gs[1:, 1:].subgridspec(1, 1)
    thumb_axes: list = []

    def _fail_axis(ax, title: str, message: str) -> None:
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)

    # Panel A: Raw RGB
    if raw_img.exists() and raw_hdr.exists():
        try:
            raw_header = _load_envi_header(raw_hdr)
            raw_mm = _memmap_bsq(raw_img, raw_header)
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
            corrected_mm = _memmap_bsq(corrected_img, corrected_header)
            nir_idx = _pick_preview_band_for_corrected(corrected_header)
            nir_band = np.asarray(corrected_mm[nir_idx], dtype=np.float32)
            nir_preview = _percentile_stretch(nir_band)
            ax_corrected.imshow(nir_preview, cmap="gray")
            ax_corrected.set_title("Corrected cube (BRDF+topo) — NIR band preview")
            ax_corrected.axis("off")
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Failed to plot corrected preview for {flight_stem}: {exc}")
            _fail_axis(
                ax_corrected,
                "Corrected cube (BRDF+topo) — NIR band preview",
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
        corrected_header = None
        corrected_mm = None
        nir_band = None

    # Panel B: Spectral correction effect
    ax_spectra.set_title("Patch-mean spectrum before vs after BRDF+topo correction")
    ax_spectra.set_xlabel("Wavelength (nm)")
    ax_spectra.set_ylabel("Reflectance")

    if raw_header is not None and corrected_header is not None and raw_mm is not None and corrected_mm is not None:
        raw_wavs = np.asarray(raw_header.get("wavelength", []), dtype=float)
        corr_wavs = np.asarray(corrected_header.get("wavelength", []), dtype=float)
        if raw_wavs.size and corr_wavs.size and raw_wavs.shape == corr_wavs.shape:
            bands = min(raw_mm.shape[0], corrected_mm.shape[0])
            if nir_band is not None:
                h, w = nir_band.shape
            else:
                h, w = raw_mm.shape[1], raw_mm.shape[2]
            cy, cx = h // 2, w // 2
            half = 12
            y0, y1 = max(0, cy - half), min(h, cy + half)
            x0, x1 = max(0, cx - half), min(w, cx + half)

            raw_means = []
            corr_means = []
            for idx in range(bands):
                raw_patch = np.asarray(raw_mm[idx, y0:y1, x0:x1], dtype=np.float32)
                corr_patch = np.asarray(corrected_mm[idx, y0:y1, x0:x1], dtype=np.float32)
                raw_means.append(float(np.nanmean(raw_patch)))
                corr_means.append(float(np.nanmean(corr_patch)))

            raw_means_arr = np.asarray(raw_means)
            corr_means_arr = np.asarray(corr_means)
            diff_arr = corr_means_arr - raw_means_arr

            ax_spectra.plot(raw_wavs[:bands], raw_means_arr, label="raw export")
            ax_spectra.plot(raw_wavs[:bands], corr_means_arr, label="corrected (BRDF+topo)")
            ax_spectra.legend(loc="upper right")
            ax_diff = ax_spectra.twinx()
            ax_diff.plot(raw_wavs[:bands], diff_arr, color="tab:gray", linestyle="--", label="difference")
            ax_diff.set_ylabel("Corrected - raw")
            ax_diff.legend(loc="lower right")
        else:
            warnings.warn(
                f"Wavelength metadata missing or mismatched for {flight_stem}; skipping spectra panel"
            )
            ax_spectra.text(0.5, 0.5, "Wavelength metadata unavailable", ha="center", va="center")
    else:
        ax_spectra.text(0.5, 0.5, "Raw or corrected data unavailable", ha="center", va="center")

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
                mm = _memmap_bsq(img_path, header)
                bands = mm.shape[0]
                if bands >= 3:
                    idxs = (0, 1, 2)
                elif bands == 2:
                    idxs = (0, 1, 1)
                else:
                    idxs = (0, 0, 0)
                rgb = _make_rgb_composite(mm, idxs)
                rgb_small = _downsample_rgb(rgb)
                ax.imshow(rgb_small)
                ax.set_title(_sensor_label(img_path, flight_stem), fontsize=9)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"Failed to plot sensor thumbnail {img_path}: {exc}")
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

    fig.suptitle(f"QA Summary: {flight_stem}", fontsize=14)

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")

    return fig


def summarize_all_flightlines(
    base_folder: Path,
    out_dir: Path | None = None,
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
            summarize_flightline_outputs(base_folder, flight_stem, out_png=out_png)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"Failed to summarize {flight_stem}: {exc}",
                RuntimeWarning,
            )


__all__ = ["summarize_flightline_outputs", "summarize_all_flightlines"]

