#!/usr/bin/env python3
"""
cross_sensor_cal.pipelines.pipeline
-----------------------------------

Updated October 2025

Implements the core NEON hyperspectral → corrected reflectance → cross-sensor pipeline.

New features:
- Corrected scientific order (ENVI → correction JSON → correction → convolution)
- Per-stage validation and skip logic
- Clear, informative logging
- Safe reruns and partial recovery

Example:
    from pathlib import Path

    from cross_sensor_cal.pipelines.pipeline import go_forth_and_multiply
    go_forth_and_multiply(
        base_folder=Path("output_tester"),
        site_code="NIWO",
        year_month="2023-08",
        flight_lines=[
            "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance",
            "NEON_D13_NIWO_DP1_L020-1_20230815_directional_reflectance",
        ],
    )
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Sequence

import numpy as np

# --- Silence Ray’s stderr warnings BEFORE any potential imports of ray ---
# Send Ray logs to files (not stderr), reduce backend log level, disable usage pings.
os.environ.setdefault("RAY_LOG_TO_STDERR", "0")
os.environ.setdefault("RAY_BACKEND_LOG_LEVEL", "ERROR")
os.environ.setdefault("RAY_usage_stats_enabled", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")
os.environ.setdefault("RAY_disable_usage_stats", "1")
os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("RAY_LOG_TO_FILE", "1")

# Optional progress bars (fallback to no-bars if tqdm not present)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _pretty_line(line: str) -> str:
    """
    Return a short, readable label for a flight line (prefer the tile like L019-1).
    Falls back to the original string if no tile pattern found.
    """

    m = re.search(r"(L\d{3}-\d)", line)
    return m.group(1) if m else line


def _emit(msg: str, bars: dict[str, tqdm] | None, *, verbose: bool) -> None:
    """
    Print a human-readable message without mangling tqdm bars.
    """
    if verbose:
        print(msg)
        return
    if bars and tqdm is not None:
        try:  # pragma: no cover - tqdm.write may fail if tqdm missing features
            tqdm.write(msg)
            return
        except Exception:  # pragma: no cover - fall back to plain print
            pass
    print(msg)


def _warn_skip_exists(
    step: str,
    targets,
    verbose: bool,
    bars: dict[str, tqdm] | None = None,
    scope: str | None = None,
) -> None:
    """
    Emit a friendly skip line when expected outputs already exist.

    scope: optional short context like a tile id (e.g., L019-1)
    """

    try:
        count = len(list(targets))
    except Exception:  # pragma: no cover - targets may be generator-like
        count = "some"
    human_step = {
        "download": "download already present",
        "H5→ENVI (main+ancillary)": "ENVI + ancillary already exported",
        "generate_config_json": "config already present",
        "topo_and_brdf_correction": "BRDF+topo correction already present",
        "resample": "resampled outputs already present",
    }.get(step, step)
    scope_txt = f" [{scope}]" if scope else ""
    _emit(f"⏭️  Skipped{scope_txt}: {human_step} ({count})", bars, verbose=verbose)


def _stale_hint(step: str) -> str:
    """Guidance appended to exceptions to hint at corrupt/stale artifacts."""
    return (
        f"\n💡 Hint: This failure occurred in '{step}'. If this step was previously skipped "
        f"because matching output files already existed, a stale/corrupt file may be present. "
        f"Delete the corresponding output(s) and re-run to recreate them fresh."
    )


def _belongs_to(line: str, path_obj: Path) -> bool:
    name = path_obj.name
    return (line in name) or (line in str(path_obj.parent))


def _safe_total(total: int) -> int:
    return max(1, int(total))


def _line_outputs_present(base: Path, flight_line: str) -> bool:
    """Return True iff both main and ancillary ENVI outputs for flight line exist."""
    main_img = next(
        (p for p in base.rglob("*_reflectance_envi.img") if _belongs_to(flight_line, p)),
        None,
    )
    main_hdr = next(
        (p for p in base.rglob("*_reflectance_envi.hdr") if _belongs_to(flight_line, p)),
        None,
    )
    anc_img = next(
        (p for p in base.rglob("*_reflectance_ancillary_envi.img") if _belongs_to(flight_line, p)),
        None,
    )
    anc_hdr = next(
        (p for p in base.rglob("*_reflectance_ancillary_envi.hdr") if _belongs_to(flight_line, p)),
        None,
    )
    return all([main_img, main_hdr, anc_img, anc_hdr])


def _tick_download_slot(base: Path, flight_line: str, tick_cb) -> None:
    """Tick the download slot for a flight line once at least one matching .h5 exists."""
    found = next((p for p in base.rglob("*.h5") if _belongs_to(flight_line, p)), None)
    if found is not None:
        tick_cb(found)


# ============================================================
# Output noise filtering for normal mode (verbose=False)
#   - suppresses Ray service warnings and HyTools chunk spam
#   - preserves important success lines (e.g., "Saved:")
#   - uses redirect_stdout/redirect_stderr so third-party prints are tamed
# ============================================================

_NOISE_PATTERNS = [
    r"^20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\s+WARNING services\.py:\d+\s+-- WARNING: The object store is using /tmp",
    r"^20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\s+INFO worker\.py:\d+\s+-- Started a local Ray instance\.",
    r"^\[20\d{2}-\d{2}-\d{2} .* logging\.cc:\d+: Set ray log level",
    r"^\(raylet\) \[20\d{2}-\d{2}-\d{2} .* logging\.cc:\d+: Set ray log level",
    r"^\(HyTools pid=\d+\)\s*GR+$",
    r"^\(HyTools pid=\d+\)\s*GR(GR)+$",
    r"^\(HyTools pid=\d+\)\s*$",
]
_NOISE_REGEX = [re.compile(p) for p in _NOISE_PATTERNS]


class _FilterStream:
    """A text stream that drops lines matching noise patterns; tee others to a sink."""

    def __init__(self, sink, keep_saved: bool = True) -> None:
        self._buffer = StringIO()
        self._sink = sink
        self._keep_saved = keep_saved

    def write(self, s: str) -> None:
        self._buffer.write(s)
        data = self._buffer.getvalue()
        while "\n" in data:
            line, rest = data.split("\n", 1)
            self._buffer = StringIO()
            self._buffer.write(rest)
            self._process_line(line + "\n")
            data = self._buffer.getvalue()

    def flush(self) -> None:
        data = self._buffer.getvalue()
        if data:
            self._process_line(data)
            self._buffer = StringIO()
        try:
            self._sink.flush()
        except Exception:  # pragma: no cover - sink may not support flush
            pass

    def _process_line(self, line: str) -> None:
        if not self._is_noise(line):
            try:
                self._sink.write(line)
            except Exception:  # pragma: no cover - sink may be read-only
                pass

    def _is_noise(self, line: str) -> bool:
        if self._keep_saved and ("Saved:" in line or "All processing complete" in line):
            return False
        return any(rx.search(line) for rx in _NOISE_REGEX)


@contextlib.contextmanager
def _silence_noise(enabled: bool):
    """Context manager to silence noisy third-party output when enabled=True."""

    if not enabled:
        yield
        return

    original_out, original_err = sys.stdout, sys.stderr
    filtered_out = _FilterStream(original_out)
    filtered_err = _FilterStream(original_err)
    with contextlib.redirect_stdout(filtered_out), contextlib.redirect_stderr(filtered_err):
        yield
    for stream in (filtered_out, filtered_err):
        try:
            stream.flush()
        except Exception:  # pragma: no cover - flush best effort
            pass

from cross_sensor_cal.brdf_topo import (
    apply_brdf_topo_correction,
    build_and_write_correction_json,
)
from cross_sensor_cal.resample import resample_chunk_to_sensor
from cross_sensor_cal.utils import get_package_data_path
from cross_sensor_cal.utils_checks import _nonempty_file, is_valid_envi_pair, is_valid_json

from ..file_types import (
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceENVIFile,
    NEONReflectanceFile,
    NEONReflectanceResampledENVIFile,
)
from ..neon_to_envi import neon_to_envi_no_hytools
from ..standard_resample import translate_to_other_sensors
from ..mask_raster import mask_raster_with_polygons
from ..polygon_extraction import control_function_for_extraction
from ..file_sort import generate_file_move_list



def _coerce_scalar(value: str):
    token = value.strip().strip('"').strip("'")
    if not token:
        return ""
    lowered = token.lower()
    if lowered == "nan":
        return float("nan")
    for caster in (int, float):
        try:
            return caster(token)
        except (TypeError, ValueError):
            continue
    return token


def _parse_envi_header(hdr_path: Path) -> dict:
    if not hdr_path.exists():
        raise FileNotFoundError(hdr_path)

    raw_entries: dict[str, str] = {}
    collecting_key: str | None = None
    collecting_value: list[str] = []

    with hdr_path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            stripped = raw_line.strip()
            if not stripped or stripped.upper() == "ENVI":
                continue

            if collecting_key is not None:
                collecting_value.append(stripped)
                if "}" in stripped:
                    value = " ".join(collecting_value)
                    raw_entries[collecting_key] = value
                    collecting_key = None
                    collecting_value = []
                continue

            if "=" not in stripped:
                continue

            key_part, value_part = stripped.split("=", 1)
            key = key_part.strip().lower()
            value = value_part.strip()

            if value.startswith("{") and "}" not in value:
                collecting_key = key
                collecting_value = [value]
                continue

            raw_entries[key] = value

    def _split_block(value: str) -> list[str]:
        inner = value.strip()
        if inner.startswith("{"):
            inner = inner[1:]
        if inner.endswith("}"):
            inner = inner[:-1]
        # Replace newlines with spaces before splitting.
        inner = inner.replace("\n", " ")
        # Avoid empty tokens from double commas.
        return [token.strip() for token in inner.split(",") if token.strip()]

    list_float_keys = {"wavelength", "fwhm"}
    list_string_keys = {"map info", "band names"}
    int_scalar_keys = {"samples", "lines", "bands", "data type", "byte order"}

    processed: dict[str, object] = {}
    for key, raw_value in raw_entries.items():
        if raw_value.startswith("{") and raw_value.endswith("}"):
            tokens = _split_block(raw_value)
            if key in list_float_keys:
                try:
                    processed[key] = [float(token) for token in tokens]
                except ValueError as exc:
                    raise RuntimeError(
                        f"Could not parse numeric list for '{key}' from ENVI header"
                    ) from exc
            elif key in list_string_keys:
                processed[key] = [token.strip('"').strip("'") for token in tokens]
            else:
                processed[key] = [
                    _coerce_scalar(token.strip('"').strip("'")) for token in tokens
                ]
            continue

        cleaned = raw_value.strip().strip('"').strip("'")
        if key in int_scalar_keys:
            try:
                processed[key] = int(cleaned)
            except ValueError as exc:  # pragma: no cover - malformed headers unexpected
                raise RuntimeError(f"Header value for '{key}' is not an integer") from exc
            continue

        processed[key] = _coerce_scalar(cleaned)

    return processed


def _format_envi_scalar(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _format_envi_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        joined = ", ".join(_format_envi_scalar(v) for v in value)
        return f"{{ {joined} }}"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        if any(ch.isspace() for ch in stripped):
            return f"{{ {stripped} }}"
        return stripped
    return _format_envi_scalar(value)


def _build_resample_header_text(header: dict) -> str:
    lines = ["ENVI"]
    for key, value in header.items():
        lines.append(f"{key} = {_format_envi_value(value)}")
    lines.append("")
    return "\n".join(lines)


def convolve_resample_product(
    corrected_hdr_path: Path,
    sensor_srf: dict[str, np.ndarray],
    out_stem_resampled: Path,
    tile_y: int = 100,
    tile_x: int = 100,
) -> None:
    src_header = _parse_envi_header(corrected_hdr_path)
    try:
        samples = int(src_header["samples"])  # type: ignore[index]
        lines = int(src_header["lines"])  # type: ignore[index]
        bands = int(src_header["bands"])  # type: ignore[index]
    except KeyError as exc:  # pragma: no cover - malformed headers unexpected
        raise RuntimeError("Header missing required dimension keys") from exc

    interleave = str(src_header.get("interleave", "")).lower()
    if interleave != "bsq":
        raise RuntimeError("convolve_resample_product only supports BSQ interleave")

    if "wavelength" not in src_header:
        raise RuntimeError("ENVI header is missing 'wavelength' metadata required for resampling")

    wavelengths_arr = np.asarray(src_header["wavelength"], dtype=np.float32)
    if wavelengths_arr.ndim != 1 or wavelengths_arr.size == 0:
        raise RuntimeError("ENVI header contains an empty wavelength list")
    if wavelengths_arr.size != bands:
        raise RuntimeError(
            "Wavelength metadata length does not match the number of hyperspectral bands"
        )

    sensor_band_names = list(sensor_srf.keys())
    out_bands = len(sensor_band_names)
    if out_bands == 0:
        raise RuntimeError("sensor_srf must contain at least one output band")

    srfs_prepped: dict[str, np.ndarray] = {}
    for band_name, weights in sensor_srf.items():
        arr = np.asarray(weights, dtype=np.float32)
        if arr.shape != (bands,):
            raise RuntimeError(
                f"SRF for band '{band_name}' must match hyperspectral band count ({bands})"
            )
        srfs_prepped[band_name] = arr

    img_path = corrected_hdr_path.with_suffix(".img")
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    mm = np.memmap(img_path, dtype="float32", mode="r", shape=(bands, lines, samples))

    out_img_path = out_stem_resampled.with_suffix(".img")
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    mm_out = np.memmap(out_img_path, dtype="float32", mode="w+", shape=(out_bands, lines, samples))

    nodata_value = src_header.get("data ignore value")
    nodata_float: float | None
    if isinstance(nodata_value, (list, tuple)) and nodata_value:
        try:
            nodata_float = float(nodata_value[0])
        except (TypeError, ValueError):
            nodata_float = None
    elif nodata_value is None:
        nodata_float = None
    else:
        try:
            nodata_float = float(nodata_value)
        except (TypeError, ValueError):
            nodata_float = None

    first_tile = True
    for ys in range(0, lines, tile_y):
        ye = min(lines, ys + tile_y)
        for xs in range(0, samples, tile_x):
            xe = min(samples, xs + tile_x)
            if first_tile:
                print("Processing resample tiles: ", end="", flush=True)
                first_tile = False
            print("GR", end="", flush=True)

            tile_bsq = mm[:, ys:ye, xs:xe]
            tile_yxb = np.transpose(tile_bsq, (1, 2, 0)).astype(np.float32, copy=False)

            sensor_tile = resample_chunk_to_sensor(
                tile_yxb,
                wavelengths_arr,
                srfs_prepped,
            )
            sensor_tile = sensor_tile.astype(np.float32, copy=False)

            if nodata_float is not None:
                if np.isnan(nodata_float):
                    invalid_mask = np.isnan(tile_yxb).all(axis=2)
                else:
                    invalid_mask = np.all(np.isclose(tile_yxb, nodata_float), axis=2)
                if invalid_mask.any():
                    sensor_tile[invalid_mask] = nodata_float

            for band_index, band_name in enumerate(sensor_band_names):
                mm_out[band_index, ys:ye, xs:xe] = sensor_tile[:, :, band_index]

    if not first_tile:
        print()

    mm_out.flush()
    del mm_out
    del mm

    out_header = {
        "samples": samples,
        "lines": lines,
        "bands": out_bands,
        "interleave": "bsq",
        "data type": 4,
        "byte order": 0,
        "map info": src_header.get("map info"),
        "projection": src_header.get("projection"),
        "wavelength units": src_header.get("wavelength units"),
        "wavelength": sensor_band_names,
        "description": (
            "Spectrally convolved product generated from BRDF+topo corrected hyperspectral cube"
        ),
    }

    out_header = {k: v for k, v in out_header.items() if v is not None}

    hdr_text = _build_resample_header_text(out_header)
    out_hdr_path = out_stem_resampled.with_suffix(".hdr")
    out_hdr_path.write_text(hdr_text, encoding="utf-8")


def _normalize_product_code(value: str | None) -> str:
    """Return the numeric component of a DP1 product code."""

    if not value:
        return "30006.001"

    trimmed = value.strip()
    if not trimmed:
        return "30006.001"

    upper = trimmed.upper()
    if upper.startswith("DP1."):
        trimmed = trimmed[4:]
    elif upper.startswith("DP1"):
        trimmed = trimmed[3:]
        if trimmed.startswith("."):
            trimmed = trimmed[1:]

    trimmed = trimmed.strip("._")
    return trimmed or "30006.001"


def _find_h5_for_flightline(base_folder: Path, flight_stem: str) -> Path:
    """Locate the NEON HDF5 file corresponding to *flight_stem*."""

    base_folder = Path(base_folder)
    direct_path = base_folder / f"{flight_stem}.h5"
    if direct_path.exists():
        return direct_path

    normalized = flight_stem.lower()
    candidates: list[Path] = []
    for path in sorted(base_folder.rglob("*.h5")):
        stem = path.stem.lower()
        if stem == normalized:
            return path
        if normalized in stem or normalized.replace(".", "_") in stem.replace(".", "_"):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find HDF5 source for flightline '{flight_stem}' in {base_folder}"
        )

    if len(candidates) > 1:
        logger.info(
            "⚠️  Multiple HDF5 matches for %s; using %s",
            flight_stem,
            candidates[0].name,
        )

    return candidates[0]


def _load_sensor_library() -> dict[str, dict[str, list[float]]]:
    """Load the spectral response functions used for convolution."""

    try:
        bands_path = get_package_data_path("landsat_band_parameters.json")
    except FileNotFoundError:
        logger.error(
            "❌ Sensor response library not found in package data. Inline resampling disabled."
        )
        return {}

    try:
        with bands_path.open("r", encoding="utf-8") as f:
            raw_library = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("❌ Could not parse %s (%s). Inline resampling disabled.", bands_path, exc)
        return {}

    if isinstance(raw_library, dict):
        return {k: v for k, v in raw_library.items() if isinstance(v, dict)}

    logger.error(
        "❌ Unexpected SRF library format in %s. Inline resampling disabled.", bands_path
    )
    return {}


def _build_sensor_srfs(
    wavelengths: np.ndarray,
    centers: Sequence[float],
    fwhm: Sequence[float],
    method: str,
) -> dict[str, np.ndarray]:
    srfs: dict[str, np.ndarray] = {}
    wl = np.asarray(wavelengths, dtype=np.float32)
    if wl.ndim != 1 or wl.size == 0:
        return srfs

    centers_arr = np.asarray(list(centers), dtype=np.float32)
    fwhm_arr = np.asarray(list(fwhm), dtype=np.float32) if fwhm else np.array([], dtype=np.float32)

    for idx, center in enumerate(centers_arr):
        band_key = f"band_{idx + 1:02d}"
        if method == "straight":
            srf = np.zeros_like(wl)
            srf[int(np.abs(wl - center).argmin())] = 1.0
        else:
            width = float(fwhm_arr[idx]) if idx < fwhm_arr.size else (
                float(fwhm_arr[-1]) if fwhm_arr.size else 0.0
            )
            if method == "straight" or width <= 0:
                srf = np.zeros_like(wl)
                srf[int(np.abs(wl - center).argmin())] = 1.0
            else:
                sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                srf = np.exp(-0.5 * ((wl - center) / sigma) ** 2)
        srfs[band_key] = np.asarray(srf, dtype=np.float32)

    return srfs


def _prepare_sensor_targets(
    *,
    corrected_file: NEONReflectanceBRDFCorrectedENVIFile,
    corrected_hdr_path: Path,
    resample_method: str,
    product_code: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Path]]:
    method_norm = (resample_method or "convolution").lower()
    inline_resample = method_norm in {"convolution", "gaussian", "straight"}
    if not inline_resample:
        logger.info(
            "⚠️  Resample method '%s' is not handled inline; skipping convolution stage.",
            resample_method,
        )
        return {}, {}

    header = _parse_envi_header(corrected_hdr_path)
    wavelengths = header.get("wavelength")
    if not isinstance(wavelengths, list) or not wavelengths:
        logger.error(
            "❌ ENVI header %s missing wavelength metadata required for resampling.",
            corrected_hdr_path,
        )
        return {}, {}

    sensor_library = _load_sensor_library()
    if not sensor_library:
        return {}, {}

    srfs_by_sensor: dict[str, dict[str, np.ndarray]] = {}
    stems_by_sensor: dict[str, Path] = {}

    suffix = corrected_file.suffix or "envi"
    raw_product = corrected_file.product or product_code
    if raw_product and not str(raw_product).upper().startswith("DP"):
        raw_product = product_code
    normalized_product = _normalize_product_code(raw_product)

    for sensor_name, params in sensor_library.items():
        centers = params.get("wavelengths", [])
        if not centers:
            continue
        fwhm = params.get("fwhms", [])
        srfs = _build_sensor_srfs(np.asarray(wavelengths, dtype=np.float32), centers, fwhm, method_norm)
        if not srfs:
            continue

        dir_prefix = f"{method_norm.capitalize()}_Reflectance_Resample"
        sensor_dir = corrected_file.directory / f"{dir_prefix}_{sensor_name.replace(' ', '_')}"
        sensor_dir.mkdir(parents=True, exist_ok=True)

        resampled_file = NEONReflectanceResampledENVIFile.from_components(
            domain=corrected_file.domain or "D00",
            site=corrected_file.site or "SITE",
            date=corrected_file.date or "00000000",
            sensor=sensor_name,
            suffix=suffix,
            folder=sensor_dir,
            time=corrected_file.time,
            tile=corrected_file.tile,
            directional=corrected_file.directional,
            product=normalized_product,
        )

        srfs_by_sensor[sensor_name] = srfs
        stems_by_sensor[sensor_name] = resampled_file.path.with_suffix("")

    return srfs_by_sensor, stems_by_sensor


def stage_export_envi_from_h5(
    *,
    base_folder: Path,
    product_code: str,
    flight_stem: str,
    brightness_offset: float | None = None,
) -> tuple[Path, Path]:
    """Ensure the raw ENVI export exists for *flight_stem*.

    Validates ``<stem>_reflectance_envi.img/.hdr`` and
    ``<stem>_reflectance_ancillary_envi.img/.hdr`` before deciding whether to
    re-run ``neon_to_envi_no_hytools``. Any missing or empty component triggers a
    fresh export so downstream stages always see intact ENVI pairs.
    """

    base_folder = Path(base_folder)
    h5_path = _find_h5_for_flightline(base_folder, flight_stem)
    refl_file = NEONReflectanceFile.from_filename(h5_path)

    product_value = _normalize_product_code(refl_file.product or product_code)
    product_token = f"DP1.{product_value}"

    if not refl_file.tile:
        raise RuntimeError(
            "Unable to determine NEON tile identifier from HDF5 filename; "
            "expected standard NEON naming convention."
        )

    raw_envi = NEONReflectanceENVIFile.from_components(
        domain=refl_file.domain or "D00",
        site=refl_file.site or "SITE",
        product=product_token,
        tile=refl_file.tile,
        date=refl_file.date or "00000000",
        time=refl_file.time,
        directional=getattr(refl_file, "directional", False),
        folder=base_folder,
    )

    raw_img_path = raw_envi.path
    raw_hdr_path = raw_img_path.with_suffix(".hdr")

    if is_valid_envi_pair(raw_img_path, raw_hdr_path):
        logger.info("✅ ENVI export already complete for %s, skipping", flight_stem)
        return raw_img_path, raw_hdr_path

    logger.info("📦 Exporting ENVI for %s", flight_stem)
    neon_to_envi_no_hytools(
        images=[str(h5_path)],
        output_dir=str(base_folder),
        brightness_offset=brightness_offset,
    )

    if not is_valid_envi_pair(raw_img_path, raw_hdr_path):
        raise RuntimeError(f"ENVI export failed for {flight_stem}")

    logger.info("✅ ENVI export completed for %s", flight_stem)
    return raw_img_path, raw_hdr_path


def stage_build_and_write_correction_json(
    *,
    base_folder: Path,
    product_code: str,
    flight_stem: str,
    raw_img_path: Path,
    raw_hdr_path: Path,
) -> Path:
    """Generate or reuse the correction JSON required for BRDF/topo processing.

    Produces ``<stem>_brdfandtopo_corrected_envi.json`` alongside the ENVI
    export. If a valid JSON already exists it is returned immediately; otherwise
    the helper recomputes BRDF coefficients, summarises ancillary rasters, and
    persists the document.
    """

    base_folder = Path(base_folder)
    h5_path = _find_h5_for_flightline(base_folder, flight_stem)
    correction_json_path = build_and_write_correction_json(
        h5_path=h5_path,
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        out_dir=base_folder,
    )

    if not is_valid_json(correction_json_path):
        raise RuntimeError(
            f"Failed to prepare correction JSON for {flight_stem}: {correction_json_path}"
        )

    return correction_json_path


def stage_apply_brdf_topo_correction(
    *,
    base_folder: Path,
    product_code: str,
    flight_stem: str,
    raw_img_path: Path,
    raw_hdr_path: Path,
    correction_json_path: Path,
) -> tuple[Path, Path]:
    """Apply BRDF + topographic correction using the precomputed JSON.

    Builds ``<stem>_brdfandtopo_corrected_envi.img/.hdr`` when absent or
    corrupted, reusing the JSON parameters captured in the previous stage. The
    outputs are validated with ``is_valid_envi_pair`` to protect later stages
    from incomplete files.
    """

    corrected_img_path, corrected_hdr_path = apply_brdf_topo_correction(
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        correction_json_path=correction_json_path,
        out_dir=base_folder,
    )

    if not is_valid_envi_pair(corrected_img_path, corrected_hdr_path):
        raise RuntimeError(
            f"Corrected ENVI invalid for {flight_stem}: {corrected_img_path}"
        )

    return corrected_img_path, corrected_hdr_path


def stage_convolve_all_sensors(
    *,
    base_folder: Path,
    product_code: str,
    flight_stem: str,
    corrected_img_path: Path,
    corrected_hdr_path: Path,
    resample_method: str = "convolution",
):
    """Convolve the corrected ENVI cube to the library of target sensors.

    Each sensor target validates and reuses ``<stem>_resampled_<sensor>.img/.hdr``
    when present, logging a skip message. Missing or invalid outputs trigger a
    fresh call to ``convolve_resample_product``.
    """

    if not is_valid_envi_pair(corrected_img_path, corrected_hdr_path):
        raise FileNotFoundError(
            f"Corrected ENVI missing or invalid for {flight_stem}: {corrected_img_path}"
        )

    try:
        corrected_file = NEONReflectanceBRDFCorrectedENVIFile.from_filename(corrected_img_path)
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse corrected ENVI filename for {flight_stem}: {corrected_img_path.name}"
        ) from exc

    srfs_by_sensor, stems_by_sensor = _prepare_sensor_targets(
        corrected_file=corrected_file,
        corrected_hdr_path=corrected_hdr_path,
        resample_method=resample_method,
        product_code=product_code,
    )

    if not srfs_by_sensor:
        return

    for sensor_name, srfs in srfs_by_sensor.items():
        out_stem = stems_by_sensor.get(sensor_name)
        if out_stem is None:
            continue

        out_img_path = out_stem.with_suffix(".img")
        out_hdr_path = out_stem.with_suffix(".hdr")

        if is_valid_envi_pair(out_img_path, out_hdr_path):
            logger.info(
                "✅ %s convolution already complete for %s, skipping",
                sensor_name,
                flight_stem,
            )
            continue

        try:
            convolve_resample_product(
                corrected_hdr_path=corrected_hdr_path,
                sensor_srf=srfs,
                out_stem_resampled=out_stem,
            )
        except Exception:
            logger.error(
                "⚠️  Resample failed for %s (%s)",
                corrected_img_path.name,
                sensor_name,
                exc_info=True,
            )
            continue

        if not is_valid_envi_pair(out_img_path, out_hdr_path):
            raise RuntimeError(
                f"{sensor_name} resample produced invalid output for {flight_stem}: {out_img_path}"
            )

        logger.info(
            "✅ Resampled %s for %s → %s",
            sensor_name,
            flight_stem,
            out_img_path,
        )


def process_one_flightline(
    *,
    base_folder: Path,
    product_code: str,
    flight_stem: str,
    resample_method: str = "convolution",
    brightness_offset: float | None = None,
):
    """Run the structured, skip-aware workflow for a single flightline.

    The function enforces the stage order:

    1. Validate/export ENVI from the source ``.h5``
    2. Materialise ``<stem>_brdfandtopo_corrected_envi.json``
    3. Create ``<stem>_brdfandtopo_corrected_envi.img/.hdr``
    4. Convolve the corrected cube for every configured sensor

    Each stage calls ``is_valid_envi_pair`` / ``is_valid_json`` to determine
    whether existing outputs can be reused. Invalid, truncated, or missing files
    trigger recomputation so partial runs recover cleanly.
    """

    logger.info("🚀 Processing %s ...", flight_stem)

    raw_img_path, raw_hdr_path = stage_export_envi_from_h5(
        base_folder=base_folder,
        product_code=product_code,
        flight_stem=flight_stem,
        brightness_offset=brightness_offset,
    )

    correction_json_path = stage_build_and_write_correction_json(
        base_folder=base_folder,
        product_code=product_code,
        flight_stem=flight_stem,
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
    )

    if not is_valid_json(correction_json_path):
        raise RuntimeError(
            f"Correction JSON invalid for {flight_stem}: {correction_json_path}"
        )

    corrected_img_path, corrected_hdr_path = stage_apply_brdf_topo_correction(
        base_folder=base_folder,
        product_code=product_code,
        flight_stem=flight_stem,
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        correction_json_path=correction_json_path,
    )

    if not is_valid_envi_pair(corrected_img_path, corrected_hdr_path):
        raise RuntimeError(
            f"Corrected ENVI invalid for {flight_stem}: {corrected_img_path}"
        )

    logger.info("🎯 Convolving corrected reflectance for %s", flight_stem)
    stage_convolve_all_sensors(
        base_folder=base_folder,
        product_code=product_code,
        flight_stem=flight_stem,
        corrected_img_path=corrected_img_path,
        corrected_hdr_path=corrected_hdr_path,
        resample_method=resample_method,
    )

    logger.info("🎉 Finished pipeline for %s", flight_stem)


def sort_and_sync_files(base_folder: str, remote_prefix: str = "", sync_files: bool = True):
    """
    Generate file sorting list and optionally sync files to iRODS using gocmd.
    
    Parameters:
    - base_folder: Base directory containing processed files
    - remote_prefix: Optional custom path to add after i:/iplant/ for remote paths
    - sync_files: Whether to actually sync files (True) or just generate the list (False)
    """
    print("\n=== Starting file sorting and syncing ===")
    
    # Generate the file move list
    print(f"Generating file move list for: {base_folder}")
    df_move_list = generate_file_move_list(base_folder, base_folder, remote_prefix)
    
    # Save the move list to base_folder (not in sorted_files subdirectory)
    csv_path = os.path.join(base_folder, "envi_file_move_list.csv")
    df_move_list.to_csv(csv_path, index=False)
    print(f"✅ File move list saved to: {csv_path}")
    
    if not sync_files:
        print("Sync disabled. File list generated but no files transferred.")
        return
    
    if len(df_move_list) == 0:
        print("No files to sync.")
        return
    
    # Sync files using gocmd
    print(f"\nStarting file sync to iRODS ({len(df_move_list)} files)...")
    
    # Process each unique source-destination directory pair
    # Group by source directory to minimize gocmd calls
    source_dirs = df_move_list.groupby(df_move_list['Source Path'].apply(lambda x: os.path.dirname(x)))
    
    total_synced = 0
    for source_dir, group in source_dirs:
        # Get unique destination directory for this group
        dest_dirs = group['Destination Path'].apply(lambda x: os.path.dirname(x)).unique()
        
        for dest_dir in dest_dirs:
            # Filter files for this specific source-dest pair
            files_to_sync = group[group['Destination Path'].apply(lambda x: os.path.dirname(x)) == dest_dir]
            
            print(f"\nSyncing {len(files_to_sync)} files from {source_dir} to {dest_dir}")
            
            try:
                # Run gocmd sync command
                cmd = ["gocmd", "sync", source_dir, dest_dir, "--progress"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ Successfully synced {len(files_to_sync)} files")
                    total_synced += len(files_to_sync)
                else:
                    print(f"❌ Error syncing files: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ Exception during sync: {str(e)}")
    
    print(f"\n✅ File sync complete. Total files synced: {total_synced}/{len(df_move_list)}")


def go_forth_and_multiply(
    base_folder: Path,
    site_code: str,
    year_month: str,
    flight_lines: list[str],
    *,
    product_code: str = "DP1.30006.001",
    resample_method: str = "convolution",
    brightness_offset: float | None = None,
) -> None:
    """Execute the full idempotent pipeline for each requested flightline.

    The driver enforces the canonical stage order (ENVI export → correction JSON →
    BRDF+topographic correction → convolution) and validates every output before
    moving forward. Existing, healthy artefacts are reused so rerunning the same
    invocation is safe and fast.

    Parameters
    ----------
    base_folder:
        Root directory where intermediate and final ENVI artefacts are written.
    site_code:
        NEON site identifier used for download helpers and logging context.
    year_month:
        Year-month string (``YYYY-MM``) that scopes NEON downloads.
    flight_lines:
        Iterable of NEON flightline stems to process.
    product_code:
        Optional NEON product override. Defaults to ``DP1.30006.001``.
    resample_method:
        Convolution strategy; ``"convolution"`` enforces corrected-cube SRF
        resampling. Alternate methods are preserved for backward compatibility.
    brightness_offset:
        Optional scalar offset added during the correction stage.
    """

    base_path = Path(base_folder)
    base_path.mkdir(parents=True, exist_ok=True)

    if not flight_lines:
        logger.info("No flightlines provided. Nothing to process.")
        return

    method_norm = (resample_method or "convolution").lower()

    for flight_stem in flight_lines:
        process_one_flightline(
            base_folder=base_path,
            product_code=product_code,
            flight_stem=flight_stem,
            resample_method=resample_method,
            brightness_offset=brightness_offset,
        )

    if method_norm in {"legacy", "resample"}:
        logger.info("🔁 Legacy resampling requested; translating corrected products.")
        resample_translation_to_other_sensors(base_path)

    logger.info("✅ All requested flightlines processed.")

def resample_translation_to_other_sensors(base_folder: Path):
    # List all subdirectories in the base folder
    brdf_corrected_header_files = NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(base_folder, 'envi')
    print("Starting translation to other sensors")
    for brdf_corrected_header_file in brdf_corrected_header_files:
        print(f"Resampling folder: {brdf_corrected_header_file}")
        translate_to_other_sensors(brdf_corrected_header_file)
    print("done resampling")


def process_base_folder(base_folder: Path, polygon_layer: str, **kwargs):
    """
    Processes subdirectories in a base folder, finding raster files and applying analysis.
    """
    # Get list of subdirectories
    raster_files = (NEONReflectanceENVIFile.find_in_directory(Path(base_folder)) +
                    NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(Path(base_folder), 'envi') +
                    NEONReflectanceResampledENVIFile.find_all_sensors_in_directory(Path(base_folder), 'envi'))

    if polygon_layer is None:
        return

    for raster_file in raster_files:
        try:
            print(f"Processing raster file: {raster_file}")

            # Mask raster with polygons
            masked_raster = mask_raster_with_polygons(
                envi_file=raster_file,
                geojson_path=polygon_layer,
                raster_crs_override=kwargs.get("raster_crs_override", None),
                polygons_crs_override=kwargs.get("polygons_crs_override", None),
                plot_output=kwargs.get("plot_output", False),
                plot_filename=kwargs.get("plot_filename", None),
                dpi=kwargs.get("dpi", 300),
            )

            if masked_raster:
                print(f"Successfully processed and saved masked raster: {masked_raster}")
            else:
                print(f"Skipping raster: {raster_file}")
        except Exception as e:
            print(f"Error processing raster file {raster_file}: {e}")
            continue

    print("All subdirectories processed.")


def process_all_subdirectories(parent_directory: Path, polygon_path):
    """Searches and processes all subdirectories."""
    if polygon_path is None:
        return

    try:
        control_function_for_extraction(parent_directory, polygon_path)
    except Exception as e:
        print(f"[ERROR] Error processing directory '{parent_directory.name}': {e}")


def jefe(
    base_folder,
    site_code,
    year_month,
    flight_lines,
    polygon_layer_path: str,
    remote_prefix: str = "",
    sync_files: bool = True,
    resample_method: str = "convolution",
    max_workers: int = 1,
    skip_download_if_present: bool = True,
    force_config: bool = False,
    brightness_offset: float = 0.0,
    verbose: bool = False,
):
    """
    A control function that orchestrates the processing of spectral data.
    It first calls go_forth_and_multiply to generate necessary data and structures,
    then processes all subdirectories within the base_folder, and finally sorts
    and syncs files to iRODS.

    Parameters:
    - base_folder (str): The base directory for both operations.
    - site_code (str): Site code for go_forth_and_multiply.
    - year_month (str): Year and month for go_forth_and_multiply.
    - flight_lines (list): A list of flight lines for go_forth_and_multiply.
    - polygon_layer_path (str): Path to polygon shapefile or GeoJSON.
    - remote_prefix (str): Optional custom path to add after i:/iplant/ for remote paths.
    - sync_files (bool): Whether to sync files to iRODS or just generate the list.
    """
    product_code = 'DP1.30006.001'

    # First, call go_forth_and_multiply with the provided parameters
    go_forth_and_multiply(
        base_folder=base_folder,
        site_code=site_code,
        product_code=product_code,
        year_month=year_month,
        flight_lines=flight_lines,
        resample_method=resample_method,
        brightness_offset=brightness_offset,
    )

    process_base_folder(
        base_folder=base_folder,
        polygon_layer=polygon_layer_path,
        raster_crs_override="EPSG:4326",  # Optional CRS override
        polygons_crs_override="EPSG:4326",  # Optional CRS override
        output_masked_suffix="_masked",  # Optional suffix for output
        plot_output=False,  # Disable plotting
        dpi=300  # Set plot resolution
    )

    # Next, process all subdirectories within the base_folder
    process_all_subdirectories(Path(base_folder), polygon_layer_path)

    # File sorting and syncing to iRODS
    sort_and_sync_files(base_folder, remote_prefix, sync_files)

    # Finally, clean the CSV files by removing rows with any NaN values
    # clean_csv_files_in_subfolders(base_folder)

    # merge_csvs_by_columns(base_folder)
    # validate_output_files(base_folder)

    print(
        "Jefe finished. Please check for the _with_mask_and_all_spectra.csv for your  hyperspectral data from NEON flight lines extracted to match your provided polygons")

def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run the JEFE pipeline for processing NEON hyperspectral data with polygon extraction."
    )

    parser.add_argument("base_folder", type=Path, help="Base folder containing NEON data")
    parser.add_argument("site_code", type=str, help="NEON site code (e.g., NIWO)")
    parser.add_argument("year_month", type=str, help="Year and month (e.g., 202008)")
    parser.add_argument("flight_lines", type=str,
                        help="Comma-separated list of flight line names (e.g., FL1,FL2)")
    parser.add_argument("--polygon_layer_path", type=Path,
                        help="Path to polygon shapefile or GeoJSON. Will extract polygons and mask output files"
                             " if specified", required=False)
    parser.add_argument("--brightness-offset", type=float, default=0.0,
                        help="Additive brightness offset applied after corrections/resampling (e.g., -0.0005).")
    parser.add_argument("--reflectance-offset", type=float, default=0.0,
                        help="DEPRECATED: use --brightness-offset instead.")
    parser.add_argument("--remote-prefix", type=str, default="",
                        help="Optional custom path to add after i:/iplant/ for remote iRODS paths")
    parser.add_argument("--no-sync", action="store_true",
                        help="Generate file list but do not sync files to iRODS")
    parser.add_argument(
        "--resample-method",
        type=str,
        choices=("convolution", "gaussian", "straight", "legacy", "resample"),
        default="convolution",
        help="Resampling strategy for Step 5. "
             "'convolution' = Δλ-normalized SRF integration (recommended); "
             "'gaussian' = Gaussian SRFs; "
             "'straight' = nearest/linear band sampling; "
             "'legacy'/'resample' = old translate_to_other_sensors path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit detailed per-step logs instead of compact progress bars.",
    )

    args = parser.parse_args(argv)
    if args.reflectance_offset and float(args.reflectance_offset) != 0.0:
        args.brightness_offset = float(args.reflectance_offset)
        print("⚠️  --reflectance-offset is deprecated; using --brightness-offset instead.")
    return args


def run_pipeline(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    flight_lines_list = [fl.strip() for fl in args.flight_lines.split(",") if fl.strip()]

    polygon_layer_path = args.polygon_layer_path
    if polygon_layer_path is not None:
        polygon_layer_path = str(polygon_layer_path)

    jefe(
        base_folder=str(args.base_folder),
        site_code=args.site_code,
        year_month=args.year_month,
        flight_lines=flight_lines_list,
        polygon_layer_path=polygon_layer_path,
        remote_prefix=args.remote_prefix,
        sync_files=not args.no_sync,
        brightness_offset=args.brightness_offset,
        verbose=args.verbose,
        resample_method=args.resample_method,
    )


if __name__ == "__main__":
    run_pipeline()
