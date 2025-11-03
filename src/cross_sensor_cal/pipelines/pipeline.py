#!/usr/bin/env python3
"""
cross_sensor_cal.pipelines.pipeline
-----------------------------------

Updated October 2025

This module implements the core single-flightline and multi-flightline processing pipeline
for NEON hyperspectral flight lines. The pipeline is now:

    1. ENVI export from NEON .h5
    2. BRDF/topo correction parameter JSON build
    3. BRDF + topographic correction
    4. Sensor convolution/resampling
    5. Parquet export for every ENVI reflectance product
    6. DuckDB merge of original/corrected/resampled Parquet tables

Key guarantees:
- The stages ALWAYS run in the order above.
- Convolution/resampling ALWAYS uses the BRDF+topo corrected ENVI product
  (<flight_stem>_brdfandtopo_corrected_envi.img/.hdr), never the raw .h5.
- Each stage is idempotent:
    * If valid outputs already exist, that stage logs "✅ ... skipping" and returns.
    * If outputs are missing or corrupted, that stage recomputes them.
- The pipeline is restart-safe. You can rerun go_forth_and_multiply() after an interruption
  and it will resume where it left off without recomputing successfully completed work.
- All file naming and file locations are defined centrally by get_flightline_products().
  Stages do not hardcode filenames directly.

Typical usage:

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

Logs will include messages like:
    "🔎 ENVI export target for ... is <stem>_envi.img / <stem>_envi.hdr"
    "✅ ENVI export already complete ... (skipping heavy export)"
    "✅ BRDF+topo correction already complete ... (skipping)"
    "🎯 Convolving corrected reflectance for ..."
    "🎉 Finished pipeline for <flight_stem>"

These logs confirm both correct ordering and skip behavior.
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
from pathlib import Path
from typing import Any, Sequence

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import numpy as np

from cross_sensor_cal.brdf_topo import (
    apply_brdf_topo_core,
    build_correction_parameters_dict,
)
from cross_sensor_cal.brightness_config import load_brightness_coefficients
from cross_sensor_cal.paths import normalize_brdf_model_path
from cross_sensor_cal.qa_plots import render_flightline_panel
from cross_sensor_cal.resample import resample_chunk_to_sensor
from cross_sensor_cal.utils import get_package_data_path
from cross_sensor_cal.utils_checks import is_valid_json

from ..envi_download import download_neon_file
from ..file_sort import generate_file_move_list
from ..mask_raster import mask_raster_with_polygons
from ..merge_duckdb import merge_flightline
from ..neon_to_envi import neon_to_envi_no_hytools
from ..polygon_extraction import control_function_for_extraction
from ..progress_utils import TileProgressReporter
from ..standard_resample import translate_to_other_sensors
from ..utils.naming import get_flight_paths, get_flightline_products
from ..file_types import (
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceENVIFile,
    NEONReflectanceResampledENVIFile,
)

# ---------------------------------------------------------------------
# Logging setup (safe even if module imported multiple times)
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
# ---------------------------------------------------------------------

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


@contextmanager
def _scoped_log_prefix(prefix: str):
    """Temporarily wrap module logger methods to prefix messages."""

    logger = logging.getLogger(__name__)

    class _PrefixAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return f"[{prefix}] {msg}", kwargs

    adapter = _PrefixAdapter(logger, {})
    old_info = logger.info
    old_warning = logger.warning
    old_error = logger.error
    old_debug = logger.debug

    logger.info = adapter.info
    logger.warning = adapter.warning
    logger.error = adapter.error
    logger.debug = adapter.debug
    try:
        yield
    finally:
        logger.info = old_info
        logger.warning = old_warning
        logger.error = old_error
        logger.debug = old_debug


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
        self._pending = ""
        self._sink = sink
        self._keep_saved = keep_saved

    def write(self, s: str) -> None:
        if not s:
            return

        for ch in s:
            if ch == "\n":
                self._emit(self._pending + "\n")
                self._pending = ""
            elif ch == "\r":
                if self._pending:
                    self._emit(self._pending)
                    self._pending = ""
                self._emit("\r")
            else:
                self._pending += ch

        if self._pending:
            self._emit(self._pending)
            self._pending = ""

    def flush(self) -> None:
        if self._pending:
            self._emit(self._pending)
            self._pending = ""
        try:
            self._sink.flush()
        except Exception:  # pragma: no cover - sink may not support flush
            pass

    def _emit(self, text: str) -> None:
        if not text:
            return
        if text == "\r":
            try:
                self._sink.write(text)
            except Exception:  # pragma: no cover - sink may be read-only
                pass
            return
        if not self._is_noise(text):
            try:
                self._sink.write(text)
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

def is_valid_envi_pair(img_path: Path, hdr_path: Path) -> bool:
    """Return ``True`` when both ENVI files exist and are non-empty."""

    try:
        img = Path(img_path)
        hdr = Path(hdr_path)
        if not (img.exists() and img.is_file()):
            return False
        if not (hdr.exists() and hdr.is_file()):
            return False
        if img.stat().st_size <= 0:
            return False
        if hdr.stat().st_size <= 0:
            return False
        return True
    except Exception:
        return False


def _export_parquet_stage(
    *,
    base_folder: Path,
    product_code: str,
    flight_stem: str,
    logger,
) -> list[Path]:
    """
    Stage: Parquet export for ENVI reflectance products.

    Returns the list of Parquet sidecars that exist (either newly written or
    previously present and validated).
    """

    from cross_sensor_cal.parquet_export import ensure_parquet_for_envi

    paths = get_flightline_products(base_folder, product_code, flight_stem)
    work_dir = Path(paths.get("work_dir", Path(base_folder) / flight_stem))

    if not work_dir.exists():
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - best effort directory creation
            logger.warning(
                "⚠️ Cannot create Parquet work directory for %s: %s",
                flight_stem,
                work_dir,
            )

    if not work_dir.exists():
        logger.warning(
            "⚠️ Cannot locate work directory for Parquet export: %s",
            work_dir,
        )
        return []

    logger.info("📦 Parquet export for %s ...", flight_stem)

    parquet_outputs: list[Path] = []

    for img_path in sorted(work_dir.glob("*.img")):
        stem_lower = img_path.stem.lower()
        if any(keyword in stem_lower for keyword in ["mask", "angle", "qa", "quality"]):
            continue

        # ensure the raw ENVI (and any other reflectance cubes) receive Parquet outputs
        parquet_path = ensure_parquet_for_envi(img_path, logger)
        if parquet_path is not None:
            parquet_outputs.append(Path(parquet_path))

    if parquet_outputs:
        validator = Path(__file__).resolve().parents[3] / "bin" / "validate_parquets"
        if validator.exists():
            try:
                subprocess.run(
                    [sys.executable, str(validator), str(work_dir)],
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "⚠️ Parquet validation reported an issue for %s: %s",
                    flight_stem,
                    exc,
                )
        else:
            logger.debug("Validator script not found at %s", validator)

    return parquet_outputs


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


def _sensor_requires_landsat_adjustment(sensor_name: str | None) -> bool:
    if not sensor_name:
        return False
    lowered = sensor_name.lower()
    return "landsat" in lowered


def _apply_landsat_brightness_adjustment(
    cube: np.ndarray,
    system_pair: str = "landsat_to_micasense",
) -> dict[int, float]:
    """Adjust Landsat-convolved cube brightness relative to MicaSense."""

    coeffs_pct = load_brightness_coefficients(system_pair)
    if cube.ndim != 3:
        raise ValueError("Expected 3D cube for brightness adjustment")

    band_axis = int(np.argmin(cube.shape))
    if cube.shape[band_axis] > 16:
        band_axis = 0

    if band_axis != 0:
        cube = np.moveaxis(cube, band_axis, 0)

    bands = cube.shape[0]
    applied: dict[int, float] = {}

    for band_idx, coeff_pct in coeffs_pct.items():
        zero_based = band_idx - 1
        if 0 <= zero_based < bands:
            frac = coeff_pct / 100.0
            cube[zero_based] = cube[zero_based] * (1.0 + frac)
            applied[band_idx] = coeff_pct

    return applied


def convolve_resample_product(
    corrected_hdr_path: Path,
    sensor_srf: dict[str, np.ndarray],
    out_stem_resampled: Path,
    tile_y: int = 100,
    tile_x: int = 100,
    progress_label: str | None = None,
    *,
    interactive_mode: bool = True,
    log_every: int = 25,
    sensor_name: str | None = None,
    brightness_system_pair: str | None = None,
) -> dict[str, dict[int, float]]:
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

    brightness_map: dict[str, dict[int, float]] = {}

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

    tiles_y = (lines + tile_y - 1) // tile_y
    tiles_x = (samples + tile_x - 1) // tile_x
    total_tiles = tiles_y * tiles_x
    label = progress_label or "Resampling tiles"

    reporter = TileProgressReporter(
        stage_name=label,
        total_tiles=total_tiles,
        interactive_mode=interactive_mode,
        log_every=log_every,
    )

    try:
        for ys in range(0, lines, tile_y):
            ye = min(lines, ys + tile_y)
            for xs in range(0, samples, tile_x):
                xe = min(samples, xs + tile_x)

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

                reporter.update(1)
    finally:
        reporter.close()

    if _sensor_requires_landsat_adjustment(sensor_name):
        system_pair = brightness_system_pair or "landsat_to_micasense"
        applied_coeffs = _apply_landsat_brightness_adjustment(mm_out, system_pair=system_pair)
        if applied_coeffs:
            brightness_map[system_pair] = applied_coeffs

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

    if brightness_map:
        serialized = {
            system_pair: {str(b): float(v) for b, v in coeffs.items()}
            for system_pair, coeffs in brightness_map.items()
        }
        out_header["brightness_coefficients"] = json.dumps(serialized, sort_keys=True)

    out_header = {k: v for k, v in out_header.items() if v is not None}

    hdr_text = _build_resample_header_text(out_header)
    out_hdr_path = out_stem_resampled.with_suffix(".hdr")
    out_hdr_path.write_text(hdr_text, encoding="utf-8")

    return brightness_map


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


def _safe_resolve_sensor_entry(
    sensor_name: str,
    sensor_library: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Resolve ``sensor_name`` within ``sensor_library`` while tolerating unknown sensors.

    Returns the matching (key, entry) if found, otherwise ``(None, None)``.
    """

    if not sensor_library:
        return None, None

    if sensor_name in sensor_library:
        return sensor_name, sensor_library[sensor_name]

    lowered = sensor_name.lower()

    for key, entry in sensor_library.items():
        key_lower = key.lower()
        if lowered == key_lower:
            return key, entry
        if lowered == key_lower.replace(" ", "_"):
            return key, entry
        if lowered.replace("_", " ") == key_lower:
            return key, entry

    alias_map: dict[str, str] = {
        "landsat_tm": "Landsat 5 TM",
        "landsat_etm+": "Landsat 7 ETM+",
        "landsat_etm_plus": "Landsat 7 ETM+",
        "landsat_oli": "Landsat 8 OLI",
        "landsat_oli2": "Landsat 9 OLI-2",
        "landsat_oli-2": "Landsat 9 OLI-2",
        "landsat_oli_2": "Landsat 9 OLI-2",
        "micasense": "MicaSense",
        "micasense_tm": "MicaSense-to-match TM and ETM+",
        "micasense_oli": "MicaSense-to-match OLI and OLI-2",
        "micasense_to_match_tm_etm+": "MicaSense-to-match TM and ETM+",
        "micasense_to_match_tm_etm_plus": "MicaSense-to-match TM and ETM+",
        "micasense_to_match_oli_oli2": "MicaSense-to-match OLI and OLI-2",
        "micasense_to_match_oli_and_oli2": "MicaSense-to-match OLI and OLI-2",
    }

    alias_key = alias_map.get(lowered)
    if alias_key and alias_key in sensor_library:
        return alias_key, sensor_library[alias_key]

    return None, None


def resample_to_sensor_bands(
    *,
    corrected_img_path: Path,
    corrected_hdr_path: Path,
    sensor_name: str,
    method: str,
    sensor_entry: dict[str, Any] | None = None,
    resolved_name: str | None = None,
    sensor_library: dict[str, dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    method_norm = (method or "convolution").lower()
    if method_norm not in {"convolution", "gaussian", "straight"}:
        raise RuntimeError(f"Resample method '{method}' is not supported for inline convolution")

    header = _parse_envi_header(corrected_hdr_path)
    wavelengths = header.get("wavelength")
    if not isinstance(wavelengths, list) or not wavelengths:
        raise RuntimeError(
            f"ENVI header {corrected_hdr_path} missing wavelength metadata required for resampling."
        )

    entry = sensor_entry
    resolved = resolved_name or sensor_name

    library = sensor_library or {}
    if entry is None:
        library = sensor_library or _load_sensor_library()
        if not library:
            raise RuntimeError(
                "Sensor response library unavailable; cannot convolve to target sensors"
            )
        resolved_lookup, entry_lookup = _safe_resolve_sensor_entry(sensor_name, library)
        if resolved_lookup is None or entry_lookup is None:
            raise KeyError(sensor_name)
        entry = entry_lookup
        resolved = resolved_lookup

    if entry is None:
        raise RuntimeError(f"Sensor entry missing for {sensor_name}")

    resolved_name = resolved

    srfs = _build_sensor_srfs(
        np.asarray(wavelengths, dtype=np.float32),
        entry.get("wavelengths", []),
        entry.get("fwhms", []),
        method_norm,
    )
    if not srfs:
        raise RuntimeError(
            f"No spectral response functions computed for {resolved_name}; cannot resample."
        )

    return srfs


def write_resampled_product(
    *,
    arr: dict[str, np.ndarray],
    out_path: Path,
    sensor_name: str,
    flight_stem: str,
    corrected_hdr_path: Path,
    interactive_mode: bool = True,
    log_every: int = 25,
) -> dict[str, dict[int, float]]:
    out_path = Path(out_path)
    out_stem = out_path.with_suffix("")
    return convolve_resample_product(
        corrected_hdr_path=corrected_hdr_path,
        sensor_srf=arr,
        out_stem_resampled=out_stem,
        progress_label=f"🧪 {sensor_name} tiles",
        interactive_mode=interactive_mode,
        log_every=log_every,
        sensor_name=sensor_name,
    )


def write_resampled_envi_cube(
    bandstack_array: dict[str, np.ndarray],
    img_path: Path,
    hdr_path: Path,
    *,
    corrected_hdr_path: Path,
    progress_label: str | None = None,
    interactive_mode: bool = True,
    log_every: int = 25,
    sensor_name: str | None = None,
) -> dict[str, dict[int, float]]:
    """Persist a resampled sensor bandstack to ENVI ``.img``/``.hdr`` files."""

    out_img_path = Path(img_path)
    out_hdr_path = Path(hdr_path)
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_hdr_path.parent.mkdir(parents=True, exist_ok=True)

    out_stem = out_img_path.with_suffix("")
    metadata = convolve_resample_product(
        corrected_hdr_path=corrected_hdr_path,
        sensor_srf=bandstack_array,
        out_stem_resampled=out_stem,
        progress_label=progress_label,
        interactive_mode=interactive_mode,
        log_every=log_every,
        sensor_name=sensor_name,
    )

    expected_img = out_stem.with_suffix(".img")
    expected_hdr = out_stem.with_suffix(".hdr")

    if expected_img != out_img_path and expected_img.exists():
        try:
            if out_img_path.exists():
                out_img_path.unlink()
            expected_img.replace(out_img_path)
        except Exception:
            logger.exception("Failed to rename resampled image to %s", out_img_path)
            raise

    if expected_hdr != out_hdr_path and expected_hdr.exists():
        try:
            if out_hdr_path.exists():
                out_hdr_path.unlink()
            expected_hdr.replace(out_hdr_path)
        except Exception:
            logger.exception("Failed to rename resampled header to %s", out_hdr_path)
            raise

    return metadata


def stage_download_h5(
    base_folder: Path,
    site_code: str,
    year_month: str,
    product_code: str,
    flight_stem: str,
) -> Path:
    """Ensure ``<base_folder>/<flight_stem>.h5`` exists and is non-empty."""

    flight_paths = get_flight_paths(base_folder, flight_stem)
    base_path = Path(flight_paths["base"])
    h5_path = Path(flight_paths["h5_path"])

    base_path.mkdir(parents=True, exist_ok=True)

    try:
        if h5_path.exists() and h5_path.stat().st_size > 0:
            logger.info(
                f"[{flight_stem}] ⬇️ Download already complete — reusing existing files ({h5_path.name})"
            )
            return h5_path
    except OSError as exc:  # pragma: no cover - filesystem permission issues
        raise FileNotFoundError(
            f"Unable to access existing HDF5 file {h5_path}: {exc}"
        ) from exc

    logger.info(
        f"[{flight_stem}] 🌐 Downloading {flight_stem} ({site_code}, {year_month}) into {h5_path} ..."
    )

    try:
        downloaded_path, _ = download_neon_file(
            site_code=site_code,
            product_code=product_code,
            year_month=year_month,
            flight_line=flight_stem,
            out_dir=base_path,
            output_path=h5_path,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not locate NEON flight line "
            f"{flight_stem} ({product_code}, {site_code}, {year_month})."
        ) from exc
    except Exception as exc:  # pragma: no cover - network/filesystem errors
        raise FileNotFoundError(
            f"Failed to download NEON HDF5 for {flight_stem}: {exc}"
        ) from exc

    if downloaded_path != h5_path and downloaded_path.exists():
        try:
            if h5_path.exists():
                h5_path.unlink()
            downloaded_path.replace(h5_path)
        except Exception as exc:  # pragma: no cover - rename failure
            raise FileNotFoundError(
                f"Unable to move downloaded file into {h5_path}: {exc}"
            ) from exc

    if (not h5_path.exists()) or h5_path.stat().st_size <= 0:
        raise FileNotFoundError(
            "Download step did not produce a valid HDF5: "
            f"{h5_path}\nExpected NEON flight line {flight_stem} "
            f"({product_code}, {site_code}, {year_month})."
        )

    logger.info("✅ Download complete for %s → %s", flight_stem, h5_path.name)
    return h5_path


def stage_export_envi_from_h5(
    base_folder: Path,
    product_code: str,
    flight_stem: str,
    brightness_offset: float | None = None,
    *,
    parallel_mode: bool = False,
    recover_missing_raw: bool = True,
) -> tuple[Path, Path]:
    """
    Ensure we have the uncorrected ENVI export (.img/.hdr) for this flightline.

    Returns
    -------
    (raw_img_path, raw_hdr_path)

    Behavior
    --------
    1. Compute the canonical raw ENVI paths inside the per-flightline work directory.
    2. If those paths already exist and validate, SKIP heavy export immediately.
       This prevents re-loading huge (>20 GB) hyperspectral cubes on reruns.
    3. Otherwise, run neon_to_envi_no_hytools() ONE TIME to generate ENVI.
    4. After export, if the canonical paths STILL aren't valid, raise RuntimeError
       that includes a diff of which new files appeared. That diff is then used to
       correct get_flightline_products(), so on the next run we really will skip.
    """

    flight_paths = get_flight_paths(base_folder, flight_stem)
    base_folder = Path(flight_paths["base"])

    work_dir = Path(flight_paths["work_dir"])
    h5_path = Path(flight_paths["h5_path"])
    raw_img_path = work_dir / f"{flight_stem}_envi.img"
    raw_hdr_path = work_dir / f"{flight_stem}_envi.hdr"

    assert raw_img_path.suffix == ".img"
    assert raw_hdr_path.suffix == ".hdr"
    raw_name_lower = raw_img_path.name.lower()
    assert "landsat" not in raw_name_lower
    assert "micasense" not in raw_name_lower

    work_dir.mkdir(parents=True, exist_ok=True)

    # Announce what we *expect* the raw ENVI export to be called.
    logger.info(
        f"📍 Preparing ENVI export at: {raw_img_path.name} / {raw_hdr_path.name}"
    )

    corrected_img = work_dir / f"{flight_stem}_brdfandtopo_corrected_envi.img"
    if corrected_img.exists() and not raw_img_path.exists():
        if recover_missing_raw:
            logger.warning(
                "♻️  Raw ENVI missing for %s but corrected exists. Rebuilding raw from HDF5.",
                flight_stem,
            )
            if not h5_path.exists():
                raise FileNotFoundError(
                    f"Cannot recover raw ENVI for {flight_stem}: {h5_path.name} not found."
                )
            neon_to_envi_no_hytools(
                images=[str(h5_path)],
                output_dir=str(work_dir),
                brightness_offset=brightness_offset,
                interactive_mode=not parallel_mode,
            )
            if not raw_img_path.exists() or not raw_hdr_path.exists():
                raise FileNotFoundError(
                    f"Recovery failed: expected {raw_img_path.name} / {raw_hdr_path.name}"
                )
            logger.info(
                f"✅ Rebuilt raw ENVI files → {raw_img_path.name} / {raw_hdr_path.name}"
            )
        else:
            raise FileNotFoundError(
                f"Raw ENVI missing for {flight_stem} but corrected exists. "
                "Re-run with recover_missing_raw=True to rebuild."
            )

    # FAST SKIP: if canonical raw ENVI already looks valid, avoid
    # calling neon_to_envi_no_hytools() entirely. This is critical to
    # prevent rereading ~23 GB cubes and killing the kernel.
    def _looks_valid(img_path: Path, hdr_path: Path) -> bool:
        return (
            img_path.exists()
            and img_path.is_file()
            and img_path.stat().st_size > 0
            and hdr_path.exists()
            and hdr_path.is_file()
            and hdr_path.stat().st_size > 0
        )

    if _looks_valid(raw_img_path, raw_hdr_path):
        logger.info(
            f"✅ Existing ENVI export found — skipping heavy export ({raw_img_path.name} / {raw_hdr_path.name})"
        )
        return raw_img_path, raw_hdr_path

    # Not valid yet: we'll try to generate it.
    logger.info(
        f"📦 No existing ENVI export detected — creating a new one from source data {h5_path.name}"
    )

    # Snapshot directory state BEFORE export so we can diff.
    before_listing = {p.name: p for p in work_dir.glob("*")}

    # This is the heavy step that logs the NeonCube memory footprint and now
    # streams tile progress via tqdm (instead of the old "GRGRGR..." spam).
    neon_to_envi_no_hytools(
        images=[str(h5_path)],
        output_dir=str(work_dir),
        brightness_offset=brightness_offset,
        interactive_mode=not parallel_mode,
    )

    # Snapshot AFTER export and collect the names of new files.
    after_listing = {p.name: p for p in work_dir.glob("*")}
    created_names = sorted(
        name for name in after_listing.keys() if name not in before_listing
    )

    # Now that export has run, re-check the canonical expected outputs.
    if _looks_valid(raw_img_path, raw_hdr_path):
        logger.info(
            f"✅ ENVI export created successfully → {raw_img_path.name} / {raw_hdr_path.name}"
        )
        return raw_img_path, raw_hdr_path

    # If we still can't validate the canonical "raw_envi_img"/"raw_envi_hdr",
    # we assume get_flightline_products() is wrong for this stage.
    # Raise a RuntimeError that tells the dev EXACTLY which files actually appeared.
    created_pretty = "\n  ".join(created_names) if created_names else "(no new files detected)"

    raise RuntimeError(
        (
            "ENVI export for {stem} ran, but the canonical raw ENVI paths from "
            "get_flightline_products() did not validate.\n\n"
            "Canonical expectation:\n"
            "  {raw_img}\n"
            "  {raw_hdr}\n\n"
            "New files actually created during this export call:\n"
            "  {created}\n\n"
            "→ ACTION REQUIRED:\n"
            "Update get_flightline_products() so that keys 'raw_envi_img' and "
            "'raw_envi_hdr' point at the actual uncorrected ENVI output that "
            "neon_to_envi_no_hytools() writes (the pre-BRDF/topo cube). Once "
            "those keys match reality, reruns will hit the '✅ ENVI export "
            "already complete ... (skipping heavy export)' branch and will no "
            "longer try to reload huge cubes into memory on every run."
        ).format(
            stem=flight_stem,
            raw_img=raw_img_path.name,
            raw_hdr=raw_hdr_path.name,
            created=created_pretty,
        )
    )


def stage_build_and_write_correction_json(
    *,
    base_folder: Path,
    product_code: str,
    flight_stem: str,
    raw_img_path: Path,
    raw_hdr_path: Path,
    parallel_mode: bool = False,
) -> Path:
    """
    Compute + persist BRDF/topo correction parameters (illumination geometry, slope/aspect,
    BRDF coefficients, etc.) before applying correction.

    Writes the canonical correction JSON, returns its path.
    Skips if valid.
    """

    paths = get_flightline_products(base_folder, product_code, flight_stem)

    correction_json_path = Path(paths["correction_json"])
    h5_path = Path(paths["h5"])
    work_dir = Path(paths.get("work_dir", correction_json_path.parent))

    work_dir.mkdir(parents=True, exist_ok=True)
    correction_json_path.parent.mkdir(parents=True, exist_ok=True)

    if is_valid_json(correction_json_path):
        logger.info(
            "✅ Correction JSON already complete for %s -> %s (skipping)",
            flight_stem,
            correction_json_path.name,
        )
        return correction_json_path

    params = build_correction_parameters_dict(
        h5_path=h5_path,
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        base_folder=work_dir,
        flight_stem=flight_stem,
        product_code=product_code,
    )

    with open(correction_json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    if not is_valid_json(correction_json_path):
        raise RuntimeError(
            f"Failed to write correction JSON for {flight_stem}: {correction_json_path}"
        )

    logger.info(
        "✅ Wrote correction JSON for %s -> %s",
        flight_stem,
        correction_json_path.name,
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
    parallel_mode: bool = False,
) -> tuple[Path, Path]:
    """
    Apply BRDF + topographic correction using the precomputed JSON.
    Produces the canonical corrected ENVI:
        *_brdfandtopo_corrected_envi.img/.hdr

    Returns (corrected_img_path, corrected_hdr_path).
    Skips if already valid.
    """

    paths = get_flightline_products(base_folder, product_code, flight_stem)

    corrected_img_path = Path(paths["corrected_img"])
    corrected_hdr_path = Path(paths["corrected_hdr"])
    work_dir = Path(paths.get("work_dir", corrected_img_path.parent))

    work_dir.mkdir(parents=True, exist_ok=True)
    corrected_img_path.parent.mkdir(parents=True, exist_ok=True)

    if is_valid_envi_pair(corrected_img_path, corrected_hdr_path):
        logger.info(
            "✅ BRDF+topo correction already complete for %s -> %s / %s (skipping)",
            flight_stem,
            corrected_img_path.name,
            corrected_hdr_path.name,
        )
        return corrected_img_path, corrected_hdr_path

    if not is_valid_json(correction_json_path):
        raise RuntimeError(
            f"Missing or invalid correction JSON for {flight_stem}: {correction_json_path}"
        )

    with open(correction_json_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    apply_brdf_topo_core(
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        params=params,
        out_img_path=corrected_img_path,
        out_hdr_path=corrected_hdr_path,
        interactive_mode=not parallel_mode,
    )

    if not is_valid_envi_pair(corrected_img_path, corrected_hdr_path):
        raise RuntimeError(
            f"BRDF/topo correction failed for {flight_stem}. "
            f"Expected {corrected_img_path.name} / {corrected_hdr_path.name}."
        )

    if "_brdfandtopo_corrected_envi" not in corrected_img_path.name:
        raise RuntimeError(
            f"Corrected output missing required suffix for {flight_stem}: {corrected_img_path.name}"
        )

    logger.info(
        "✅ BRDF+topo correction completed for %s -> %s / %s",
        flight_stem,
        corrected_img_path.name,
        corrected_hdr_path.name,
    )
    return corrected_img_path, corrected_hdr_path


def stage_convolve_all_sensors(
    *,
    base_folder: Path,
    product_code: str,
    flight_stem: str,
    corrected_img_path: Path | None = None,
    corrected_hdr_path: Path | None = None,
    resample_method: str | None = "convolution",
    parallel_mode: bool = False,
):
    """Convolve the BRDF+topo corrected ENVI cube into sensor bandstacks."""

    paths = get_flightline_products(base_folder, product_code, flight_stem)

    corrected_img = Path(paths["corrected_img"])
    corrected_hdr = Path(paths["corrected_hdr"])
    default_work_dir = corrected_img.parent
    flightline_dir = Path(paths.get("work_dir", default_work_dir)).resolve()

    if corrected_img_path is not None:
        corrected_img = Path(corrected_img_path)
    if corrected_hdr_path is not None:
        corrected_hdr = Path(corrected_hdr_path)

    sensor_products_raw = paths.get("sensor_products", {})
    sensor_products: dict[str, dict[str, Path]] = {}
    if isinstance(sensor_products_raw, dict):
        for key, value in sensor_products_raw.items():
            if isinstance(value, dict):
                sensor_products[key] = value

    if not is_valid_envi_pair(corrected_img, corrected_hdr):
        raise FileNotFoundError(
            f"Corrected ENVI missing or invalid for {flight_stem}: {corrected_img}, {corrected_hdr}"
        )

    logger.info("🎯 Convolving corrected reflectance for %s", flight_stem)

    sensor_library = _load_sensor_library()
    if not sensor_library:
        logger.error(
            "❌ Sensor response library unavailable; skipping convolution stage for %s",
            flight_stem,
        )
        logger.info("🎉 Finished pipeline for %s", flight_stem)
        return

    method_norm = (resample_method or "convolution").lower()

    created: list[str] = []
    existing: list[str] = []
    failed: list[str] = []

    for sensor_name, out_pair in sensor_products.items():
        out_img_path = out_pair.get("img")
        out_hdr_path = out_pair.get("hdr")

        if out_img_path is None or out_hdr_path is None:
            logger.warning(
                "⚠️  Sensor %s output paths are undefined; skipping for %s",
                sensor_name,
                flight_stem,
            )
            failed.append(sensor_name)
            continue

        out_img = Path(out_img_path)
        out_hdr = Path(out_hdr_path)

        if is_valid_envi_pair(out_img, out_hdr):
            logger.info(
                "✅ %s product already complete for %s -> %s / %s (skipping)",
                sensor_name,
                flight_stem,
                out_img.name,
                out_hdr.name,
            )
            existing.append(sensor_name)
            continue

        resolved_name, sensor_entry = _safe_resolve_sensor_entry(sensor_name, sensor_library)
        if resolved_name is None or sensor_entry is None:
            logger.warning(
                "⚠️  Sensor %s is not defined in sensor_library; skipping for %s",
                sensor_name,
                flight_stem,
            )
            failed.append(sensor_name)
            continue

        try:
            bandstack_array = resample_to_sensor_bands(
                corrected_img_path=corrected_img,
                corrected_hdr_path=corrected_hdr,
                sensor_name=resolved_name,
                method=method_norm,
                sensor_entry=sensor_entry,
                resolved_name=resolved_name,
                sensor_library=sensor_library,
            )

            write_resampled_envi_cube(
                bandstack_array,
                out_img,
                out_hdr,
                corrected_hdr_path=corrected_hdr,
                progress_label=f"🧪 {sensor_name} tiles",
                interactive_mode=not parallel_mode,
                sensor_name=resolved_name,
            )

            if not is_valid_envi_pair(out_img, out_hdr):
                logger.error(
                    "⚠️  %s resample produced invalid ENVI pair for %s: %s / %s",
                    sensor_name,
                    flight_stem,
                    out_img,
                    out_hdr,
                )
                failed.append(sensor_name)
            else:
                logger.info(
                    "✅ Wrote %s product for %s -> %s / %s",
                    sensor_name,
                    flight_stem,
                    out_img.name,
                    out_hdr.name,
                )
                created.append(sensor_name)

        except Exception:  # noqa: BLE001 - want to log full traceback
            logger.exception(
                "⚠️  Resample threw for %s (%s) -> intended %s / %s",
                flight_stem,
                sensor_name,
                out_img,
                out_hdr,
            )
            failed.append(sensor_name)

    parquet_outputs = _export_parquet_stage(
        base_folder=base_folder,
        product_code=product_code,
        flight_stem=flight_stem,
        logger=logger,
    )
    logger.info("✅ Parquet stage complete for %s", flight_stem)

    if parquet_outputs:
        merge_out = merge_flightline(flightline_dir, out_name=None, emit_qa_panel=True)
        logger.info("✅ DuckDB master → %s", merge_out)
    else:
        logger.warning(
            "⚠️ Skipping DuckDB merge for %s because no Parquet outputs were produced",
            flight_stem,
        )

    logger.info(
        "📊 Sensor convolution summary for %s | created=%s, existing=%s, failed=%s",
        flight_stem,
        created,
        existing,
        failed,
    )

    if len(created) == 0 and len(existing) == 0:
        raise RuntimeError(
            f"All sensor resamples failed or were undefined for {flight_stem}. Failed sensors: {failed}"
        )

    logger.info("🎉 Finished pipeline for %s", flight_stem)

    return {
        "created": created,
        "existing": existing,
        "failed": failed,
    }


def process_one_flightline(
    *,
    base_folder: Path,
    product_code: str,
    flight_stem: str,
    resample_method: str | None = "convolution",
    brightness_offset: float | None = None,
    parallel_mode: bool = False,
):
    """Run the structured, skip-aware workflow for a single flightline.

    Enforced stage order:
      1) export ENVI from H5
      2) build correction JSON
      3) apply BRDF + topographic correction
      4) convolve / resample to target sensors
      5) export Parquet sidecars for all reflectance ENVI outputs
      6) merge Parquet sidecars into a DuckDB master table

    Each stage:
      - uses get_flightline_products() for canonical file naming
      - checks if its outputs already exist and are valid
      - skips if possible
      - re-runs only missing or invalid work

    Partially written / corrupt files will fail validation and
    trigger recomputation so partial runs recover safely.
    """

    logger.info(f"✅ Starting processing for {flight_stem}")

    flight_paths = get_flight_paths(base_folder, flight_stem)
    Path(flight_paths["work_dir"]).mkdir(parents=True, exist_ok=True)

    raw_img_path, raw_hdr_path = stage_export_envi_from_h5(
        base_folder=base_folder,
        product_code=product_code,
        flight_stem=flight_stem,
        brightness_offset=brightness_offset,
        parallel_mode=parallel_mode,
    )

    flightline_dir = Path(flight_paths["work_dir"]).resolve()
    normalized = normalize_brdf_model_path(flightline_dir)
    if normalized:
        logger.info("🔧 Normalized BRDF model name → %s", normalized.name)

    correction_json_path = stage_build_and_write_correction_json(
        base_folder=base_folder,
        product_code=product_code,
        flight_stem=flight_stem,
        raw_img_path=raw_img_path,
        raw_hdr_path=raw_hdr_path,
        parallel_mode=parallel_mode,
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
        parallel_mode=parallel_mode,
    )
    if not is_valid_envi_pair(corrected_img_path, corrected_hdr_path):
        raise RuntimeError(
            f"Corrected ENVI invalid for {flight_stem}: {corrected_img_path}"
        )

    stage_convolve_all_sensors(
        base_folder=base_folder,
        product_code=product_code,
        flight_stem=flight_stem,
        corrected_img_path=corrected_img_path,
        corrected_hdr_path=corrected_hdr_path,
        resample_method=resample_method,
        parallel_mode=parallel_mode,
    )

    try:
        render_flightline_panel(Path(base_folder) / flight_stem, quick=True, save_json=True)
    except Exception as e:  # pragma: no cover - metrics best effort
        logger.warning("⚠️  QA rendering failed for %s: %s", flight_stem, e)


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
    resample_method: str | None = "convolution",
    brightness_offset: float | None = None,
    max_workers: int = 2,
) -> None:
    """High-level orchestrator for processing multiple flight lines.

    Steps:

      1. Ensure the NEON ``.h5`` for each ``flight_stem`` is present locally via
         :func:`stage_download_h5` (performed serially).
      2. Run the per-flightline pipeline (ENVI export → corrections → resample →
         Parquet) by delegating to :func:`process_one_flightline` in parallel.

    The function maintains canonical naming, idempotent stage behavior, and
    optional legacy resample translation. ``max_workers`` bounds parallelism so
    callers can balance throughput against memory/CPU pressure.
    """

    base_path = Path(base_folder)
    base_path.mkdir(parents=True, exist_ok=True)

    method_norm = (resample_method or "convolution").lower()
    parallel_mode = max_workers is not None and max_workers > 1

    first_run_detected = False
    if flight_lines:
        for stem in flight_lines:
            try:
                paths = get_flight_paths(base_path, stem)
            except Exception:  # pragma: no cover - defensive fallback
                first_run_detected = True
                break
            work_dir = Path(paths["work_dir"])
            raw_img = work_dir / f"{stem}_envi.img"
            raw_hdr = work_dir / f"{stem}_envi.hdr"
            if not (raw_img.exists() and raw_hdr.exists()):
                first_run_detected = True
                break

    if first_run_detected:
        logger.info(
            "✨ First run detected: outputs will be created as needed. Existing files will be reused automatically."
        )
    else:
        logger.info(
            "✨ Existing ENVI exports detected — pipeline will reuse validated files automatically."
        )

    # Phase A: ensure downloads exist before spinning up heavy processing
    for flight_stem in flight_lines:
        stage_download_h5(
            base_folder=base_path,
            site_code=site_code,
            year_month=year_month,
            product_code=product_code,
            flight_stem=flight_stem,
        )

    def _worker(flight_stem: str) -> str:
        with _scoped_log_prefix(flight_stem):
            process_one_flightline(
                base_folder=base_path,
                product_code=product_code,
                flight_stem=flight_stem,
                resample_method=method_norm,
                brightness_offset=brightness_offset,
                parallel_mode=parallel_mode,
            )
        return flight_stem

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for flight_stem in flight_lines:
            futures.append(pool.submit(_worker, flight_stem))

        for fut in as_completed(futures):
            done_flight = fut.result()
            logger.info("🎉 Finished pipeline for %s (parallel worker join)", done_flight)

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
