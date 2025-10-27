#!/usr/bin/env python3
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

from cross_sensor_cal.corrections import (
    apply_brdf_correct,
    apply_topo_correct,
)
from cross_sensor_cal.envi_writer import EnviWriter
from cross_sensor_cal.neon_cube import NeonCube
from cross_sensor_cal.resample import resample_chunk_to_sensor

from ..envi_download import download_neon_flight_lines
from ..file_types import (
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceCoefficientsFile,
    NEONReflectanceENVIFile,
    NEONReflectanceFile,
    NEONReflectanceResampledENVIFile,
)
from ..neon_to_envi import neon_to_envi
from ..topo_and_brdf_correction import apply_offset_to_envi
from ..standard_resample import translate_to_other_sensors
from ..mask_raster import mask_raster_with_polygons
from ..polygon_extraction import control_function_for_extraction
from ..file_sort import generate_file_move_list

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))


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
    base_folder="output",
    resample_method: str = "convolution",
    max_workers: int = 1,
    skip_download_if_present: bool = True,
    force_config: bool = False,
    brightness_offset: float = 0.0,
    verbose: bool = False,
    **kwargs,
):
    base_path = Path(base_folder)
    base_path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("\n📥 Downloading NEON flight lines...")
    existing_h5 = list(base_path.rglob("*.h5"))
    if skip_download_if_present and existing_h5:
        _warn_skip_exists("download", existing_h5, verbose)
    else:
        try:
            _emit("⬇️  Fetching flight line HDF5...", bars=None, verbose=verbose)
            with _silence_noise(enabled=not verbose):
                download_neon_flight_lines(out_dir=base_path, **kwargs)
        except Exception as exc:
            raise RuntimeError(str(exc) + _stale_hint("download")) from exc
    if verbose:
        print("✅ Download step complete.")

    flight_lines = kwargs.get("flight_lines") or []
    bars: dict[str, tqdm] = {}
    totals = {fl: 1 for fl in flight_lines}  # download slot always tracked

    if not verbose and tqdm is not None and flight_lines:
        bar_fmt = "{desc:<14} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps"
        for fl in flight_lines:
            pretty = _pretty_line(fl)
            bars[pretty] = tqdm(
                total=_safe_total(totals[fl]),
                unit="steps",
                desc=pretty,
                bar_format=bar_fmt,
                dynamic_ncols=True,
                leave=True,
            )
    elif not verbose and tqdm is None:
        print("⚠️  tqdm not installed; progress bars disabled.")

    def _key_for_hint(line_hint: str | None) -> str | None:
        if not line_hint or not bars:
            return None
        key = _pretty_line(line_hint)
        return key if key in bars else None

    def _plan_step_for_line(line_hint: str | None, amount: int = 1) -> None:
        key = _key_for_hint(line_hint)
        if key and amount:
            bars[key].total += amount
            bars[key].refresh()

    def _complete_step_for_line(line_hint: str | None, amount: int = 1) -> None:
        key = _key_for_hint(line_hint)
        if key and amount:
            # Ensure total never lags behind completion when planning happens late
            progress = bars[key]
            if progress.total < progress.n + amount:
                progress.total = progress.n + amount
                progress.refresh()
            progress.update(amount)

    def _plan_step_for_path(path_obj: Path, amount: int = 1) -> None:
        candidate = _pretty_line(str(path_obj))
        if candidate in bars:
            _plan_step_for_line(candidate, amount)

    def _complete_step_for_path(path_obj: Path, amount: int = 1) -> None:
        candidate = _pretty_line(str(path_obj))
        if candidate in bars:
            _complete_step_for_line(candidate, amount)

    for fl in flight_lines:
        _tick_download_slot(base_path, fl, lambda path, line=fl: _complete_step_for_line(line))

    if verbose:
        print("📦 Step 2/5 Converting H5 files to ENVI format...")
    for fl in flight_lines:
        _plan_step_for_line(fl)
        if _line_outputs_present(base_path, fl):
            _warn_skip_exists(
                "H5→ENVI (main+ancillary)", [fl], verbose, bars, scope=_pretty_line(fl)
            )
            _complete_step_for_line(fl)
            continue

        h5s_for_line = [h5 for h5 in base_path.rglob("*.h5") if _belongs_to(fl, h5)]
        if not h5s_for_line:
            if verbose:
                logging.warning("No .h5 files found for line: %s", fl)
            _complete_step_for_line(fl)
            continue
        try:
            _emit(
                f"📦 Exporting ENVI (main + ancillary) [{_pretty_line(fl)}]...",
                bars,
                verbose=verbose,
            )
            with _silence_noise(enabled=not verbose):
                neon_to_envi(images=[str(p) for p in h5s_for_line], output_dir=str(base_path), anc=True)
            _complete_step_for_line(fl)
        except TypeError:
            for h5 in h5s_for_line:
                try:
                    with _silence_noise(enabled=not verbose):
                        neon_to_envi(images=[str(h5)], output_dir=str(base_path), anc=True)
                except Exception as exc:
                    raise RuntimeError(str(exc) + _stale_hint("H5→ENVI")) from exc
            _complete_step_for_line(fl)
        except Exception as exc:
            raise RuntimeError(str(exc) + _stale_hint("H5→ENVI")) from exc

    hdrs = list(base_path.rglob("*.hdr"))
    if not hdrs:
        logging.error("❌ No ENVI HDR files found after conversion. Investigate.")
    elif verbose:
        print(f"✅ ENVI conversion complete. {len(hdrs)} HDR files present.")

    overwrite_corrected = bool(kwargs.get("overwrite_corrected", False))
    overwrite_resampled = bool(kwargs.get("overwrite_resampled", False))

    method_norm = (resample_method or "convolution").lower()
    inline_resample = method_norm in {"convolution", "gaussian", "straight"}
    legacy_resample = method_norm in {"legacy", "resample"}

    def _normalize_product_code(value: str | None) -> str:
        if not value:
            return "30006.001"
        trimmed = value.strip()
        upper = trimmed.upper()
        if upper.startswith("DP1."):
            trimmed = trimmed[4:]
        elif upper.startswith("DP1"):
            trimmed = trimmed[3:]
            if trimmed.startswith("."):
                trimmed = trimmed[1:]
        trimmed = trimmed.strip("._")
        return trimmed or "30006.001"

    def _attach_brdf_coefficients(cube: NeonCube, refl_file: NEONReflectanceFile) -> None:
        candidates = NEONReflectanceCoefficientsFile.find_in_directory(
            refl_file.directory,
            correction="brdfandtopo",
            suffix="envi",
        )
        if not candidates:
            candidates = NEONReflectanceCoefficientsFile.find_in_directory(
                refl_file.directory,
                correction="brdf",
            )

        coeff_data = None
        for cand in candidates:
            try:
                with cand.file_path.open("r", encoding="utf-8") as coeff_file:
                    coeff_data = json.load(coeff_file)
                    break
            except Exception as exc:  # pragma: no cover - defensive guard against corrupt JSON
                logging.warning(
                    "⚠️  Could not read BRDF coefficient file %s: %s",
                    cand.file_path,
                    exc,
                )

        if coeff_data is None:
            logging.warning(
                "⚠️  No BRDF coefficient file found for %s; using neutral coefficients.",
                refl_file.path.name,
            )
            bands = cube.bands
            cube.brdf_coefficients = {
                "iso": np.ones(bands, dtype=np.float32),
                "vol": np.zeros(bands, dtype=np.float32),
                "geo": np.zeros(bands, dtype=np.float32),
                "volume_kernel": "RossThick",
                "geom_kernel": "LiSparseReciprocal",
            }
            return

        iso = np.asarray(coeff_data.get("iso"), dtype=np.float32)
        vol = np.asarray(coeff_data.get("vol"), dtype=np.float32)
        geo = np.asarray(coeff_data.get("geo"), dtype=np.float32)

        expected = cube.bands
        if iso.size != expected or vol.size != expected or geo.size != expected:
            logging.warning(
                "⚠️  BRDF coefficient size mismatch for %s (expected %d bands); using neutral coefficients.",
                refl_file.path.name,
                expected,
            )
            cube.brdf_coefficients = {
                "iso": np.ones(expected, dtype=np.float32),
                "vol": np.zeros(expected, dtype=np.float32),
                "geo": np.zeros(expected, dtype=np.float32),
                "volume_kernel": "RossThick",
                "geom_kernel": "LiSparseReciprocal",
            }
            return

        cube.brdf_coefficients = {
            "iso": iso,
            "vol": vol,
            "geo": geo,
            "volume_kernel": coeff_data.get("volume_kernel", "RossThick"),
            "geom_kernel": coeff_data.get("geom_kernel", "LiSparseReciprocal"),
        }

    def _build_sensor_srfs(
        wavelengths: np.ndarray,
        centers: Sequence[float],
        fwhm: Sequence[float],
    ) -> dict[str, np.ndarray]:
        srfs: dict[str, np.ndarray] = {}
        wl = np.asarray(wavelengths, dtype=np.float32)
        if wl.ndim != 1 or wl.size == 0:
            return srfs
        centers_arr = np.asarray(centers, dtype=np.float32)
        fwhm_arr = np.asarray(fwhm, dtype=np.float32) if len(fwhm) else np.array([], dtype=np.float32)
        for idx, center in enumerate(centers_arr):
            band_key = f"band_{idx + 1:02d}"
            if method_norm == "straight" or (idx < fwhm_arr.size and fwhm_arr[idx] <= 0):
                srf = np.zeros_like(wl)
                srf[int(np.abs(wl - center).argmin())] = 1.0
            else:
                width = float(fwhm_arr[idx]) if idx < fwhm_arr.size else float(fwhm_arr[-1]) if fwhm_arr.size else 0.0
                if width <= 0:
                    srf = np.zeros_like(wl)
                    srf[int(np.abs(wl - center).argmin())] = 1.0
                else:
                    sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                    srf = np.exp(-0.5 * ((wl - center) / sigma) ** 2)
            srfs[band_key] = np.asarray(srf, dtype=np.float32)
        return srfs

    def _prepare_sensor_writers(
        cube: NeonCube,
        corrected_file: NEONReflectanceBRDFCorrectedENVIFile,
        refl_file: NEONReflectanceFile,
        base_header: dict,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, EnviWriter], list[Path], list[Path]]:
        sensor_srfs: dict[str, dict[str, np.ndarray]] = {}
        sensor_writers: dict[str, EnviWriter] = {}
        skipped: list[Path] = []
        created: list[Path] = []
        for sensor_name, params in sensor_library.items():
            centers = params.get("wavelengths", [])
            fwhm = params.get("fwhms", [])
            if not centers:
                continue
            srfs = _build_sensor_srfs(cube.wavelengths, centers, fwhm)
            if not srfs:
                continue
            sensor_srfs[sensor_name] = srfs
            dir_prefix = f"{method_norm.capitalize()}_Reflectance_Resample"
            sensor_dir = corrected_file.directory / f"{dir_prefix}_{sensor_name.replace(' ', '_')}"
            sensor_dir.mkdir(parents=True, exist_ok=True)
            resampled_file = NEONReflectanceResampledENVIFile.from_components(
                domain=corrected_file.domain or refl_file.domain or "D00",
                site=corrected_file.site or refl_file.site or "SITE",
                date=corrected_file.date or refl_file.date or "00000000",
                sensor=sensor_name,
                suffix=corrected_file.suffix or "envi",
                folder=sensor_dir,
                time=corrected_file.time or refl_file.time,
                tile=corrected_file.tile or refl_file.tile,
                directional=corrected_file.directional,
                product=_normalize_product_code(refl_file.product),
            )
            resampled_img_path = resampled_file.path
            resampled_hdr_path = resampled_img_path.with_suffix(".hdr")
            if resampled_img_path.exists() and resampled_hdr_path.exists() and not overwrite_resampled:
                skipped.append(resampled_img_path)
                continue
            if overwrite_resampled:
                for existing in (resampled_img_path, resampled_hdr_path):
                    if existing.exists():
                        try:
                            existing.unlink()
                        except OSError as exc:
                            logging.warning(
                                "⚠️  Could not remove %s before resampling: %s",
                                existing,
                                exc,
                            )
            resampled_header = dict(base_header)
            resampled_header["bands"] = len(srfs)
            resampled_header["wavelength"] = list(map(float, centers))
            resampled_header["fwhm"] = list(map(float, fwhm))
            resampled_header["description"] = (
                f"{sensor_name} simulated reflectance ({method_norm}) based on BRDF + topographic corrected data"
            )
            resampled_header.setdefault("data type", 4)
            resampled_header.setdefault("byte order", base_header.get("byte order", 0))
            writer = EnviWriter(resampled_img_path.parent / resampled_img_path.stem, resampled_header)
            sensor_writers[sensor_name] = writer
            created.append(resampled_img_path)
        return sensor_srfs, sensor_writers, skipped, created

    sensor_library: dict[str, dict[str, list[float]]] = {}
    if inline_resample:
        bands_path = Path(PROJ_DIR) / "data" / "landsat_band_parameters.json"
        try:
            with bands_path.open("r", encoding="utf-8") as f:
                raw_library = json.load(f)
            if isinstance(raw_library, dict):
                sensor_library = {k: v for k, v in raw_library.items() if isinstance(v, dict)}
            else:
                logging.error("❌ Unexpected SRF library format in %s. Inline resampling disabled.", bands_path)
                inline_resample = False
        except FileNotFoundError:
            logging.error("❌ Sensor response library not found at %s. Inline resampling disabled.", bands_path)
            inline_resample = False
        except json.JSONDecodeError as exc:
            logging.error("❌ Could not parse %s (%s). Inline resampling disabled.", bands_path, exc)
            inline_resample = False

    reflectance_h5: list[NEONReflectanceFile] = []
    for h5_path in sorted(base_path.rglob("*.h5")):
        try:
            refl_file = NEONReflectanceFile.from_filename(h5_path)
        except ValueError:
            continue
        if flight_lines and not any(_belongs_to(fl, h5_path) for fl in flight_lines):
            continue
        reflectance_h5.append(refl_file)

    if verbose:
        print("⛰️ Step 3/5 Applying topographic and BRDF corrections with NeonCube...")

    errors = 0
    corrected_files: list[NEONReflectanceBRDFCorrectedENVIFile] = []
    resampled_outputs_all: list[Path] = []
    offset_applied_during_processing = False

    for refl_file in reflectance_h5:
        line_hint = refl_file.tile or refl_file.name
        _plan_step_for_line(line_hint)

        try:
            cube = NeonCube(h5_path=refl_file.file_path)
        except ValueError as exc:
            errors += 1
            logging.error(
                "⚠️  Skipping %s due to missing ancillary data: %s",
                refl_file.path.name,
                exc,
            )
            _complete_step_for_line(line_hint)
            continue
        except Exception as exc:
            errors += 1
            logging.error(
                "⚠️  Failed to initialise NeonCube for %s: %s",
                refl_file.path.name,
                exc,
            )
            _complete_step_for_line(line_hint)
            continue

        corrected_file = NEONReflectanceBRDFCorrectedENVIFile.from_components(
            domain=refl_file.domain or "D00",
            site=refl_file.site or "SITE",
            date=refl_file.date or "00000000",
            time=refl_file.time,
            suffix="envi",
            folder=refl_file.directory,
            tile=refl_file.tile,
            directional=getattr(refl_file, "directional", False),
            product=_normalize_product_code(refl_file.product),
        )
        out_img_path = corrected_file.path
        out_hdr_path = out_img_path.with_suffix(".hdr")

        if out_img_path.exists() and out_hdr_path.exists() and not overwrite_corrected:
            _warn_skip_exists(
                "topo_and_brdf_correction",
                [out_img_path, out_hdr_path],
                verbose,
                bars,
                scope=_pretty_line(line_hint),
            )
            _complete_step_for_line(line_hint)
            corrected_files.append(corrected_file)
            continue

        if overwrite_corrected:
            for existing in (out_img_path, out_hdr_path):
                if existing.exists():
                    try:
                        existing.unlink()
                    except OSError as exc:
                        logging.warning(
                            "⚠️  Could not remove %s before rewriting: %s",
                            existing,
                            exc,
                        )

        header = cube.build_envi_header()
        header["description"] = (
            "BRDF + topographic corrected reflectance (float32); generated by cross-sensor-cal pipeline"
        )
        header.setdefault("data type", 4)
        header.setdefault("byte order", 0)
        if hasattr(cube, "no_data"):
            header.setdefault("data ignore value", float(getattr(cube, "no_data")))

        try:
            _attach_brdf_coefficients(cube, refl_file)
        except Exception as exc:  # pragma: no cover - unexpected coefficient failure
            errors += 1
            logging.error(
                "⚠️  Failed to prepare BRDF coefficients for %s: %s",
                refl_file.path.name,
                exc,
            )
            _complete_step_for_line(line_hint)
            continue

        writer = EnviWriter(out_img_path.parent / out_img_path.stem, header)

        sensor_srfs: dict[str, dict[str, np.ndarray]] = {}
        sensor_writers: dict[str, EnviWriter] = {}
        skipped_sensor_paths: list[Path] = []
        new_sensor_paths: list[Path] = []
        if inline_resample and sensor_library:
            sensor_srfs, sensor_writers, skipped_sensor_paths, new_sensor_paths = _prepare_sensor_writers(
                cube,
                corrected_file,
                refl_file,
                header,
            )
            if skipped_sensor_paths:
                _warn_skip_exists(
                    "resample",
                    skipped_sensor_paths,
                    verbose,
                    bars,
                    scope=_pretty_line(line_hint),
                )

        _emit(f"Processing flightline {cube.base_key} ...", bars, verbose=verbose)

        offset_value: np.float32 | None = None
        if brightness_offset and float(brightness_offset) != 0.0:
            offset_value = np.float32(float(brightness_offset))

        try:
            for ys, ye, xs, xe, raw_chunk in cube.iter_chunks():
                chunk = np.asarray(raw_chunk, dtype=np.float32)
                corrected_chunk = apply_topo_correct(cube, chunk, ys, ye, xs, xe)
                corrected_chunk = apply_brdf_correct(cube, corrected_chunk, ys, ye, xs, xe)
                if offset_value is not None:
                    corrected_chunk = corrected_chunk + offset_value
                    offset_applied_during_processing = True
                corrected_chunk = corrected_chunk.astype(np.float32, copy=False)
                writer.write_chunk(corrected_chunk, ys, xs)
                if sensor_writers:
                    mask = cube.mask_no_data[ys:ye, xs:xe] if hasattr(cube, "mask_no_data") else None
                    no_data_value = np.float32(getattr(cube, "no_data", np.nan))
                    for sensor_name, srfs in sensor_srfs.items():
                        sensor_chunk = resample_chunk_to_sensor(corrected_chunk, cube.wavelengths, srfs)
                        if mask is not None:
                            sensor_chunk = np.where(mask[..., np.newaxis], sensor_chunk, no_data_value)
                        sensor_writers[sensor_name].write_chunk(sensor_chunk.astype(np.float32, copy=False), ys, xs)
        except Exception as exc:
            errors += 1
            logging.error(
                "⚠️  Correction failed for %s: %r%s",
                refl_file.path.name,
                exc,
                _stale_hint("topo_and_brdf_correction"),
            )
        finally:
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - close should rarely fail
                logging.error("⚠️  Failed to close writer for %s: %s", refl_file.path.name, exc)
            for sensor_writer in sensor_writers.values():
                try:
                    sensor_writer.close()
                except Exception as exc:  # pragma: no cover - close should rarely fail
                    logging.error(
                        "⚠️  Failed to close resampled writer for %s (%s): %s",
                        refl_file.path.name,
                        sensor_writer.out_stem,
                        exc,
                    )

        _emit(
            f"Wrote corrected ENVI for {cube.base_key} to {out_img_path}",
            bars,
            verbose=verbose,
        )

        corrected_files.append(corrected_file)
        resampled_outputs_all.extend(new_sensor_paths)
        _complete_step_for_line(line_hint)

    if verbose:
        print(
            f"✅ Corrections done. Corrected files found: {len(corrected_files)}. Errors: {errors}."
        )

    if legacy_resample:
        if verbose:
            print("🔁 Step 4/5 Resampling (legacy translate_to_other_sensors)...")
        resample_translation_to_other_sensors(base_path)
    elif inline_resample and not sensor_library:
        logging.warning("⚠️  Inline resampling requested but no sensor library was available.")

    offset_requested = brightness_offset and float(brightness_offset) != 0.0
    if offset_requested and not offset_applied_during_processing:
        if verbose:
            print(f"🧮 Applying brightness offset: {float(brightness_offset):+g}")
        try:
            names_to_match = ["brdfandtopo_corrected_envi", "resampled_envi"]
            candidates = [
                path
                for path in base_path.rglob("*.img")
                if any(name in path.name for name in names_to_match)
            ]
            if not candidates:
                _warn_skip_exists(
                    "brightness_offset (no eligible targets)",
                    candidates,
                    verbose,
                    bars,
                )
            changed = apply_offset_to_envi(
                input_dir=base_path,
                offset=float(brightness_offset),
                clip_to_01=True,
                only_if_name_contains=names_to_match,
            )
            for path in candidates:
                _plan_step_for_path(path)
                _complete_step_for_path(path)
            if verbose:
                print(f"✅ Offset applied to {changed} ENVI file(s).")
        except Exception as exc:
            logging.error("⚠️  Offset application failed: %r%s", exc, _stale_hint("brightness_offset"))
    elif offset_requested and offset_applied_during_processing and verbose:
        print("🧮 Brightness offset already applied during chunk processing; skipping post-hoc offset.")

    if bars:
        for progress in bars.values():
            progress.close()

    print("🎉 Pipeline complete!")

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
        max_workers=max_workers,
        skip_download_if_present=skip_download_if_present,
        force_config=force_config,
        brightness_offset=brightness_offset,
        verbose=verbose,
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
