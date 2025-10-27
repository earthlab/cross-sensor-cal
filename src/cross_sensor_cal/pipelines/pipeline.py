#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Sequence

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

from ..envi_download import download_neon_flight_lines
from ..file_types import NEONReflectanceConfigFile, \
    NEONReflectanceBRDFCorrectedENVIFile, NEONReflectanceENVIFile, NEONReflectanceResampledENVIFile
from ..neon_to_envi import neon_to_envi
from ..topo_and_brdf_correction import (
    generate_config_json,
    topo_and_brdf_correction,
    apply_offset_to_envi,
)
from ..convolution_resample import resample as convolution_resample
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

    if verbose:
        print("📝 Step 3/5 Generating configuration JSON...")
    existing_cfgs = list(base_path.rglob("*reflectance_envi_config_envi.json"))
    if not force_config and existing_cfgs:
        _warn_skip_exists("generate_config_json", existing_cfgs, verbose, bars)
        for cfg in existing_cfgs:
            _plan_step_for_path(Path(cfg))
            _complete_step_for_path(Path(cfg))
    else:
        try:
            generate_config_json(base_path)
            new_cfgs = list(NEONReflectanceConfigFile.find_in_directory(base_path))
            for cfg in new_cfgs:
                _plan_step_for_line(cfg.tile or str(cfg.file_path))
                _complete_step_for_line(cfg.tile or str(cfg.file_path))
        except Exception as exc:
            raise RuntimeError(str(exc) + _stale_hint("generate_config_json")) from exc

    config_files = NEONReflectanceConfigFile.find_in_directory(base_path)
    if verbose:
        print(f"✅ Config JSON step complete. Found {len(config_files)} configs.")

    if verbose:
        print("⛰️ Step 4/5 Applying topographic and BRDF corrections...")
    if config_files:
        errors = 0
        for cfg in config_files:
            corrected_dir = cfg.file_path.parent
            existing_corrected = list(corrected_dir.glob("*brdfandtopo_corrected_envi*.hdr")) + list(
                corrected_dir.glob("*brdfandtopo_corrected_envi*.img")
            )
            _plan_step_for_line(cfg.tile or str(cfg.file_path))
            if existing_corrected:
                _warn_skip_exists(
                    "topo_and_brdf_correction",
                    existing_corrected,
                    verbose,
                    bars,
                    scope=_pretty_line(cfg.tile or cfg.file_path.name),
                )
                _complete_step_for_line(cfg.tile or str(cfg.file_path))
                continue
            try:
                topo_and_brdf_correction(str(cfg.file_path))
                _complete_step_for_line(cfg.tile or str(cfg.file_path))
            except Exception as exc:
                errors += 1
                logging.error(
                    "⚠️  Correction failed for %s: %r%s",
                    cfg.file_path.name,
                    exc,
                    _stale_hint("topo_and_brdf_correction"),
                )
        corrected = NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(base_path)
        if verbose:
            print(f"✅ Corrections done. Corrected files found: {len(corrected)}. Errors: {errors}.")
    else:
        logging.warning("❌ No configuration JSON files found. Skipping corrections.")

    # --- Step 5/5: Resampling/harmonization ---
    method_norm = (resample_method or "convolution").lower()
    if method_norm in {"resample", "legacy"}:
        if verbose:
            print("🔁 Step 5/5 Resampling (legacy translate_to_other_sensors)...")
        resample_translation_to_other_sensors(base_path)
    elif method_norm in {"convolution", "gaussian", "straight"}:
        if verbose:
            print(f"🔁 Step 5/5 Resampling data ({method_norm})...")
        corrected_files = NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(base_path)
        if not corrected_files:
            logging.warning("❌ No BRDF-corrected ENVI files found for resampling. Check naming or previous steps.")
        else:
            if verbose:
                print(f"📂 Found {len(corrected_files)} BRDF-corrected files to process.")
            for corrected_file in corrected_files:
                _plan_step_for_line(corrected_file.tile or str(corrected_file.path))
                existing_resampled = [
                    resampled.path
                    for resampled in NEONReflectanceResampledENVIFile.find_in_directory(
                        corrected_file.directory
                    )
                ]
                if existing_resampled:
                    _warn_skip_exists(
                        "resample",
                        existing_resampled,
                        verbose,
                        bars,
                        scope=corrected_file.path.name,
                    )
                    _complete_step_for_line(corrected_file.tile or str(corrected_file.path))
                    continue
                try:
                    # Prefer new signature with method=...; fall back if not available
                    try:
                        convolution_resample(corrected_file.directory, method=method_norm)
                    except TypeError:
                        # Older resampler without 'method' kwarg—call as before
                        convolution_resample(corrected_file.directory)
                except Exception as exc:
                    logging.error(
                        "⚠️  Resample failed for %s: %r%s",
                        corrected_file.name,
                        exc,
                        _stale_hint("resample"),
                    )
                _complete_step_for_line(corrected_file.tile or str(corrected_file.path))
        if verbose:
            print(f"✅ Resampling complete ({method_norm}).")
    else:
        logging.warning("Unknown resample_method=%s (skipping Step 5).", resample_method)

    if brightness_offset and float(brightness_offset) != 0.0:
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
