#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import subprocess
import sys
from enum import Enum
from io import StringIO
from pathlib import Path

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


class OutputMode(Enum):
    NORMAL = "normal"
    VERBOSE = "verbose"
    SKIP = "skip"

    def __str__(self) -> str:  # pragma: no cover - argparse representation
        return self.value

    @classmethod
    def from_string(cls, value: str | None) -> "OutputMode":
        if not value:
            return cls.NORMAL
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unsupported output mode: {value}")


class BarBook:
    """Manage per-flight-line progress bars."""

    def __init__(self, mode: OutputMode) -> None:
        self._mode = mode
        self._bars: dict[str, "tqdm"] = {}

    @property
    def enabled(self) -> bool:
        return self._mode == OutputMode.NORMAL and tqdm is not None

    def add(self, key: str, *, total: int, desc: str | None = None) -> None:
        if not self.enabled:
            return
        if key in self._bars:
            bar = self._bars[key]
            bar.total = max(bar.total, total)
            bar.refresh()
            return
        bar_fmt = "{desc:<14} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps"
        self._bars[key] = tqdm(
            total=max(1, int(total)),
            unit="steps",
            desc=desc or key,
            bar_format=bar_fmt,
            dynamic_ncols=True,
            leave=True,
        )

    def increment_total(self, key: str, amount: int) -> None:
        if amount <= 0:
            return
        bar = self._bars.get(key)
        if not bar:
            return
        bar.total += amount
        bar.refresh()

    def tick(self, key: str, amount: int = 1) -> None:
        bar = self._bars.get(key)
        if not bar:
            return
        bar.update(amount)

    def complete(self, key: str) -> None:
        bar = self._bars.get(key)
        if not bar:
            return
        if bar.total < bar.n:
            bar.total = bar.n
        remaining = bar.total - bar.n
        if remaining > 0:
            bar.update(remaining)
        else:
            bar.refresh()

    def close_all(self, *, complete_remaining: bool = False) -> None:
        for key, bar in list(self._bars.items()):
            if complete_remaining:
                self.complete(key)
            bar.close()


class SkipCollector:
    """Collect skipped items grouped by pipeline step."""

    def __init__(self) -> None:
        self._data: dict[str, list[str]] = {}

    def record(self, step: str, items: list[str]) -> None:
        if not items:
            return
        bucket = self._data.setdefault(step, [])
        bucket.extend(items)

    def is_empty(self) -> bool:
        return not self._data

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for step in sorted(self._data.keys()):
            items = sorted(dict.fromkeys(self._data[step]))
            lines.append(f"• {step}: {len(items)} item(s)")
            for name in items:
                lines.append(f"   - {name}")
        return lines


class Out:
    """Mode-aware output helper."""

    def __init__(self, mode: OutputMode, barbook: BarBook, skip_collector: SkipCollector) -> None:
        self.mode = mode
        self._barbook = barbook
        self._skips = skip_collector

    def _write(self, message: str) -> None:
        if self.mode == OutputMode.SKIP:
            return
        if self.mode == OutputMode.VERBOSE or not self._barbook.enabled:
            print(message)
            return
        try:
            tqdm.write(message)
        except Exception:  # pragma: no cover - tqdm fallback
            print(message)

    def step(self, message: str) -> None:
        if self.mode in (OutputMode.NORMAL, OutputMode.VERBOSE):
            self._write(message)

    def info(self, message: str) -> None:
        if self.mode in (OutputMode.NORMAL, OutputMode.VERBOSE):
            self._write(message)

    def warn(self, message: str) -> None:
        if self.mode == OutputMode.SKIP:
            return
        self._write(message)

    def error(self, message: str) -> None:
        print(message, file=sys.stderr)

    def skip(self, step: str, items, scope: str | None = None) -> None:
        names: list[str] = []
        for item in items:
            text = str(item)
            if os.sep in text:
                text = Path(text).name
            names.append(text)
        self._skips.record(step, names)
        if self.mode == OutputMode.SKIP:
            return
        scope_txt = f" [{scope}]" if scope else ""
        human_step = {
            "download": "download already present",
            "H5→ENVI (main+ancillary)": "ENVI + ancillary already exported",
            "generate_config_json": "config already present",
            "topo_and_brdf_correction": "BRDF+topo correction already present",
            "resample": "resampled outputs already present",
        }.get(step, step)
        msg = f"⏭️  Skipped{scope_txt}: {human_step} ({len(names)})"
        self._write(msg)

    def print_skip_summary(self) -> None:
        if self.mode != OutputMode.SKIP:
            return
        if self._skips.is_empty():
            print("Skipped summary: nothing to report.")
            return
        print("Skipped summary:")
        for line in self._skips.summary_lines():
            print(line)


def _pretty_line(line: str) -> str:
    """
    Return a short, readable label for a flight line (prefer the tile like L019-1).
    Falls back to the original string if no tile pattern found.
    """

    m = re.search(r"(L\d{3}-\d)", line)
    return m.group(1) if m else line


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


_RAY_FILTERS = [
    re.compile(r"^20\d{2}-\d{2}-\d{2} .*WARNING services\.py:.*object store is using /tmp"),
    re.compile(r"^20\d{2}-\d{2}-\d{2} .*INFO worker\.py:.*Started a local Ray instance\."),
    re.compile(r"^\[20\d{2}-\d{2}-\d{2} .* logging\.cc:\d+: Set ray log level"),
    re.compile(r"^\(raylet\) \[20\d{2}-\d{2}-\d{2} .* logging\.cc:\d+: Set ray log level"),
]

_HYTOOLS_GR = re.compile(r"^\(HyTools pid=\d+\)\s*(GR)+\s*$")

_ALLOW_LINES = [
    re.compile(r"Saved:"),
    re.compile(r"All processing complete"),
]


def _should_keep(line: str) -> bool:
    if any(pattern.search(line) for pattern in _ALLOW_LINES):
        return True
    if any(pattern.search(line) for pattern in _RAY_FILTERS):
        return False
    if _HYTOOLS_GR.search(line):
        return False
    return False


def run_noisy_step_quietly(argv: list[str], mode: OutputMode) -> int:
    """Run a noisy step in a subprocess and filter its output."""

    env = os.environ.copy()
    env.setdefault("RAY_LOG_TO_STDERR", "0")
    env.setdefault("RAY_BACKEND_LOG_LEVEL", "ERROR")
    env.setdefault("RAY_DISABLE_DASHBOARD", "1")
    env.setdefault("RAY_usage_stats_enabled", "0")
    env.setdefault("RAY_disable_usage_stats", "1")
    env.setdefault("RAY_LOG_TO_DRIVER", "0")

    if mode == OutputMode.VERBOSE:
        proc = subprocess.Popen(argv, env=env)
        return proc.wait()

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stderr is not None
    for stream in (proc.stdout, proc.stderr):
        for raw in stream:
            line = raw.rstrip("\n")
            if _should_keep(line):
                try:
                    if tqdm is not None:
                        tqdm.write(line)
                    else:
                        print(line)
                except Exception:
                    print(line)
    return proc.wait()

from src.envi_download import download_neon_flight_lines
from src.file_types import NEONReflectanceConfigFile, \
    NEONReflectanceBRDFCorrectedENVIFile, NEONReflectanceENVIFile, NEONReflectanceResampledENVIFile
from src.neon_to_envi import neon_to_envi
from src.topo_and_brdf_correction import (
    generate_config_json,
    topo_and_brdf_correction,
    apply_offset_to_envi,
)
from src.convolution_resample import resample as convolution_resample
from src.standard_resample import translate_to_other_sensors
from src.mask_raster import mask_raster_with_polygons
from src.polygon_extraction import control_function_for_extraction
from src.file_sort import generate_file_move_list

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
    output_mode: OutputMode = OutputMode.NORMAL,
    **kwargs,
):
    base_path = Path(base_folder)
    base_path.mkdir(parents=True, exist_ok=True)

    flight_lines = kwargs.get("flight_lines") or []

    skip_collector = SkipCollector()
    barbook = BarBook(output_mode)
    out = Out(output_mode, barbook, skip_collector)

    if output_mode == OutputMode.NORMAL and not barbook.enabled and flight_lines:
        out.warn("⚠️  tqdm not installed; progress bars disabled.")

    for fl in flight_lines:
        pretty = _pretty_line(fl)
        barbook.add(pretty, total=_safe_total(1), desc=pretty)

    def _key(line: str) -> str:
        return _pretty_line(line)

    def _add_total(line: str, amount: int) -> None:
        if amount:
            barbook.increment_total(_key(line), amount)

    def _add_total_for_path(path_obj: Path, amount: int = 1) -> None:
        for line in flight_lines:
            if _belongs_to(line, path_obj):
                _add_total(line, amount)
                break

    def _tick_line(line: str, amount: int = 1) -> None:
        barbook.tick(_key(line), amount)

    def _tick_for_path(path_obj: Path, amount: int = 1) -> None:
        for line in flight_lines:
            if _belongs_to(line, path_obj):
                _tick_line(line, amount)
                break

    try:
        out.step("📥 Downloading NEON flight lines...")
        existing_h5 = list(base_path.rglob("*.h5"))
        if skip_download_if_present and existing_h5:
            out.skip("download", existing_h5)
        else:
            try:
                out.step("⬇️  Fetching flight line HDF5...")
                with _silence_noise(enabled=output_mode != OutputMode.VERBOSE):
                    download_neon_flight_lines(out_dir=base_path, **kwargs)
            except Exception as exc:
                raise RuntimeError(str(exc) + _stale_hint("download")) from exc

        for fl in flight_lines:
            _tick_download_slot(base_path, fl, _tick_for_path)

        out.step("📦 Step 2/5 Converting H5 files to ENVI format...")
        for fl in flight_lines:
            _add_total(fl, 1)
            pretty = _pretty_line(fl)
            if _line_outputs_present(base_path, fl):
                out.skip("H5→ENVI (main+ancillary)", [fl], scope=pretty)
                _tick_line(fl)
                continue

            h5s_for_line = [h5 for h5 in base_path.rglob("*.h5") if _belongs_to(fl, h5)]
            if not h5s_for_line:
                out.warn(f"⚠️  No .h5 files found for line: {fl}")
                _tick_line(fl)
                continue
            try:
                out.step(f"📦 Exporting ENVI (main + ancillary) [{pretty}]...")
                if output_mode == OutputMode.VERBOSE:
                    neon_to_envi(images=[str(p) for p in h5s_for_line], output_dir=str(base_path), anc=True)
                else:
                    code = run_noisy_step_quietly(
                        [
                            "python",
                            "-m",
                            "src._noisy_wrappers",
                            "neon_to_envi",
                            "--images",
                            *[str(p) for p in h5s_for_line],
                            "--output_dir",
                            str(base_path),
                            "--anc",
                        ],
                        output_mode,
                    )
                    if code != 0:
                        raise RuntimeError("neon_to_envi failed" + _stale_hint("H5→ENVI"))
                _tick_line(fl)
            except TypeError:
                for h5 in h5s_for_line:
                    if output_mode == OutputMode.VERBOSE:
                        neon_to_envi(images=[str(h5)], output_dir=str(base_path), anc=True)
                    else:
                        code = run_noisy_step_quietly(
                            [
                                "python",
                                "-m",
                                "src._noisy_wrappers",
                                "neon_to_envi",
                                "--images",
                                str(h5),
                                "--output_dir",
                                str(base_path),
                                "--anc",
                            ],
                            output_mode,
                        )
                        if code != 0:
                            raise RuntimeError("neon_to_envi failed" + _stale_hint("H5→ENVI"))
                _tick_line(fl)
            except Exception as exc:
                raise RuntimeError(str(exc) + _stale_hint("H5→ENVI")) from exc

        hdrs = list(base_path.rglob("*.hdr"))
        if not hdrs:
            out.error("❌ No ENVI HDR files found after conversion. Investigate.")

        out.step("📝 Step 3/5 Generating configuration JSON...")
        existing_cfgs = list(base_path.rglob("*reflectance_envi_config_envi.json"))
        if not force_config and existing_cfgs:
            out.skip("generate_config_json", existing_cfgs)
            for cfg in existing_cfgs:
                path_obj = Path(cfg)
                _add_total_for_path(path_obj)
                _tick_for_path(path_obj)
        else:
            try:
                generate_config_json(base_path)
                new_cfgs = list(NEONReflectanceConfigFile.find_in_directory(base_path))
                for cfg in new_cfgs:
                    cfg_path = Path(cfg.file_path)
                    _add_total_for_path(cfg_path)
                    _tick_for_path(cfg_path)
            except Exception as exc:
                raise RuntimeError(str(exc) + _stale_hint("generate_config_json")) from exc

        config_files = NEONReflectanceConfigFile.find_in_directory(base_path)

        out.step("⛰️ Step 4/5 Applying topographic and BRDF corrections...")
        if config_files:
            for cfg in config_files:
                cfg_path = Path(cfg.file_path)
                _add_total_for_path(cfg_path)
                corrected_dir = cfg.file_path.parent
                existing_corrected = list(corrected_dir.glob("*brdfandtopo_corrected_envi*.hdr")) + list(
                    corrected_dir.glob("*brdfandtopo_corrected_envi*.img")
                )
                scope = _pretty_line(cfg.tile or cfg.file_path.name)
                if existing_corrected:
                    out.skip("topo_and_brdf_correction", existing_corrected, scope=scope)
                    _tick_for_path(cfg_path)
                    continue
                try:
                    if output_mode == OutputMode.VERBOSE:
                        topo_and_brdf_correction(str(cfg.file_path))
                    else:
                        code = run_noisy_step_quietly(
                            [
                                "python",
                                "-m",
                                "src._noisy_wrappers",
                                "topo",
                                "--config",
                                str(cfg.file_path),
                            ],
                            output_mode,
                        )
                        if code != 0:
                            raise RuntimeError(
                                "topo_and_brdf_correction failed" + _stale_hint("topo_and_brdf_correction")
                            )
                    _tick_for_path(cfg_path)
                except Exception as exc:
                    out.error(
                        f"⚠️  Correction failed for {cfg.file_path.name}: {exc}{_stale_hint('topo_and_brdf_correction')}"
                    )
        else:
            out.warn("❌ No configuration JSON files found. Skipping corrections.")

        if resample_method == "convolution":
            out.step("🔁 Step 5/5 Resampling and translating data (convolutional)...")
            corrected_files = NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(base_path)
            if not corrected_files:
                out.warn(
                    "❌ No BRDF-corrected ENVI files found for resampling. Check naming or previous steps."
                )
            else:
                for corrected_file in corrected_files:
                    _add_total_for_path(corrected_file.path)
                    existing_resampled = [
                        resampled.path
                        for resampled in NEONReflectanceResampledENVIFile.find_in_directory(
                            corrected_file.directory
                        )
                    ]
                    if existing_resampled:
                        out.skip("resample", existing_resampled, scope=corrected_file.path.name)
                        _tick_for_path(corrected_file.path)
                        continue
                    try:
                        if output_mode == OutputMode.VERBOSE:
                            convolution_resample(corrected_file.directory)
                        else:
                            code = run_noisy_step_quietly(
                                [
                                    "python",
                                    "-m",
                                    "src._noisy_wrappers",
                                    "resample",
                                    "--dir",
                                    str(corrected_file.directory),
                                ],
                                output_mode,
                            )
                            if code != 0:
                                raise RuntimeError("resample failed" + _stale_hint("resample"))
                    except Exception as exc:
                        out.error(f"⚠️  Resample failed for {corrected_file.name}: {exc}{_stale_hint('resample')}")
                    finally:
                        _tick_for_path(corrected_file.path)
        elif resample_method == "resample":
            resample_translation_to_other_sensors(base_path)
        else:
            out.warn(f"Unknown resample_method={resample_method} (skipping Step 5).")

        if brightness_offset and float(brightness_offset) != 0.0:
            out.step(f"🧮 Applying brightness offset: {float(brightness_offset):+g}")
            try:
                names_to_match = ["brdfandtopo_corrected_envi", "resampled_envi"]
                candidates = [
                    path
                    for path in base_path.rglob("*.img")
                    if any(name in path.name for name in names_to_match)
                ]
                if not candidates:
                    out.warn("No ENVI files found for brightness offset application.")
                changed = apply_offset_to_envi(
                    input_dir=base_path,
                    offset=float(brightness_offset),
                    clip_to_01=True,
                    only_if_name_contains=names_to_match,
                )
                for path in candidates:
                    _add_total_for_path(path)
                    _tick_for_path(path)
                out.info(f"✅ Offset applied to {changed} ENVI file(s).")
            except Exception as exc:
                out.error(f"⚠️  Offset application failed: {exc}{_stale_hint('brightness_offset')}")

        out.info("🎉 Pipeline complete!")

    finally:
        barbook.close_all(complete_remaining=True)
        if output_mode == OutputMode.SKIP:
            out.print_skip_summary()


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
    output_mode: OutputMode = OutputMode.NORMAL,
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
    - output_mode (OutputMode): Controls user-visible output style.
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
        output_mode=output_mode,
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

    if output_mode != OutputMode.SKIP:
        print(
            "Jefe finished. Please check for the _with_mask_and_all_spectra.csv for your  hyperspectral data from NEON flight lines extracted to match your provided polygons"
        )

def parse_args():
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
        "--verbose",
        action="store_true",
        help="Emit detailed per-step logs instead of compact progress bars.",
    )
    parser.add_argument(
        "--output",
        choices=[mode.value for mode in OutputMode],
        default=None,
        help="Output style: normal (default), verbose, or skip.",
    )

    args = parser.parse_args()
    if args.reflectance_offset and float(args.reflectance_offset) != 0.0:
        args.brightness_offset = float(args.reflectance_offset)
        print("⚠️  --reflectance-offset is deprecated; using --brightness-offset instead.")
    if args.output is None:
        args.output = OutputMode.VERBOSE.value if args.verbose else OutputMode.NORMAL.value
    elif args.verbose and args.output != OutputMode.VERBOSE.value:
        print("⚠️  --verbose flag ignored because --output was provided.", file=sys.stderr)
    return args


def main():
    args = parse_args()

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
        output_mode=OutputMode.from_string(args.output),
    )


if __name__ == "__main__":
    main()
