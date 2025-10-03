import argparse
import os
import subprocess
from pathlib import Path

# Optional progress bars (fallback to no-bars if tqdm not present)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

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
    resample_method: str = 'convolution',
    max_workers: int = 1,
    skip_download_if_present: bool = True,
    force_config: bool = False,
    brightness_offset: float = 0.0,
    verbose: bool = False,
    **kwargs,
):
    base_path = Path(base_folder)
    base_path.mkdir(parents=True, exist_ok=True)

    flight_lines = kwargs.get("flight_lines") or []
    bars = {}
    if not verbose and flight_lines and tqdm is not None:
        for flight_line in flight_lines:
            bars[flight_line] = tqdm(total=0, unit="task", desc=flight_line, leave=True)
    elif not verbose and tqdm is None:
        print("⚠️  tqdm not installed; progress bars disabled.")

    def _belongs_to(line: str, path_obj: Path) -> bool:
        name = path_obj.name
        return (line in name) or (line in str(path_obj.parent))

    def _add_total(line: str, amount: int) -> None:
        if bars and line in bars and amount:
            current_total = bars[line].total or 0
            bars[line].total = current_total + amount
            bars[line].refresh()

    def _tick_for_path(path_obj: Path) -> None:
        if not bars:
            return
        for line in flight_lines:
            if _belongs_to(line, path_obj):
                bars[line].update(1)
                break

    # Step 1: Download NEON flight lines
    if verbose:
        print("\n📥 Downloading NEON flight lines...")
    if skip_download_if_present and list(base_path.rglob("*.h5")):
        if verbose:
            print("HDF5 files already present. Skipping download.")
    else:
        download_neon_flight_lines(out_dir=base_path, **kwargs)
    if verbose:
        print("✅ Download complete.\n")

    # Step 2: Convert H5 to ENVI format using neon_to_envi
    if verbose:
        print("📦 Converting H5 files to ENVI format...")
    h5_files = list(base_path.rglob("*.h5"))
    if not h5_files:
        if verbose:
            print("❌ No .h5 files found for conversion.")
    else:
        pending = [
            h5
            for h5 in h5_files
            if not any(h5.with_suffix(ext).exists() for ext in (".hdr", ".img", ".dat"))
        ]
        if not pending:
            if verbose:
                print("All H5 files already converted. Skipping.")
        else:
            if bars:
                for flight_line in flight_lines:
                    tasks = sum(1 for h5 in pending if _belongs_to(flight_line, h5))
                    _add_total(flight_line, tasks)
            for index, h5_file in enumerate(pending, start=1):
                if verbose:
                    print(f"🔄 [{index}/{len(pending)}] Converting: {h5_file.name}")
                neon_to_envi(images=[str(h5_file)], output_dir=str(base_path), anc=True)
                if verbose:
                    print(f"✅ Finished: {h5_file.name}\n")
                _tick_for_path(h5_file)
    if verbose:
        print("✅ ENVI conversion complete.\n")

    # Step 3: Generate configuration JSON
    if verbose:
        print("📝 Generating configuration JSON...")
    config_jsons = list(base_path.rglob("*_reflectance_envi_config_envi.json"))
    if force_config or not config_jsons:
        generate_config_json(base_path, num_cpus=max_workers)
        if verbose:
            print("✅ Config JSON generation complete.\n")
    else:
        if verbose:
            print("Existing config JSON found. Skipping (use --force-config to regenerate).\n")

    config_files = NEONReflectanceConfigFile.find_in_directory(base_path)
    if bars and config_files:
        for flight_line in flight_lines:
            tasks = sum(1 for cfg in config_files if _belongs_to(flight_line, cfg.file_path))
            _add_total(flight_line, tasks)
        for cfg in config_files:
            _tick_for_path(cfg.file_path)

    # Step 4: Apply topographic and BRDF corrections
    if verbose:
        print("⛰️ Applying topographic and BRDF corrections...")

    if config_files:
        if bars:
            for flight_line in flight_lines:
                tasks = sum(1 for cfg in config_files if _belongs_to(flight_line, cfg.file_path))
                _add_total(flight_line, tasks)
        for config_file in config_files:
            if verbose:
                print(f"⚙️ Applying corrections to: {config_file.file_path}")
            topo_and_brdf_correction(str(config_file.file_path))
            _tick_for_path(config_file.file_path)
        if verbose:
            print("✅ All corrections applied.\n")
    else:
        if verbose:
            print("❌ No configuration JSON files found. Skipping corrections.\n")

    # Step 5: Resample and translate data to other sensor formats
    if resample_method == 'convolution':
        if verbose:
            print("🔁 Resampling and translating data...")
        corrected_files = NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(base_path)
        if not corrected_files:
            print("❌ No BRDF-corrected ENVI files found for resampling. Check naming or previous steps.\n")
        else:
            if verbose:
                print(f"📂 Found {len(corrected_files)} BRDF-corrected files to process.")
            if bars:
                for flight_line in flight_lines:
                    tasks = sum(
                        1 for corrected in corrected_files if _belongs_to(flight_line, corrected.path)
                    )
                    _add_total(flight_line, tasks)
            for index, corrected_file in enumerate(corrected_files, start=1):
                if verbose:
                    print(f"🔄 [{index}/{len(corrected_files)}] Resampling: {corrected_file.name}")
                convolution_resample(corrected_file.directory)
                if verbose:
                    print(f"✅ Resampled: {corrected_file.name}\n")
                _tick_for_path(corrected_file.path)
        if verbose:
            print("✅ Resampling and translation complete.\n")
    elif resample_method == 'resample':
        resample_translation_to_other_sensors(base_path)

    # TODO: Move this to after the convolution diagnostic option to keep the unadjusted ones
    if brightness_offset and float(brightness_offset) != 0.0:
        if verbose:
            print(f"🧮 Applying brightness offset: {brightness_offset:+g}")
        try:
            names_to_match = [
                "brdfandtopo_corrected_envi",
                "resampled_envi",
            ]
            candidates = [
                path
                for path in base_path.rglob("*.img")
                if any(name in path.name for name in names_to_match)
            ]
            if bars:
                for flight_line in flight_lines:
                    tasks = sum(1 for path in candidates if _belongs_to(flight_line, path))
                    _add_total(flight_line, tasks)
            changed = apply_offset_to_envi(
                input_dir=base_path,
                offset=float(brightness_offset),
                clip_to_01=True,
                only_if_name_contains=names_to_match,
            )
            if verbose:
                print(f"✅ Offset applied to {changed} ENVI file(s).")
            if bars:
                count_remaining = changed
                for path in candidates:
                    if count_remaining <= 0:
                        break
                    _tick_for_path(path)
                    count_remaining -= 1
        except Exception as exc:
            print(f"⚠️ Offset application failed: {exc}")

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

    args = parser.parse_args()
    if args.reflectance_offset and float(args.reflectance_offset) != 0.0:
        args.brightness_offset = float(args.reflectance_offset)
        print("⚠️  --reflectance-offset is deprecated; using --brightness-offset instead.")
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
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
