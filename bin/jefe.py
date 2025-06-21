import os
import json
from pathlib import Path

import ray

from src.envi_download import download_neon_flight_lines
from src.file_types import NEONReflectanceConfigFile, NEONReflectanceBRDFCorrectedENVIHDRFile, \
    NEONReflectanceBRDFCorrectedENVIFile, NEONReflectanceENVIFile, SensorType, NEONReflectanceResampledENVIFile
from src.neon_to_envi import flight_lines_to_envi
from src.topo_and_brdf_correction import generate_config_json, topo_and_brdf_correction
from src.convolution_resample import resample as convolution_resample
from src.standard_resample import translate_to_other_sensors
from src.mask_raster import mask_raster_with_polygons, find_raster_files
from src.polygon_extraction import control_function_for_extraction

from tqdm import tqdm

import glob

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))


def go_forth_and_multiply(base_folder="output", envi_outdir: str = None, resample_outdir: str = None, resample_method: str = 'convolution', **kwargs):
    # Create the base folder if it doesn't exist
    os.makedirs(base_folder, exist_ok=True)
    if envi_outdir is None:
        envi_outdir = os.path.join(base_folder, 'neon_to_envi')
    os.makedirs(envi_outdir, exist_ok=True)

    if resample_outdir is None:
        resample_outdir = os.path.join(base_folder, 'resampled')

    # Step 1: Download NEON flight lines with kwargs passed to this step
    download_neon_flight_lines(out_dir=base_folder, **kwargs)

    # Step 2: Convert flight lines to ENVI format
    flight_lines_to_envi(input_dir=base_folder, output_dir=envi_outdir)
    # Step 3: Generate configuration JSON
    generate_config_json(envi_outdir)
    # Step 4: Apply topographic and BRDF corrections
    apply_topo_and_brdf_corrections(Path(envi_outdir))

    # Step 5: Resample and translate data to other sensor formats
    if resample_method == 'convolution':
        convolution_resample(Path(envi_outdir), Path(resample_outdir))
    elif resample_method == 'resample':
        resample_translation_to_other_sensors(Path(envi_outdir))

    print("Processing complete.")


def apply_topo_and_brdf_corrections(input_dir: Path):
    print("Starting topo and BRDF correction. This takes a long time.")
    envi_config_files = NEONReflectanceConfigFile.find_in_directory(input_dir, "envi")

    for envi_config_file in envi_config_files:
        print(f"\nProcessing folder for BRDF correction: {envi_config_file.directory}")
        try:
            topo_and_brdf_correction(envi_config_file.file_path)
        except Exception as e:
             print(f"❌ Error executing BRDF correction: {e}")
        else:
             print(f"✅ Successfully processed BRDF correction for: {envi_config_file.file_path}")

    print("\nAll topo and BRDF corrections completed.")



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
    raster_files = (NEONReflectanceENVIFile.find_in_directory(base_folder) +
                    NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(base_folder, 'envi') +
                    NEONReflectanceResampledENVIFile.find_all_sensors_in_directory(base_folder, 'envi'))

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


def process_all_subdirectories(parent_directory, polygon_path):
    """Searches and processes all subdirectories."""
    subdirectories = [
        os.path.join(parent_directory, sub_dir) for sub_dir in os.listdir(parent_directory) if
        os.path.isdir(os.path.join(parent_directory, sub_dir))
    ]

    with tqdm(total=len(subdirectories), desc="Processing subdirectories", unit="directory") as pbar:
        for subdirectory in subdirectories:
            try:
                control_function_for_extraction(subdirectory, polygon_path)
                pbar.update(1)
            except Exception as e:
                print(f"[ERROR] Error processing directory '{subdirectory}': {e}")




def jefe(base_folder, site_code, year_month, flight_lines, polygon_layer_path: str, product_code = 'DP1.30006.001'):
    """
    A control function that orchestrates the processing of spectral data.
    It first calls go_forth_and_multiply to generate necessary data and structures,
    then processes all subdirectories within the base_folder.

    Parameters:
    - base_folder (str): The base directory for both operations.
    - site_code (str): Site code for go_forth_and_multiply.
    - product_code (str): Product code for go_forth_and_multiply.
    - year_month (str): Year and month for go_forth_and_multiply.
    - flight_lines (list): A list of flight lines for go_forth_and_multiply.
    """
    # First, call go_forth_and_multiply with the provided parameters
    go_forth_and_multiply(
        base_folder=base_folder,
        site_code=site_code,
        product_code=product_code,
        year_month=year_month,
        flight_lines=flight_lines
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
    process_all_subdirectories(base_folder, polygon_layer_path)

    # Finally, clean the CSV files by removing rows with any NaN values
    # clean_csv_files_in_subfolders(base_folder)
    #
    # merge_csvs_by_columns(base_folder)
    # validate_output_files(base_folder)

    print(
        "Jefe finished. Please check for the _with_mask_and_all_spectra.csv for your  hyperspectral data from NEON flight lines extracted to match your provided polygons")
