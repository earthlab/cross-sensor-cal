import os
import glob
import json
import time
import random
import subprocess
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import h5py
import ray
from shapely.geometry import box
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.features import rasterize
from sklearn.ensemble import GradientBoostingRegressor
import hytools as ht

# ============================
# Helper Classes and Functions
# ============================

class ENVIProcessor:
    """
    A class to handle ENVI raster data processing.
    """

    def __init__(self, file_path):
        """
        Initializes the ENVIProcessor with the given file path.

        Parameters:
        - file_path (str): Path to the ENVI raster file.
        """
        self.file_path = file_path
        self.data = None  # Holds the raster data array
        self.file_type = "envi"

    def load_data(self):
        """Loads the raster data from the file_path into self.data."""
        with rasterio.open(self.file_path) as src:
            self.data = src.read()  # Read all bands

    def get_chunk_from_extent(self, corrections=None, resample=False):
        """
        Retrieves a chunk of raster data based on extent and applies optional processing.

        Parameters:
        - corrections (list): List of corrections to apply.
        - resample (bool): Whether to resample the data.

        Returns:
        - np.ndarray: Processed chunk of raster data.
        """
        corrections = corrections or []
        self.load_data()  # Ensure data is loaded
        with rasterio.open(self.file_path) as src:
            bounds = src.bounds
            width, height = src.width, src.height
            col_start, line_start = 0, 0
            col_end, line_end = width, height

            # Extract the chunk
            chunk = self.data[:, line_start:line_end, col_start:col_end]

            # Apply any processing to chunk here...
            # Example: Flip chunk vertically
            chunk = np.flip(chunk, axis=1)

            return chunk

def load_and_combine_rasters(raster_paths):
    """
    Loads and combines raster data from a list of file paths.

    Parameters:
    - raster_paths (list): List of raster file paths.

    Returns:
    - np.ndarray: Combined raster data array.
    """
    chunks = []
    for path in raster_paths:
        processor = ENVIProcessor(path)
        chunk = processor.get_chunk_from_extent(corrections=['some_correction'], resample=False)
        chunks.append(chunk)

    combined_array = np.concatenate(chunks, axis=0)  # Combine along the first axis (bands)
    return combined_array

def download_neon_file(site_code, product_code, year_month, flight_line):
    """
    Downloads a NEON flight line file based on provided parameters.

    Parameters:
    - site_code (str): The site code.
    - product_code (str): The product code.
    - year_month (str): The year and month in 'YYYY-MM' format.
    - flight_line (str): The flight line identifier.
    """
    server = 'http://data.neonscience.org/api/v0/'
    data_url = f'{server}data/{product_code}/{site_code}/{year_month}'

    # Make the API request
    response = requests.get(data_url)
    if response.status_code == 200:
        print(f"Data retrieved successfully for {year_month}!")
        data_json = response.json()
        
        # Initialize a flag to check if the file was found
        file_found = False
        
        # Iterate through files in the JSON response to find the specific flight line
        for file_info in data_json['data']['files']:
            file_name = file_info['name']
            if flight_line in file_name:
                print(f"Downloading {file_name} from {file_info['url']}")
                os.system(f'wget --no-check-certificate "{file_info["url"]}" -O "{file_name}"')
                file_found = True
                break
        
        if not file_found:
            print(f"Flight line {flight_line} not found in the data for {year_month}.")
    else:
        print(f"Failed to retrieve data for {year_month}. Status code: {response.status_code}, Response: {response.text}")

def download_neon_flight_lines(site_code, product_code, year_month, flight_lines):
    """
    Downloads NEON flight line files for given parameters.

    Parameters:
    - site_code (str): The site code.
    - product_code (str): The product code.
    - year_month (str): The year and month in 'YYYY-MM' format.
    - flight_lines (str or list): Single or list of flight line identifiers.
    """
    if isinstance(flight_lines, str):
        flight_lines = [flight_lines]
    
    for flight_line in flight_lines:
        print(f"Processing flight line: {flight_line}")
        download_neon_file(site_code, product_code, year_month, flight_line)
        print("Download completed.\n")

def process_hdf5_with_neon2envi(image_path, site_code):
    """
    Processes an HDF5 file using the neon2envi2.py script.

    Parameters:
    - image_path (str): Path to the HDF5 image file.
    - site_code (str): The site code.
    """
    command = [
        "/opt/conda/envs/macrosystems/bin/python", "neon2envi2.py",
        "--output_dir", "output/",
        "--site_code", site_code,
        "-anc",
        image_path
    ]

    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                print(line, end='')  # Print each line of output in real-time
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

def get_spectral_data_and_wavelengths(filename, row_step, col_step):
    """
    Retrieves spectral data and wavelengths from a specified file using HyTools library.

    Parameters:
    - filename (str): Path to the file to be read.
    - row_step (int): Step size to sample rows by.
    - col_step (int): Step size to sample columns by.

    Returns:
    - tuple: (original np.ndarray, wavelengths np.ndarray)
    """
    envi = ht.HyTools()
    envi.read_file(filename, 'envi')
    
    colrange = np.arange(0, envi.columns).tolist()
    pixel_lines = np.arange(0, envi.lines).tolist()
    rowrange = sorted(random.sample(pixel_lines, envi.columns))
    
    # Retrieve the pixels' spectral data
    original = envi.get_pixels(rowrange, colrange)
    wavelengths = envi.wavelengths
    
    return original, wavelengths

def load_spectra(filenames, row_step=6, col_step=1):
    """
    Loads spectral data from multiple files.

    Parameters:
    - filenames (list): List of file paths.
    - row_step (int): Step size for rows.
    - col_step (int): Step size for columns.

    Returns:
    - dict: Dictionary with filenames as keys and spectral data as values.
    """
    results = {}
    for filename in filenames:
        try:
            spectral_data, wavelengths = get_spectral_data_and_wavelengths(filename, row_step, col_step)
            results[filename] = {"spectral_data": spectral_data, "wavelengths": wavelengths}
        except TypeError:
            print(f"Error processing file: {filename}")
    return results

def extract_overlapping_layers_to_2d_dataframe(raster_path, gpkg_path):
    """
    Extracts overlapping raster layers into a 2D DataFrame based on polygon geometries.

    Parameters:
    - raster_path (str): Path to the raster file.
    - gpkg_path (str): Path to the GeoPackage file containing polygons.

    Returns:
    - pd.DataFrame: DataFrame containing mean raster values for each polygon and layer.
    """
    polygons = gpd.read_file(gpkg_path)
    data = []

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if polygons.crs != raster_crs:
            polygons = polygons.to_crs(raster_crs)

        raster_bounds = src.bounds
        polygons['intersects'] = polygons.geometry.apply(lambda geom: geom.intersects(box(*raster_bounds)))
        overlapping_polygons = polygons[polygons['intersects']].copy()

        for index, polygon in overlapping_polygons.iterrows():
            mask_result, _ = mask(src, [polygon.geometry], crop=True, all_touched=True)
            row = {'polygon_id': index}
            for layer in range(mask_result.shape[0]):
                valid_values = mask_result[layer][mask_result[layer] != src.nodata]
                layer_mean = valid_values.mean() if valid_values.size > 0 else np.nan
                row[f'layer_{layer+1}'] = layer_mean
            data.append(row)

    results_df = pd.DataFrame(data)
    return results_df

def rasterize_polygons_to_match_envi(gpkg_path, existing_raster_path, output_raster_path, attribute=None):
    """
    Rasterizes polygons to match an existing ENVI raster's spatial properties.

    Parameters:
    - gpkg_path (str): Path to the GeoPackage file containing polygons.
    - existing_raster_path (str): Path to the existing ENVI raster.
    - output_raster_path (str): Path to save the rasterized polygons.
    - attribute (str, optional): Attribute to use for rasterization values.

    Returns:
    - None
    """
    polygons = gpd.read_file(gpkg_path)

    with rasterio.open(existing_raster_path) as existing_raster:
        existing_meta = existing_raster.meta
        existing_crs = existing_raster.crs

    # Plot existing raster and polygons
    fig, axs = plt.subplots(1, 3, figsize=(21, 40))
    with rasterio.open(existing_raster_path) as existing_raster:
        show(existing_raster, ax=axs[0], title="Existing Raster")

    if polygons.crs != existing_crs:
        polygons = polygons.to_crs(existing_crs)
    polygons.plot(ax=axs[1], color='red', edgecolor='black')
    axs[1].set_title("Polygons Layer")

    # Rasterize polygons
    rasterized_polygons = rasterize(
        shapes=((geom, value) for geom, value in zip(
            polygons.geometry, 
            polygons[attribute] if attribute and attribute in polygons.columns else polygons.index
        )),
        out_shape=(existing_meta['height'], existing_meta['width']),
        fill=0,
        transform=existing_meta['transform'],
        all_touched=True,
        dtype=existing_meta['dtype']
    )

    # Save rasterized polygons
    with rasterio.open(output_raster_path, 'w', **existing_meta) as out_raster:
        out_raster.write(rasterized_polygons, 1)

    # Plot the rasterized layer
    with rasterio.open(output_raster_path) as new_raster:
        show(new_raster, ax=axs[2], title="Rasterized Polygons Layer")

    plt.tight_layout()
    plt.show()

    print(f"Rasterization complete. Output saved to {output_raster_path}")

# ============================
# Data Processing Functions
# ============================

def jefe(base_folder, site_code, product_code, year_month, flight_lines):
    """
    Orchestrates the processing of spectral data.

    Steps:
    1. Generates necessary data and structures.
    2. Processes all subdirectories within the base_folder.

    Parameters:
    - base_folder (str): Base directory for operations.
    - site_code (str): Site code.
    - product_code (str): Product code.
    - year_month (str): Year and month in 'YYYY-MM' format.
    - flight_lines (list): List of flight lines.
    """
    go_forth_and_multiply(
        base_folder=base_folder,
        site_code=site_code,
        product_code=product_code,
        year_month=year_month,
        flight_lines=flight_lines
    )
    
    process_all_subdirectories(base_folder)

def process_all_subdirectories(parent_directory):
    """
    Processes all subdirectories within the given parent directory.

    Parameters:
    - parent_directory (str): Parent directory containing subdirectories to process.
    """
    for item in os.listdir(parent_directory):
        full_path = os.path.join(parent_directory, item)
        if os.path.isdir(full_path):
            control_function(full_path)
            print(f"Finished processing for directory: {full_path}")
        else:
            print(f"Skipping non-directory item: {full_path}")

def find_raster_files(directory):
    """
    Finds raster files in the specified directory based on specific patterns.

    Excludes files ending with '.hdr'.

    Parameters:
    - directory (str): Directory to search in.

    Returns:
    - list: Sorted list of unique matching raster file paths.
    """
    pattern = "*"
    full_pattern = os.path.join(directory, pattern)
    all_files = glob.glob(full_pattern)
    
    filtered_files = [
        file for file in all_files
        if (
            file.endswith('_envi.img') or
            (file.endswith('_envi') and not file.endswith('.hdr')) or
            file.endswith('_reflectance.img') or
            (file.endswith('_reflectance') and not file.endswith('.hdr')) or
            "_envi_resample_Landsat" in file
        ) and not file.endswith('.hdr')
    ]
    
    found_files = sorted(set(filtered_files))
    return found_files

def control_function(directory):
    """
    Controls the processing of raster files within a directory.

    Steps:
    1. Finds raster files.
    2. Loads and combines raster data.
    3. Processes and flattens the data into a DataFrame.
    4. Cleans the data and writes to a CSV file.

    Parameters:
    - directory (str): Directory to process raster files.
    """
    raster_paths = find_raster_files(directory)
    
    if not raster_paths:
        print("No matching raster files found.")
        return

    # Placeholder for actual raster processing functions
    combined_array = load_and_combine_rasters(raster_paths)
    df_processed = process_and_flatten_array(combined_array)
    
    folder_name = os.path.basename(os.path.normpath(directory))
    output_csv_name = f"{folder_name}_spectral_data_all_sensors.csv"
    output_csv_path = os.path.join(directory, output_csv_name)
    
    if os.path.exists(output_csv_path):
        print(f"CSV already exists: {output_csv_path}. Skipping processing.")
        return
        
    clean_data_and_write_to_csv(df_processed, output_csv_path)

    print(f"Processed and cleaned data saved to {output_csv_path}")

def clean_data_and_write_to_csv(df, output_csv_path, chunk_size=100000):
    """
    Cleans the DataFrame and writes it to a CSV file in chunks to manage memory usage.

    Replaces values approximately equal to -9999 with NaN in non-'Pixel' columns and drops rows where all such columns are NaN.

    Parameters:
    - df (pd.DataFrame): DataFrame to clean.
    - output_csv_path (str): Path to save the cleaned CSV.
    - chunk_size (int): Number of rows per chunk.
    """
    total_rows = df.shape[0]
    num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

    print(f"Starting cleaning process, total rows: {total_rows}, chunk size: {chunk_size}, total chunks: {num_chunks}")

    first_chunk = True

    for i, start_row in enumerate(range(0, total_rows, chunk_size)):
        chunk = df.iloc[start_row:start_row + chunk_size].copy()

        # Replace values close to -9999 with NaN
        for col in chunk.columns:
            if not col.startswith('Pixel'):
                chunk[col] = np.where(np.isclose(chunk[col], -9999, atol=1), np.nan, chunk[col])

        # Drop rows where all non-'Pixel' columns are NaN
        non_pixel_columns = [col for col in chunk.columns if not col.startswith('Pixel')]
        chunk.dropna(subset=non_pixel_columns, how='all', inplace=True)
        
        # Write processed chunk to CSV
        if first_chunk:
            chunk.to_csv(output_csv_path, mode='w', header=True, index=False)
            first_chunk = False
        else:
            chunk.to_csv(output_csv_path, mode='a', header=False, index=False)
        
        print(f"Processed and wrote chunk {i+1}/{num_chunks} to CSV.")

    print("Cleaning process completed and data written to CSV.")

def process_and_flatten_array(array, landsat_versions=[5, 7, 8, 9], bands_per_landsat=6):
    """
    Processes a 3D numpy array into a flattened DataFrame with renamed columns and Pixel_id.

    Parameters:
    - array (np.ndarray): 3D array of shape (bands, rows, cols).
    - landsat_versions (list): List of Landsat versions for naming.
    - bands_per_landsat (int): Number of bands per Landsat version.

    Returns:
    - pd.DataFrame: Flattened and processed DataFrame.
    """
    if len(array.shape) != 3:
        raise ValueError("Input array must be 3-dimensional.")
    
    bands, rows, cols = array.shape
    reshaped_array = array.reshape(bands, -1).T  # Transpose to make bands as columns
    pixel_indices = np.indices((rows, cols)).reshape(2, -1).T  # Row and col indices
    
    df = pd.DataFrame(reshaped_array, columns=[f'Band_{i+1}' for i in range(bands)])
    df.insert(0, 'Pixel_Col', pixel_indices[:, 1])
    df.insert(0, 'Pixel_Row', pixel_indices[:, 0])
    df.insert(0, 'Pixel_id', np.arange(len(df)))

    # Renaming columns
    total_bands = bands
    original_and_corrected_bands = total_bands - bands_per_landsat * len(landsat_versions)
    band_per_version = original_and_corrected_bands // 2  # Assuming equal original and corrected bands
    
    new_names = ([f"Original_band_{i}" for i in range(1, band_per_version + 1)] +
                [f"Corrected_band_{i}" for i in range(1, band_per_version + 1)])
    
    for version in landsat_versions:
        new_names.extend([f"Landsat_{version}_band_{i}" for i in range(1, bands_per_landsat + 1)])
    
    df.columns = ['Pixel_id', 'Pixel_Row', 'Pixel_Col'] + new_names

    return df

# ============================
# Correction and Configuration
# ============================

def go_forth_and_multiply(base_folder="output", **kwargs):
    """
    Executes the main processing steps:
    1. Downloads NEON flight lines.
    2. Converts flight lines to ENVI format.
    3. Generates configuration JSON files.
    4. Applies topographic and BRDF corrections.
    5. Resamples and translates data to other sensor formats.

    Parameters:
    - base_folder (str): Directory for output.
    - **kwargs: Additional keyword arguments for downloading flight lines.
    """
    os.makedirs(base_folder, exist_ok=True)
    
    # Step 1: Download NEON flight lines
    download_neon_flight_lines(**kwargs)

    # Step 2: Convert flight lines to ENVI format
    flight_lines_to_envi(output_dir=base_folder)

    # Step 3: Generate configuration JSON
    generate_config_json(base_folder)

    # Step 4: Apply topographic and BRDF corrections
    apply_topo_and_brdf_corrections(base_folder)

    # Step 5: Resample and translate data to other sensor formats
    resample_translation_to_other_sensors(base_folder)

    print("Processing complete.")

def resample_translation_to_other_sensors(base_folder, conda_env_path='/opt/conda/envs/macrosystems/bin/python'):
    """
    Resamples and translates data to other sensor formats for each subdirectory.

    Parameters:
    - base_folder (str): Base directory containing subdirectories to process.
    - conda_env_path (str): Path to the Conda environment's Python executable.
    """
    subdirectories = [os.path.join(base_folder, d) for d in os.listdir(base_folder) 
                      if os.path.isdir(os.path.join(base_folder, d))]

    for folder in subdirectories:
        print(f"Processing folder: {folder}")
        translate_to_other_sensors(folder, conda_env_path)
    print("Resampling completed.")

def translate_to_other_sensors(folder_path, conda_env_path='/opt/conda/envs/macrosystems/bin/python'):
    """
    Translates raster files to various sensor formats using the resampling_demo.py script.

    Parameters:
    - folder_path (str): Path to the folder containing raster files.
    - conda_env_path (str): Path to the Conda environment's Python executable.
    """
    sensor_types = [
        'Landsat 5 TM',
        'Landsat 7 ETM+',
        'Landsat 8 OLI',
        'Landsat 9 OLI-2',
        'MicaSense',
        'MicaSense-to-match TM and ETM+',
        'MicaSense-to-match OLI and OLI-2'
    ]

    pattern = os.path.join(folder_path, '*_envi')
    envi_files = [file for file in glob.glob(pattern) 
                  if not file.endswith('config_envi') and not file.endswith('.json')]
    
    if len(envi_files) != 1:
        print(f"Error: Expected to find exactly one file with '_envi' but found {len(envi_files)}: {envi_files}")
        return

    resampling_file_path = envi_files[0]
    json_file = os.path.join('Resampling', 'landsat_band_parameters.json')

    for sensor_type in sensor_types:
        hdr_path = f"{resampling_file_path}.hdr"
        output_path = os.path.join(folder_path, 
                                   f"{os.path.basename(resampling_file_path)}_resample_{sensor_type.replace(' ', '_').replace('+', 'plus')}.hdr")

        command = [
            conda_env_path, 'Resampling/resampling_demo.py',
            '--resampling_file_path', resampling_file_path,
            '--json_file', json_file,
            '--hdr_path', hdr_path,
            '--sensor_type', sensor_type,
            '--output_path', output_path
        ]

        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if process.returncode != 0:
            print(f"Error executing command: {' '.join(command)}")
            print(f"Standard Output: {process.stdout}")
            print(f"Error Output: {process.stderr}")
        else:
            print(f"Command executed successfully for sensor type: {sensor_type}")
            print(f"Standard Output: {process.stdout}")

def apply_topo_and_brdf_corrections(base_folder_path, conda_env_path='/opt/conda/envs/macrosystems'):
    """
    Applies topographic and BRDF corrections to raster files using the image_correct.py script.

    Parameters:
    - base_folder_path (str): Path to the base folder containing subdirectories to process.
    - conda_env_path (str): Path to the Conda environment.
    """
    python_executable = os.path.join(conda_env_path, "bin", "python")
    subfolders = glob.glob(os.path.join(base_folder_path, '*/'))

    for folder in subfolders:
        folder_name = os.path.basename(os.path.normpath(folder))
        json_file_name = f"{folder_name}_config__envi.json"
        json_file_path = os.path.join(folder, json_file_name)
        
        if os.path.isfile(json_file_path):
            command = f"{python_executable} image_correct.py {json_file_path}"
            process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"Processed {json_file_path}")
            if process.returncode != 0:
                print(f"Error executing command: {command}")
                print(f"Standard Output: {process.stdout}")
                print(f"Error Output: {process.stderr}")
            else:
                print("Command executed successfully")
                print(f"Standard Output: {process.stdout}")
        else:
            print(f"JSON file not found: {json_file_path}")

    print("All corrections applied.")

def generate_config_json(parent_directory):
    """
    Generates configuration JSON files for all subdirectories within the parent directory.

    Parameters:
    - parent_directory (str): Directory containing subdirectories to process.
    """
    exclude_dirs = ['.ipynb_checkpoints']
    subdirectories = [
        os.path.join(parent_directory, d) for d in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, d)) and d not in exclude_dirs
    ]
    
    for directory in subdirectories:
        print(f"Generating configuration files for directory: {directory}")
        generate_correction_configs_for_directory(directory)
        print("Configuration files generation completed.\n")

def generate_correction_configs_for_directory(directory):
    """
    Generates correction configuration files for a specific directory.

    Parameters:
    - directory (str): Directory containing the main image and ancillary files.
    """
    bad_bands = [[300, 400], [1337, 1430], [1800, 1960], [2450, 2600]]
    file_type = 'envi'

    main_image_name = os.path.basename(directory)
    main_image_file = os.path.join(directory, main_image_name)  # Assuming the main image file has .h5 extension

    anc_files_pattern = os.path.join(directory, "*_ancillary*")
    anc_files = sorted(glob.glob(anc_files_pattern))

    aviris_anc_names = ['path_length', 'sensor_az', 'sensor_zn', 'solar_az', 'solar_zn', 'slope', 'aspect', 'phase', 'cosine_i']
    suffix_labels = ["envi", "anc"]  # Define suffixes for different types of files

    for i, anc_file in enumerate(anc_files):
        if i == 0:
            suffix_label = f"_envi"
        elif i == 1:
            suffix_label = f"_anc"
        else:
            suffix_label = f"_{i}"  # Fallback for unexpected 'i' values

        config_dict = {
            'bad_bands': bad_bands,
            'file_type': file_type,
            "input_files": [main_image_file],
            "anc_files": {
                main_image_file: dict(zip(aviris_anc_names, [[anc_file, a] for a in range(len(aviris_anc_names))]))
            },
            'export': {
                'coeffs': True,
                'image': True,
                'masks': True,
                'subset_waves': [],
                'output_dir': directory,
                "suffix": suffix_label
            },
            "corrections": ['topo','brdf'],
            "topo": {
                'type': 'scs+c',
                'calc_mask': [
                    ["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}],
                    ['ancillary', {'name': 'slope', 'min': np.radians(5), 'max': '+inf'}],
                    ['ancillary', {'name': 'cosine_i', 'min': 0.12, 'max': '+inf'}],
                    ['cloud', {'method': 'zhai_2018', 'cloud': True, 'shadow': True, 
                              'T1': 0.01, 't2': 1/10, 't3': 1/4, 't4': 1/2, 'T7': 9, 'T8': 9}]
                ],
                'apply_mask': [
                    ["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}],
                    ['ancillary', {'name': 'slope', 'min': np.radians(5), 'max': '+inf'}],
                    ['ancillary', {'name': 'cosine_i', 'min': 0.12, 'max': '+inf'}]
                ],
                'c_fit_type': 'nnls'
            },
            "brdf": {
                'solar_zn_type': 'scene',
                'type': 'flex',
                'grouped': True,
                'sample_perc': 0.1,
                'geometric': 'li_dense_r',
                'volume': 'ross_thick',
                "b/r": 10,  
                "h/b": 2,
                'interp_kind': 'linear',
                'calc_mask': [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]],
                'apply_mask': [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]],
                'diagnostic_plots': True,
                'diagnostic_waves': [448, 849, 1660, 2201],
                'bin_type': 'dynamic',
                'num_bins': 25,
                'ndvi_bin_min': 0.05,
                'ndvi_bin_max': 1.0,
                'ndvi_perc_min': 10,
                'ndvi_perc_max': 95
            },
            'num_cpus': 1,
            "resample": False,
            "resampler": {
                'type': 'cubic',
                'out_waves': [],
                'out_fwhm': []
            }
        }

        # Remove bad bands from output waves
        for wavelength in range(450, 660, 100):
            bad = any((wavelength >= start) and (wavelength <= end) for start, end in config_dict['bad_bands'])
            if not bad:
                config_dict["resampler"]['out_waves'].append(wavelength)

        # Construct the filename for the configuration JSON
        config_filename = f"{main_image_name}_config_{suffix_label}.json"
        config_file_path = os.path.join(directory, config_filename)

        # Save the configuration to a JSON file
        with open(config_file_path, 'w') as outfile:
            json.dump(config_dict, outfile, indent=3)
        
        print(f"Configuration saved to {config_file_path}")

def generate_correction_configs(main_image_file, output_dir):
    """
    Generates correction configuration files for a main image and its ancillary files.

    Parameters:
    - main_image_file (str): Path to the main image file.
    - output_dir (str): Directory to save the configuration files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(main_image_file).split('.')[0]
    input_dir = os.path.dirname(main_image_file)
    
    bad_bands = [[300, 400], [1337, 1430], [1800, 1960], [2450, 2600]]
    file_type = 'envi'
    aviris_anc_names = ['path_length', 'sensor_az', 'sensor_zn', 'solar_az', 'solar_zn', 'phase', 'slope', 'aspect', 'cosine_i']

    anc_files_pattern = os.path.join(input_dir, f"{base_name}_ancillary*")
    anc_files = sorted(glob.glob(anc_files_pattern))

    files_to_process = [main_image_file] + anc_files

    for file_path in files_to_process:
        is_ancillary = '_ancillary' in file_path
        config_type = 'ancillary' if is_ancillary else ''
        suffix_label = "_anc" if config_type == 'ancillary' else "_envi"

        config_dict = {
            'bad_bands': bad_bands,
            'file_type': file_type,
            "input_files": [main_image_file],
            "export": {
                "coeffs": True,
                "image": True,
                "masks": True,
                "subset_waves": [],
                "output_dir": output_dir,
                "suffix": f"_corrected_{config_type}"
            },
            "corrections": ['topo','brdf'],
            "topo": {
                'type': 'scs+c',
                'calc_mask': [
                    ["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}],
                    ['ancillary', {'name': 'slope', 'min': np.radians(5), 'max': '+inf'}],
                    ['ancillary', {'name': 'cosine_i', 'min': 0.12, 'max': '+inf'}],
                    ['cloud', {'method': 'zhai_2018', 'cloud': True, 'shadow': True, 
                              'T1': 0.01, 't2': 1/10, 't3': 1/4, 't4': 1/2, 'T7': 9, 'T8': 9}]
                ],
                'apply_mask': [
                    ["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}],
                    ['ancillary', {'name': 'slope', 'min': np.radians(5), 'max': '+inf'}],
                    ['ancillary', {'name': 'cosine_i', 'min': 0.12, 'max': '+inf'}]
                ],
                'c_fit_type': 'nnls'
            },
            "brdf": {
                'solar_zn_type': 'scene',
                'type': 'flex',
                'grouped': True,
                'sample_perc': 0.1,
                'geometric': 'li_dense_r',
                'volume': 'ross_thick',
                "b/r": 10,  
                "h/b": 2,
                'interp_kind': 'linear',
                'calc_mask': [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]],
                'apply_mask': [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]],
                'diagnostic_plots': True,
                'diagnostic_waves': [448, 849, 1660, 2201],
                'bin_type': 'dynamic',
                'num_bins': 25,
                'ndvi_bin_min': 0.05,
                'ndvi_bin_max': 1.0,
                'ndvi_perc_min': 10,
                'ndvi_perc_max': 95
            },
            'num_cpus': 1,
            "resample": False,
            "resampler": {
                'type': 'cubic',
                'out_waves': [],
                'out_fwhm': []
            }
        }

        # Remove bad bands from output waves
        for wavelength in range(450, 660, 100):
            bad = any((wavelength >= start) and (wavelength <= end) for start, end in config_dict['bad_bands'])
            if not bad:
                config_dict["resampler"]['out_waves'].append(wavelength)

        # Construct the filename for the configuration JSON
        config_filename = f"{base_name}_config_{suffix_label}.json"
        config_file_path = os.path.join(output_dir, config_filename)

        # Save the configuration to a JSON file
        with open(config_file_path, 'w') as outfile:
            json.dump(config_dict, outfile, indent=3)
        
        print(f"Configuration saved to {config_file_path}")

# ============================
# Plotting Functions
# ============================

def prepare_spectral_data(spectral_data, wavelengths):
    """
    Prepares spectral data by merging with wavelength information.

    Parameters:
    - spectral_data (np.ndarray): 2D array where each row corresponds to a pixel's spectral data.
    - wavelengths (np.ndarray): Array of wavelengths corresponding to each spectral band.

    Returns:
    - pd.DataFrame: Merged DataFrame with spectral data and wavelengths.
    """
    long_df = pd.melt(pd.DataFrame(spectral_data).transpose(), var_name="band", value_name="reflectance")
    
    waves = pd.DataFrame(wavelengths, columns=["wavelength_nm"])
    waves['band'] = range(len(waves))
    
    merged_data = pd.merge(long_df, waves, on='band')
    merged_data["wavelength_nm"] = pd.to_numeric(merged_data["wavelength_nm"])
    
    return merged_data

def reshape_spectra(results, index):
    """
    Reshapes spectral data for a specific sensor based on index.

    Parameters:
    - results (dict): Dictionary containing spectral data and wavelengths.
    - index (int): Index of the sensor to reshape.

    Returns:
    - pd.DataFrame or None: Reshaped DataFrame or None if index is out of range.
    """
    keys = list(results.keys())
    if index < 0 or index >= len(keys):
        print("Index out of range")
        return None

    first_key = keys[index]
    spectral_data = results[first_key]['spectral_data']
    wavelengths = results[first_key]['wavelengths']

    if index < 4:
        df_spectral_data = pd.DataFrame(spectral_data, columns=wavelengths.astype(str))
        long_df = pd.melt(df_spectral_data, var_name="wavelength_nm", value_name="reflectance")
        waves = pd.DataFrame(wavelengths, columns=["wavelength_nm"])
        waves['band'] = range(len(waves))
        long_df["wavelength_nm"] = pd.to_numeric(long_df["wavelength_nm"])
        merged_data = pd.merge(long_df, waves, on='wavelength_nm')
        
        # Add label columns
        first_key = keys[index].replace("export/resample_", "").replace(".img", "")
        merged_data['sensor'] = first_key
        merged_data = merged_data.reindex(columns=['sensor', 'band', 'wavelength_nm', 'reflectance','pixel'])
        
        length = len(merged_data)
        sequence = np.arange(0, 1071)
        repeated_sequence = np.resize(sequence, length)
        merged_data['pixel'] = repeated_sequence
        merged_data['sensor_band'] = merged_data['sensor'].astype(str) + '_' + merged_data['band'].astype(str)
        return merged_data
    else:
        merged_data = prepare_spectral_data(spectral_data, wavelengths)
        first_key = keys[index].replace("export/ENVI__corrected_0", "hyperspectral_corrected")
        first_key = first_key.replace("output/ENVI", "hyperspectral_original")
        merged_data['sensor'] = first_key
        merged_data = merged_data.reindex(columns=['sensor', 'band', 'wavelength_nm', 'reflectance','pixel'])
        
        length = len(merged_data)
        sequence = np.arange(0, 1071)
        repeated_sequence = np.resize(sequence, length)
        merged_data['pixel'] = repeated_sequence
        merged_data['sensor_band'] = merged_data['sensor'].astype(str) + '_' + merged_data['band'].astype(str)
        return merged_data

def concatenate_sensors(reshape_spectra_function, spectra, sensors_range):
    """
    Concatenates reshaped spectra from multiple sensors into a single DataFrame.

    Parameters:
    - reshape_spectra_function (function): Function to reshape individual sensor spectra.
    - spectra (dict): Dictionary containing spectral data and wavelengths.
    - sensors_range (range): Range of sensor indices to process.

    Returns:
    - pd.DataFrame: Concatenated DataFrame of all sensors.
    """
    all_spectra = []
    for sensor in sensors_range:
        reshaped_spectra = reshape_spectra_function(spectra, sensor)
        if reshaped_spectra is not None:
            all_spectra.append(reshaped_spectra)
    
    concatenated_spectra = pd.concat(all_spectra, ignore_index=True)
    return concatenated_spectra

def plot_spectral_data(df, highlight_pixel):
    """
    Plots spectral data with a highlighted pixel.

    Parameters:
    - df (pd.DataFrame): DataFrame containing spectral data.
    - highlight_pixel (int): Pixel ID to highlight.
    """
    df = df[df['wavelength_nm'] > 0]
    df['reflectance'] = df['reflectance'].replace(-9999, np.nan)
    unique_indices = df['pixel'].unique()

    for idx in unique_indices:
        subset = df[df['pixel'] == idx]
        plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.05, color="blue")

    highlighted_subset = df[df['pixel'] == highlight_pixel]

    if (highlighted_subset['reflectance'] == -9999).all() or highlighted_subset['reflectance'].isna().all():
        print(f"Warning: Pixel {highlight_pixel} data is entirely -9999 or NaN.")

    plt.plot(highlighted_subset['wavelength_nm'], highlighted_subset['reflectance'], color='red', linewidth=10, label=f'Pixel {highlight_pixel}')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 10000)
    plt.show()

def plot_each_sensor_with_highlight(concatenated_sensors, highlight_pixel, save_path=None):
    """
    Plots spectral data for each sensor with a highlighted pixel.

    Parameters:
    - concatenated_sensors (pd.DataFrame): DataFrame containing concatenated sensor data.
    - highlight_pixel (int): Pixel ID to highlight.
    - save_path (str, optional): Path to save the plot.
    """
    sensors = concatenated_sensors['sensor'].unique()
    fig, axs = plt.subplots(len(sensors), 1, figsize=(10, 5 * len(sensors)))
    
    for i, sensor in enumerate(sensors):
        df = concatenated_sensors[concatenated_sensors['sensor'] == sensor]
        pixels = df['pixel'].unique()
        
        for pixel in pixels:
            subset = df[df['pixel'] == pixel]
            axs[i].plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.05, color="blue")
        
        highlighted_subset = df[df['pixel'] == highlight_pixel]
        if not highlighted_subset.empty and not highlighted_subset['reflectance'].isna().all():
            axs[i].plot(highlighted_subset['wavelength_nm'], highlighted_subset['reflectance'], color='red', linewidth=2, label=f'Pixel {highlight_pixel}')
        
        axs[i].set_title(sensor)
        axs[i].set_xlabel('Wavelength (nm)')
        axs[i].set_ylabel('Reflectance')
        axs[i].set_ylim(0, 10000)
        axs[i].set_xlim(350, 2550)
        axs[i].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1])
    plt.show()

def plot_with_highlighted_sensors(concatenated_sensors, highlight_pixels, save_path=None):
    """
    Plots spectral data with highlighted pixels across all sensors.

    Parameters:
    - concatenated_sensors (pd.DataFrame): DataFrame containing concatenated sensor data.
    - highlight_pixels (int or list): Pixel ID(s) to highlight.
    - save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    
    # Ensure highlight_pixels is a list
    if not isinstance(highlight_pixels, list):
        highlight_pixels = [highlight_pixels]
    
    # Initial data cleaning
    concatenated_sensors = concatenated_sensors[concatenated_sensors['wavelength_nm'] > 0]
    concatenated_sensors['reflectance'] = concatenated_sensors['reflectance'].replace(-9999, np.nan)
    
    # Plot hyperspectral corrected data in blue
    hyperspectral_corrected = concatenated_sensors[concatenated_sensors['sensor'] == 'hyperspectral_corrected']
    for pixel in hyperspectral_corrected['pixel'].unique():
        subset = hyperspectral_corrected[hyperspectral_corrected['pixel'] == pixel]
        plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.05, color="blue")
    
    # Overlay highlighted lines from each sensor in red
    for sensor in concatenated_sensors['sensor'].unique():
        if sensor != 'hyperspectral_corrected':
            for highlight_pixel in highlight_pixels:
                highlighted_subset = concatenated_sensors[
                    (concatenated_sensors['pixel'] == highlight_pixel) & 
                    (concatenated_sensors['sensor'] == sensor)
                ]
                if not highlighted_subset.empty:
                    plt.plot(highlighted_subset['wavelength_nm'], highlighted_subset['reflectance'], color='red', linewidth=2, label=sensor)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.ylim(0, None)
    plt.xlim(350, 2550)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1])
    plt.show()

def plot_with_highlighted_sensors_modified(concatenated_sensors, highlight_pixels, save_path=None):
    """
    Modified function to plot spectral data with highlighted sensors, specifically for boosted quantile predictions.

    Parameters:
    - concatenated_sensors (pd.DataFrame): DataFrame containing concatenated sensor data.
    - highlight_pixels (int or list): Pixel ID(s) to highlight.
    - save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    
    if not isinstance(highlight_pixels, list):
        highlight_pixels = [highlight_pixels]
    
    concatenated_sensors = concatenated_sensors[concatenated_sensors['wavelength_nm'] > 0]
    concatenated_sensors['reflectance'] = concatenated_sensors['reflectance'].replace(-9999, np.nan)
    
    # Plot hyperspectral corrected data in blue
    hyperspectral_corrected = concatenated_sensors[concatenated_sensors['sensor'] == 'hyperspectral_corrected']
    for pixel in hyperspectral_corrected['pixel'].unique():
        subset = hyperspectral_corrected[hyperspectral_corrected['pixel'] == pixel]
        plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.05, color="blue")
    
    # Overlay boosted quantile predictions from each sensor in red
    for sensor in concatenated_sensors['sensor'].unique():
        if sensor != 'hyperspectral_corrected':
            for highlight_pixel in highlight_pixels:
                highlighted_subset = concatenated_sensors[
                    (concatenated_sensors['pixel'] == highlight_pixel) & 
                    (concatenated_sensors['sensor'] == sensor)
                ]
                if not highlighted_subset.empty:
                    plt.plot(highlighted_subset['wavelength_nm'], highlighted_subset['0.50'], linewidth=10, label=sensor)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.ylim(0, 10000)
    plt.xlim(350, 2550)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1])
    plt.show()

def plot_each_sensor_with_highlight(concatenated_sensors, highlight_pixel, save_path=None):
    """
    Plots spectral data for each sensor with a highlighted pixel.

    Parameters:
    - concatenated_sensors (pd.DataFrame): DataFrame containing concatenated sensor data.
    - highlight_pixel (int): Pixel ID to highlight.
    - save_path (str, optional): Path to save the plot.
    """
    sensors = concatenated_sensors['sensor'].unique()
    fig, axs = plt.subplots(len(sensors), 1, figsize=(10, 5 * len(sensors)))
    
    for i, sensor in enumerate(sensors):
        df = concatenated_sensors[concatenated_sensors['sensor'] == sensor]
        pixels = df['pixel'].unique()
        
        for pixel in pixels:
            subset = df[df['pixel'] == pixel]
            axs[i].plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.05, color="blue")
        
        highlighted_subset = df[df['pixel'] == highlight_pixel]
        if not highlighted_subset.empty and not highlighted_subset['reflectance'].isna().all():
            axs[i].plot(highlighted_subset['wavelength_nm'], highlighted_subset['reflectance'], color='red', linewidth=2, label=f'Pixel {highlight_pixel}')
        
        axs[i].set_title(sensor)
        axs[i].set_xlabel('Wavelength (nm)')
        axs[i].set_ylabel('Reflectance')
        axs[i].set_ylim(0, 10000)
        axs[i].set_xlim(350, 2550)
        axs[i].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1])
    plt.show()

def plot_with_highlighted_sensors(concatenated_sensors, highlight_pixels, save_path=None):
    """
    Plots spectral data with highlighted pixels across all sensors.

    Parameters:
    - concatenated_sensors (pd.DataFrame): DataFrame containing concatenated sensor data.
    - highlight_pixels (int or list): Pixel ID(s) to highlight.
    - save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    
    # Ensure highlight_pixels is a list
    if not isinstance(highlight_pixels, list):
        highlight_pixels = [highlight_pixels]
    
    # Initial data cleaning
    concatenated_sensors = concatenated_sensors[concatenated_sensors['wavelength_nm'] > 0]
    concatenated_sensors['reflectance'] = concatenated_sensors['reflectance'].replace(-9999, np.nan)
    
    # Plot hyperspectral corrected data in blue
    hyperspectral_corrected = concatenated_sensors[concatenated_sensors['sensor'] == 'hyperspectral_corrected']
    for pixel in hyperspectral_corrected['pixel'].unique():
        subset = hyperspectral_corrected[hyperspectral_corrected['pixel'] == pixel]
        plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.05, color="blue")
    
    # Overlay highlighted lines from each sensor in red
    for sensor in concatenated_sensors['sensor'].unique():
        if sensor != 'hyperspectral_corrected':
            for highlight_pixel in highlight_pixels:
                highlighted_subset = concatenated_sensors[
                    (concatenated_sensors['pixel'] == highlight_pixel) & 
                    (concatenated_sensors['sensor'] == sensor)
                ]
                if not highlighted_subset.empty:
                    plt.plot(highlighted_subset['wavelength_nm'], highlighted_subset['reflectance'], color='red', linewidth=2, label=sensor)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.ylim(0, None)
    plt.xlim(350, 2550)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1])
    plt.show()

def fit_models_with_different_alpha(data, n_levels=100):
    """
    Fits Gradient Boosting Regressor models with different alpha values.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'wavelength_nm' and 'reflectance'.
    - n_levels (int): Number of alpha levels to iterate over.

    Returns:
    - tuple: (list of trained models, DataFrame with predictions)
    """
    data['reflectance'] = data['reflectance'].replace(np.nan, 0)
    
    X = data[['wavelength_nm']]
    y = data['reflectance']
    
    models = []
    alphas = np.linspace(0.01, 0.99, n_levels)
    
    for alpha in alphas:
        model = GradientBoostingRegressor(
            n_estimators=500, 
            max_depth=15, 
            learning_rate=0.09,
            subsample=0.1, 
            loss='quantile', 
            alpha=alpha
        )
        model.fit(X, y)
        models.append(model)
        
        # Store predictions
        data[f'{alpha:.2f}'] = model.predict(X)
    
    return models, data

def boosted_quantile_plot(data, num_lines=10, title='Hyperspectral Corrected Predictions by Alpha', save_path=None):
    """
    Plots boosted quantile predictions alongside hyperspectral corrected data.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'wavelength_nm' and quantile predictions.
    - num_lines (int): Number of quantile lines to plot.
    - title (str): Title of the plot.
    - save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    total_alphas = 100
    step = total_alphas // num_lines

    # Plotting hyperspectral corrected data in blue
    hyperspectral_corrected = data[data['sensor'] == 'hyperspectral_corrected']
    for pixel in hyperspectral_corrected['pixel'].unique():
        subset = hyperspectral_corrected[hyperspectral_corrected['pixel'] == pixel]
        plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.04, color="blue", linewidth=0.3)
    
    alpha_values = np.linspace(0.01, 0.99, total_alphas)[::step]
    for alpha_value in alpha_values:
        alpha_col = f'{alpha_value:.2f}'
        adjusted_alpha = 1 - alpha_value if alpha_value > 0.5 else alpha_value
        plt.plot(data['wavelength_nm'], data[alpha_col], label=f'Probability {alpha_col}', alpha=adjusted_alpha, color="red", linewidth=adjusted_alpha*6)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Predicted Reflectance')
    plt.title(title)
    plt.ylim(0, 10000)
    plt.xlim(350, 2550)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def boosted_quantile_plot_by_sensor(data, num_lines=10, title='Hyperspectral Corrected Predictions by Alpha', save_path=None):
    """
    Plots boosted quantile predictions for each sensor separately.

    Parameters:
    - data (pd.DataFrame): DataFrame containing concatenated sensor data and quantile predictions.
    - num_lines (int): Number of quantile lines to plot per sensor.
    - title (str): Title of the plot.
    - save_path (str, optional): Path to save the plot.
    """
    sensors = data['sensor'].unique()[:6]  # Limit to the first 6 sensors
    total_alphas = 100
    step = total_alphas // num_lines
    num_sensors = len(sensors)
    
    plt.figure(figsize=(10, 40))  # Adjusting the figure size
    
    for i, sensor in enumerate(sensors, 1):
        plt.subplot(num_sensors, 1, i)  # Creating a subplot for each sensor
        sensor_data = data[data['sensor'] == sensor]
        
        # Plotting data specific to each sensor
        for pixel in sensor_data['pixel'].unique():
            subset = sensor_data[sensor_data['pixel'] == pixel]
            plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.04, color="blue", linewidth=0.3)
        
        alpha_values = np.linspace(0.01, 0.99, total_alphas)[::step]
        for alpha_value in alpha_values:
            alpha_col = f'{alpha_value:.2f}'
            adjusted_alpha = 1 - alpha_value if alpha_value > 0.5 else alpha_value
            adjusted_label = f'{adjusted_alpha:.2f}'
            plt.plot(sensor_data['wavelength_nm'], sensor_data[alpha_col], label=f'Probability {adjusted_label}', alpha=adjusted_alpha, color="red", linewidth=adjusted_alpha*6)
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(f'Predicted Reflectance for {sensor}')
        plt.title(f'{title} - {sensor}')
        plt.ylim(0, 10000)
        plt.xlim(350, 2550)
        plt.legend(loc='best', fontsize='small')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def boosted_quantile_plot_combined(data, num_lines=10, title='Hyperspectral Corrected Predictions by Alpha', save_path=None):
    """
    Plots boosted quantile predictions for combined sensors.

    Parameters:
    - data (pd.DataFrame): DataFrame containing concatenated sensor data and quantile predictions.
    - num_lines (int): Number of quantile lines to plot.
    - title (str): Title of the plot.
    - save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")
    
    # Initial data cleaning
    data = data[data['wavelength_nm'] > 0]
    data['reflectance'] = data['reflectance'].replace(-9999, np.nan)
    
    # Plot hyperspectral corrected data in blue
    hyperspectral_corrected = data[data['sensor'] == 'hyperspectral_corrected']
    for pixel in hyperspectral_corrected['pixel'].unique():
        subset = hyperspectral_corrected[hyperspectral_corrected['pixel'] == pixel]
        plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.05, color="blue")
    
    # Overlay boosted quantile predictions from each sensor in red
    for sensor in data['sensor'].unique():
        if sensor != 'hyperspectral_corrected':
            for pixel in data['pixel'].unique():
                highlighted_subset = data[
                    (data['pixel'] == pixel) & 
                    (data['sensor'] == sensor)
                ]
                if not highlighted_subset.empty:
                    plt.plot(highlighted_subset['wavelength_nm'], highlighted_subset['0.50'], linewidth=10, label=sensor)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.ylim(0, 10000)
    plt.xlim(350, 2550)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ============================
# Additional Processing Functions
# ============================

def prepare_spectral_data(spectral_data, wavelengths):
    """
    Prepares spectral data by merging with wavelength information.

    Parameters:
    - spectral_data (np.ndarray): 2D array where each row corresponds to a pixel's spectral data.
    - wavelengths (np.ndarray): Array of wavelengths corresponding to each spectral band.

    Returns:
    - pd.DataFrame: Merged DataFrame with spectral data and wavelengths.
    """
    long_df = pd.melt(pd.DataFrame(spectral_data).transpose(), var_name="band", value_name="reflectance")
    
    waves = pd.DataFrame(wavelengths, columns=["wavelength_nm"])
    waves['band'] = range(len(waves))
    
    merged_data = pd.merge(long_df, waves, on='band')
    merged_data["wavelength_nm"] = pd.to_numeric(merged_data["wavelength_nm"])
    
    return merged_data

# ============================
# Rasterization and Geospatial Functions
# ============================

def show_rgb(file_paths, r=660, g=550, b=440):
    """
    Displays RGB composites of given raster files.

    Parameters:
    - file_paths (list or str): List of file paths or a single file path.
    - r (int): Wavelength for red channel.
    - g (int): Wavelength for green channel.
    - b (int): Wavelength for blue channel.
    """
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    
    n_files = len(file_paths)
    fig_width = 7 * n_files
    fig_height = 100
    
    warning_message = (
        "WARNING: The images displayed are part of a panel and "
        "the flight lines are not presented on the same map. "
        "Spatial relationships between panels may not be accurate."
    )
    print(warning_message)
    
    fig, axs = plt.subplots(1, n_files, figsize=(fig_width, fig_height), squeeze=False)

    for file_path, ax in zip(file_paths, axs.flatten()):
        hy_obj = ht.HyTools()
        hy_obj.read_file(file_path, 'envi')
        
        rgb = np.stack([
            hy_obj.get_wave(r),
            hy_obj.get_wave(g),
            hy_obj.get_wave(b)
        ])
        rgb = np.moveaxis(rgb, 0, -1).astype(float)
        rgb[rgb == hy_obj.no_data] = np.nan
        
        # Apply percentile stretch
        bottom = np.nanpercentile(rgb, 5, axis=(0, 1))
        top = np.nanpercentile(rgb, 95, axis=(0, 1))
        rgb = np.clip(rgb, bottom, top)
        
        # Normalize
        rgb = (rgb - np.nanmin(rgb, axis=(0, 1))) / (np.nanmax(rgb, axis=(0, 1)) - np.nanmin(rgb, axis=(0, 1)))
        
        # Plotting
        ax.imshow(rgb)
        ax.axis('off')
        ax.set_title(f"RGB Composite: {os.path.basename(file_path)}")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

# ============================
# Main Execution Guard
# ============================

if __name__ == "__main__":
    # Example usage:
    # Define parameters
    base_folder = "output"
    site_code = "SITE123"
    product_code = "PRODUCT456"
    year_month = "2023-08"
    flight_lines = ["FLIGHT1", "FLIGHT2"]

    # Run the main processing function
    jefe(base_folder, site_code, product_code, year_month, flight_lines)

 
