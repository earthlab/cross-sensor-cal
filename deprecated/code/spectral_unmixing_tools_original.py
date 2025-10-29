# DEPRECATED: We no longer export GeoTIFF deliverables.
# Final products are ENVI (.img/.hdr) plus optional .parquet.
# This file has been staged for removal.
# Standard Library Imports
import glob
import json
import os
import random
import subprocess
import sys
import time

# Third-Party Imports
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
try:  # pragma: no cover - optional dependency guard
    import ray
except ModuleNotFoundError:  # pragma: no cover - handled when features are used
    ray = None  # type: ignore[assignment]
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import box
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

# Local Application/Specific Imports
import hytools as ht














def generate_correction_configs(main_image_file, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(main_image_file).split('.')[0]
    input_dir = os.path.dirname(main_image_file)
    
    # Define the bad bands, file type, and names for the ancillary datasets
    bad_bands = [[300, 400], [1337, 1430], [1800, 1960], [2450, 2600]]
    file_type = 'envi'
    aviris_anc_names = ['path_length', 'sensor_az', 'sensor_zn', 'solar_az', 'solar_zn', 'phase', 'slope', 'aspect', 'cosine_i']

    # Find ancillary files related to the main image file
    anc_files_pattern = os.path.join(input_dir, f"{base_name}_ancillary*")
    anc_files = glob.glob(anc_files_pattern)
    anc_files.sort()

    files_to_process = [main_image_file] + anc_files

    for file_path in files_to_process:
        is_ancillary = '_ancillary' in file_path
        config_type = 'ancillary' if is_ancillary else ''
        config_filename = f"{base_name}_{config_type}.json"
        config_file_path = os.path.join(output_dir, config_filename)

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
                    ['cloud', {'method': 'zhai_2018', 'cloud': True, 'shadow': True, 'T1': 0.01, 't2': 1/10, 't3': 1/4, 't4': 1/2, 'T7': 9, 'T8': 9}]
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

        # Adjust the ancillary settings as needed for your process
        if is_ancillary:
            print("ancillary extras")
            # Example: Populate config_dict["anc_files"] as necessary

        with open(config_file_path, 'w') as outfile:
            json.dump(config_dict, outfile, indent=3)
        print(f"Configuration saved to {config_file_path}")

pass








def process_hdf5_with_neon2envi(image_path, site_code):
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



pass

def get_spectral_data_and_wavelengths(filename, row_step, col_step):
    """
    Retrieve spectral data and wavelengths from a specified file using HyTools library.

    Parameters:
    - filename: str, the path to the file to be read.
    - row_step: int, the step size to sample rows by.
    - col_step: int, the step size to sample columns by.

    Returns:
    - original: np.ndarray, a 2D array where each row corresponds to the spectral data from one pixel.
    - wavelengths: np.ndarray, an array containing the wavelengths corresponding to each spectral band.
    """
    # Initialize the HyTools object
    envi = ht.HyTools()
    
    # Read the file using the specified format
    envi.read_file(filename, 'envi')
    
    colrange = np.arange(0, envi.columns).tolist()  # Adjusted to use envi.columns for dynamic range
    pixel_lines = np.arange(0,envi.lines).tolist()
    #pixel_lines
    rowrange =  sorted(random.sample(pixel_lines, envi.columns))
    # Retrieve the pixels' spectral data
    original = envi.get_pixels(rowrange, colrange)

    #original = pd.DataFrame(envi.get_pixels(rowrange, colrange))
    #original['index'] = np.arange(original.shape[0])
    
    # Also retrieve the wavelengths
    wavelengths = envi.wavelengths
    
    return original, wavelengths
pass



def load_spectra(filenames, row_step=6, col_step=1):
    results = {}
    for filename in filenames:
        try:
            spectral_data, wavelengths = get_spectral_data_and_wavelengths(filename, row_step, col_step)
            results[filename] = {"spectral_data": spectral_data, "wavelengths": wavelengths}
        except TypeError:
            print(f"Error processing file: {filename}")
    return results

pass




def extract_overlapping_layers_to_2d_dataframe(raster_path, gpkg_path):
    # Load polygons
    polygons = gpd.read_file(gpkg_path)

    # Initialize a list to store data
    data = []

    # Ensure polygons are in the same CRS as the raster
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if polygons.crs != raster_crs:
            polygons = polygons.to_crs(raster_crs)

        raster_bounds = src.bounds
        # Corrected line: Use geom to refer to each geometry in the GeoSeries
        polygons['intersects'] = polygons.geometry.apply(lambda geom: geom.intersects(box(*raster_bounds)))
        overlapping_polygons = polygons[polygons['intersects']].copy()

        # Process each overlapping polygon
        for index, polygon in overlapping_polygons.iterrows():
            mask_result, _ = mask(src, [polygon.geometry], crop=True, all_touched=True)
            row = {'polygon_id': index}
            for layer in range(mask_result.shape[0]):
                # Compute the mean of the raster values for this layer, excluding nodata values
                valid_values = mask_result[layer][mask_result[layer] != src.nodata]
                if valid_values.size > 0:
                    layer_mean = valid_values.mean()
                else:
                    layer_mean = np.nan  # Use NaN for areas with only nodata values
                row[f'layer_{layer+1}'] = layer_mean
            
            # Append the row to the data list
            data.append(row)

    # Create DataFrame from accumulated data
    results_df = pd.DataFrame(data)

    return results_df



pass


def rasterize_polygons_to_match_envi(gpkg_path, existing_raster_path, output_raster_path, attribute=None):
    # Load polygons
    polygons = gpd.read_file(gpkg_path)

    # Read existing raster metadata
    with rasterio.open(existing_raster_path) as existing_raster:
        existing_meta = existing_raster.meta
        existing_crs = existing_raster.crs

    # Plot the existing raster
    fig, axs = plt.subplots(1, 3, figsize=(21, 40))
    with rasterio.open(existing_raster_path) as existing_raster:
        show(existing_raster, ax=axs[0], title="Existing Raster")

    # Reproject polygons if necessary and plot them
    if polygons.crs != existing_crs:
        polygons = polygons.to_crs(existing_crs)
    polygons.plot(ax=axs[1], color='red', edgecolor='black')
    axs[1].set_title("Polygons Layer")

    # Rasterize polygons
    rasterized_polygons = rasterize(
        shapes=((geom, value) for geom, value in zip(polygons.geometry, polygons[attribute] if attribute and attribute in polygons.columns else polygons.index)),
        out_shape=(existing_meta['height'], existing_meta['width']),
        fill=0,
        transform=existing_meta['transform'],
        all_touched=True,
        dtype=existing_meta['dtype']
    )

    # Save the rasterized polygons to a new ENVI file
    with rasterio.open(output_raster_path, 'w', **existing_meta) as out_raster:
        out_raster.write(rasterized_polygons, 1)

    # Plot the new rasterized layer
    with rasterio.open(output_raster_path) as new_raster:
        show(new_raster, ax=axs[2], title="Rasterized Polygons Layer")

    plt.tight_layout()
    plt.show()

    print(f"Rasterization complete. Output saved to {output_raster_path}")

pass

def prepare_spectral_data(spectral_data, wavelengths):
    # Transpose and melt the spectral data to long format
    long_df = pd.melt(pd.DataFrame(spectral_data).transpose(), var_name="band", value_name="reflectance")
    
    # Create a DataFrame for wavelengths and assign a 'band' column based on index
    waves = pd.DataFrame(wavelengths, columns=["wavelength_nm"])
    waves['band'] = range(len(waves))
    
    # Merge the spectral data with wavelengths using the 'band' column
    merged_data = pd.merge(long_df, waves, on='band')
    
    # Convert 'wavelength_nm' to numeric, if necessary
    merged_data["wavelength_nm"] = pd.to_numeric(merged_data["wavelength_nm"])
    
    return merged_data

pass

def reshape_spectra(results, index):
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
        # Add a label column to the merged data
        first_key = keys[index].replace("export/resample_", "").replace(".img", "")  # Modified here to remove "export/"
        merged_data['sensor'] = first_key
        merged_data = merged_data.reindex(columns=['sensor', 'band', 'wavelength_nm', 'reflectance','pixel'])
        
        length = len(merged_data)
        sequence = np.arange(0, 1071)  # Creates an array [1, 2, ..., 999]
        repeated_sequence = np.resize(sequence, length)  # Resize the sequence to match the DataFrame's length

        merged_data['pixel'] = repeated_sequence  # Add the column to your DataFrame
        merged_data['sensor_band'] = merged_data['sensor'].astype(str) + '_' + merged_data['band'].astype(str)
        return merged_data
    else:
        merged_data = prepare_spectral_data(spectral_data, wavelengths)
        # Add a label column to the merged data
        first_key = keys[index].replace("export/ENVI__corrected_0", "hyperspectral_corrected")
        first_key = first_key.replace("output/ENVI", "hyperspectral_original")
        merged_data['sensor'] = first_key
        merged_data = merged_data.reindex(columns=['sensor', 'band', 'wavelength_nm', 'reflectance','pixel'])
        length = len(merged_data)
        sequence = np.arange(0, 1071)  # Creates an array [1, 2, ..., 999]
        repeated_sequence = np.resize(sequence, length)  # Resize the sequence to match the DataFrame's length

        merged_data['pixel'] = repeated_sequence  # Add the column to your DataFrame
        merged_data['sensor_band'] = merged_data['sensor'].astype(str) + '_' + merged_data['band'].astype(str)
        return merged_data



pass

def concatenate_sensors(reshape_spectra_function, spectra, sensors_range):
    all_spectra = []
    for sensor in sensors_range:  # Typically range(6) for sensors 0 to 5
        reshaped_spectra = reshape_spectra_function(spectra, sensor)
        all_spectra.append(reshaped_spectra)
    
    # Concatenate all reshaped spectra DataFrames into one, preserving columns
    concatenated_spectra = pd.concat(all_spectra, ignore_index=True)
    return concatenated_spectra



pass

def plot_spectral_data(df, highlight_pixel):
    df = df[df['wavelength_nm'] > 0]  # Exclude negative wavelength_nm values
    df['reflectance'] = df['reflectance'].replace(-9999, np.nan)
    unique_indices = df['pixel'].unique()

    for idx in unique_indices:
        subset = df[df['pixel'] == idx]
        plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.05, color="blue")

    # Highlight a specific pixel
    highlighted_subset = df[df['pixel'] == highlight_pixel]
    # Inside the plot_spectral_data function, after highlighting the specific pixel


    if (highlighted_subset['reflectance'] == -9999).all() or highlighted_subset['reflectance'].isna().all():
        print(f"Warning: Pixel {highlight_pixel} data is entirely -9999 or NaN.")

    plt.plot(highlighted_subset['wavelength_nm'], highlighted_subset['reflectance'], color='red', linewidth=10, label=f'Pixel {highlight_pixel}')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 10000)
    plt.show()


pass

def plot_each_sensor_with_highlight(concatenated_sensors, highlight_pixel, save_path=None):
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




pass

def plot_with_highlighted_sensors(concatenated_sensors, highlight_pixels, save_path=None):
    plt.figure(figsize=(10, 5))
    
    # Ensure highlight_pixels is a list for iteration
    if not isinstance(highlight_pixels, list):
        highlight_pixels = [highlight_pixels]
    
    # Apply initial data cleaning
    concatenated_sensors = concatenated_sensors[concatenated_sensors['wavelength_nm'] > 0]
    concatenated_sensors['reflectance'] = concatenated_sensors['reflectance'].replace(-9999, np.nan)
    
    # Plotting hyperspectral corrected data in blue
    hyperspectral_corrected = concatenated_sensors[concatenated_sensors['sensor'] == 'hyperspectral_corrected']
    for pixel in hyperspectral_corrected['pixel'].unique():
        subset = hyperspectral_corrected[hyperspectral_corrected['pixel'] == pixel]
        plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.05, color="blue")
    
    # Overlaying highlighted lines from each sensor in red
    for sensor in concatenated_sensors['sensor'].unique():
        if sensor != 'hyperspectral_corrected':  # Exclude hyperspectral corrected data from red lines
            for highlight_pixel in highlight_pixels:
                highlighted_subset = concatenated_sensors[(concatenated_sensors['pixel'] == highlight_pixel) & (concatenated_sensors['sensor'] == sensor)]
                if not highlighted_subset.empty:
                    plt.plot(highlighted_subset['wavelength_nm'], highlighted_subset['reflectance'], color='red', linewidth=2, label=sensor)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.ylim(0, None)  # Adjusted to auto-scale based on data
    plt.xlim(350, 2550)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1])
    plt.show()

pass


def fit_models_with_different_alpha(data, n_levels=100):
    data['reflectance'] = data['reflectance'].replace(np.nan, 0)
    
    X = data[['wavelength_nm']]
    y = data['reflectance']
    
    # Store models and predictions
    models = []
    alphas = np.linspace(0.01, 0.99, n_levels)
    
    for alpha in alphas:
        model = GradientBoostingRegressor(n_estimators=500, max_depth=15, learning_rate=0.09,
                                          subsample=0.1, loss='quantile', alpha=alpha)
        model.fit(X, y)
        models.append(model)
        
        # You can also store predictions if needed
        data[f'{alpha:.2f}'] = model.predict(X)
    
    return models, data


pass


def plot_with_highlighted_sensors(concatenated_sensors, highlight_pixels, save_path=None):
    plt.figure(figsize=(10, 5))
    
    # Ensure highlight_pixels is a list for iteration
    if not isinstance(highlight_pixels, list):
        highlight_pixels = [highlight_pixels]
    
    # Apply initial data cleaning
    concatenated_sensors = concatenated_sensors[concatenated_sensors['wavelength_nm'] > 0]
    #concatenated_sensors['boosted_predictions_01'] = concatenated_sensors['boosted_predictions_01'].replace(-9999, np.nan)
    concatenated_sensors['reflectance'] = concatenated_sensors['reflectance'].replace(-9999, np.nan)
    
    # Plotting hyperspectral corrected data in blue
    hyperspectral_corrected = concatenated_sensors[concatenated_sensors['sensor'] == 'hyperspectral_corrected']
    for pixel in hyperspectral_corrected['pixel'].unique():
        subset = hyperspectral_corrected[hyperspectral_corrected['pixel'] == pixel]
        plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.05, color="blue")
    
    # Overlaying highlighted lines from each sensor in red
    for sensor in concatenated_sensors['sensor'].unique():
        if sensor != 'hyperspectral_corrected':  # Exclude hyperspectral corrected data from red lines
            for highlight_pixel in highlight_pixels:
                highlighted_subset = concatenated_sensors[(concatenated_sensors['pixel'] == highlight_pixel) & (concatenated_sensors['sensor'] == sensor)]
                if not highlighted_subset.empty:
                    plt.plot(highlighted_subset['wavelength_nm'], highlighted_subset['0.50'], linewidth=10, label=sensor)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.ylim(0, 10000)  # Adjusted to auto-scale based on data
    plt.xlim(350, 2550)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1])
    plt.show()




pass



def boosted_quantile_plot(data, num_lines=10, title='Hyperspectral Corrected Predictions by Alpha', save_path=None):
    plt.figure(figsize=(10, 6))
    total_alphas = 100
    step = total_alphas // num_lines

# Plotting hyperspectral corrected data in blue
    hyperspectral_corrected = concatenated_sensors[concatenated_sensors['sensor'] == 'hyperspectral_corrected']
    for pixel in hyperspectral_corrected['pixel'].unique():
        subset = hyperspectral_corrected[hyperspectral_corrected['pixel'] == pixel]
        plt.plot(subset['wavelength_nm'], subset['reflectance'], alpha=0.04, color="blue", linewidth=0.3)
    
    alpha_values = np.linspace(0.01, 0.99, total_alphas)[::step]
    for alpha_value in alpha_values:
        alpha_col = f'{alpha_value:.2f}'
        adjusted_alpha = 1 - alpha_value if alpha_value > 0.5 else alpha_value
        plt.plot(data['wavelength_nm'], data[alpha_col], label=f'Probability {alpha_col}', alpha=adjusted_alpha ,color="red", linewidth=adjusted_alpha*6)
    
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


pass

def boosted_quantile_plot_by_sensor(data, num_lines=10, title='Hyperspectral Corrected Predictions by Alpha', save_path=None):
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
            adjusted_label = f'{adjusted_alpha:.2f}'  # Label reflects adjusted alpha
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





def show_rgb(file_paths, r=660, g=550, b=440):
    # Ensure file_paths is a list to simplify processing
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    
    # Determine the number of files to set the layout accordingly
    n_files = len(file_paths)
    # Set the figure size larger if more images, adjust width & height as needed
    fig_width = 7 * n_files  # Increase the width for each additional image
    fig_height = 100  # Adjust the height as needed

     # Warning message
    warning_message = ("WARNING: The images displayed are part of a panel and "
                       "the flight lines are not presented on the same map. "
                       "Spatial relationships between panels may not be accurate.")
    print(warning_message)
    
    # Create the figure with adjusted dimensions
    fig, axs = plt.subplots(1, n_files, figsize=(fig_width, fig_height), squeeze=False)

    for file_path, ax in zip(file_paths, axs.flatten()):
        # Initialize the HyTools object and read the file
        hy_obj = ht.HyTools()
        hy_obj.read_file(file_path, 'envi')
        
        # Extract RGB bands based on specified wavelengths (or band indices)
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
        ax.axis('off')  # Hide axis
        ax.set_title(f"RGB Composite: {file_path.split('/')[-1]}")

    plt.subplots_adjust(wspace=0, hspace=0)  # Remove any white space between subplots
    plt.show()


pass



def clean_csv_files_in_subfolders(base_folder, chunk_size=100000, suffix='_no_NaN'):
    """
    Cleans CSV files within each subdirectory of the given base folder by removing rows with any NaN values.
    
    The function searches for CSV files following a specific naming pattern within each subfolder,
    processes them in chunks to handle large files efficiently, and saves the cleaned data to a new CSV file
    with a specified suffix.
    
    Parameters:
    - base_folder (str): The path to the base directory containing subdirectories with CSV files.
    - chunk_size (int, optional): The number of rows per chunk to process. Defaults to 100,000.
    - suffix (str, optional): Suffix to append to the cleaned CSV filenames. Defaults to '_no_NaN'.
    
    Raises:
    - FileNotFoundError: If the original CSV file does not exist in a subdirectory.
    - Exception: For any unexpected errors during processing.
    """
    print(f"Starting CSV cleaning in base folder: {base_folder}\n")
    
    # Iterate through each subdirectory in the base folder
    subdirectories = [os.path.join(base_folder, d) for d in os.listdir(base_folder) 
                      if os.path.isdir(os.path.join(base_folder, d))]
    
    if not subdirectories:
        print(f"No subdirectories found in the base folder: {base_folder}\n")
        return
    
    for subdir in subdirectories:
        try:
            # Define the pattern to identify the original CSV file
            original_csv_pattern = os.path.join(subdir, '*_spectral_data_all_sensors.csv')
            original_csv_files = glob.glob(original_csv_pattern)
            
            if not original_csv_files:
                print(f"No original CSV files found in subdirectory: {subdir}\n")
                continue  # Skip to the next subdirectory
            
            for original_csv in original_csv_files:
                # Define the cleaned CSV file path
                base_name = os.path.splitext(os.path.basename(original_csv))[0]
                cleaned_csv = os.path.join(subdir, f"{base_name}{suffix}.csv")
                
                print(f"Processing file: {original_csv}")
                print(f"Cleaned CSV will be saved to: {cleaned_csv}\n")
                
                # Determine the total number of lines in the file (minus the header)
                with open(original_csv, 'r') as f:
                    total_lines = sum(1 for _ in f) - 1  # Subtract 1 for the header
                
                if total_lines <= 0:
                    print(f"The file {original_csv} is empty or has only a header. Skipping.\n")
                    continue
                
                # Calculate the total number of chunks
                num_chunks = (total_lines // chunk_size) + 1
                
                # Process the file in chunks with a progress bar
                with pd.read_csv(original_csv, chunksize=chunk_size) as reader, open(cleaned_csv, 'w') as output_file:
                    for i, chunk in enumerate(tqdm(reader, total=num_chunks, desc=f"Cleaning {base_name}")):
                        # Drop rows with any NaN values
                        chunk_cleaned = chunk.dropna()
                        
                        # Write to the output file
                        mode = 'w' if i == 0 else 'a'  # Write mode for the first chunk, append for others
                        header = i == 0  # Write the header only for the first chunk
                        chunk_cleaned.to_csv(output_file, mode=mode, index=False, header=header)
                
                print(f"Cleaned CSV saved to: {cleaned_csv}\n")
        
        except FileNotFoundError as fnf_error:
            print(f"File not found: {fnf_error}\n")
        except pd.errors.EmptyDataError:
            print(f"No data: The file {original_csv} is empty.\n")
        except Exception as e:
            print(f"An error occurred while processing {subdir}: {e}\n")
    
    print("CSV cleaning process completed.\n")
pass


def validate_output_files(base_folder):
    """
    Validates that all expected output files are present and valid within each subdirectory of the base folder.

    Parameters:
    - base_folder (str): The path to the base directory containing subdirectories with output files.

    Returns:
    - None: Prints a simplified validation summary directly.
    """
    # Define the exact expected file patterns
    expected_files = [
        '*_reflectance',
        '*_envi',
        '*_envi.hdr',
        '*_envi_mask',
        '*_envi_mask.hdr',
        '*_resample_Landsat_5_TM.hdr',
        '*_resample_Landsat_5_TM.img',
        '*_resample_Landsat_7_ETMplus.hdr',
        '*_resample_Landsat_7_ETMplus.img',
        '*_resample_Landsat_8_OLI.hdr',
        '*_resample_Landsat_8_OLI.img',
        '*_resample_Landsat_9_OLI-2.hdr',
        '*_resample_Landsat_9_OLI-2.img',
        '*_resample_MicaSense.hdr',
        '*_resample_MicaSense.img',
        '*_ancillary',
        '*_ancillary.hdr',
        '*_brdf_coeffs__envi.json',
        '*_config__anc.json',
        '*_config__envi.json',
        '*_spectral_data_all_sensors.csv',
        '*_spectral_data_all_sensors_no_NaN.csv',
        '*_topo_coeffs__envi.json',
        '*.hdr'
    ]
    
    subdirectories = [
        os.path.join(base_folder, d) for d in os.listdir(base_folder) 
        if os.path.isdir(os.path.join(base_folder, d)) and not d.startswith('.ipynb_checkpoints')
    ]
    
    if not subdirectories:
        print(f"No subdirectories found in the base folder: {base_folder}")
        return
    
    print(f"Starting validation of output files in base folder: {base_folder}\n")
    
    for subdir in tqdm(subdirectories, desc="Validating subdirectories"):
        subdir_name = os.path.basename(subdir)
        missing_files = []
        invalid_files = []
        
        # Validate each expected file pattern
        for pattern in expected_files:
            matched_files = glob.glob(os.path.join(subdir, pattern))
            
            if not matched_files:
                missing_files.append(pattern)
            else:
                for file in matched_files:
                    try:
                        if file.endswith(('.hdr')):
                            # Skip validation for .hdr files; just check if they exist
                            pass
                        elif file.endswith(('.img', '_envi', '_mask')):
                            # Validate raster files
                            with rasterio.open(file) as src:
                                src.meta  # Access metadata to confirm file is valid
                        elif file.endswith('.csv'):
                            # Validate CSV files
                            pd.read_csv(file, nrows=5)  # Read first few rows
                        elif file.endswith('.json'):
                            # Validate JSON files
                            with open(file, 'r') as f:
                                json.load(f)  # Try loading JSON
                    except Exception:
                        invalid_files.append(file)
        
        # Print summary for each subdirectory
        if not missing_files and not invalid_files:
            print(f"Subdirectory: {subdir_name}\n  All expected files are present and valid.\n")
        else:
            print(f"Subdirectory: {subdir_name}")
            if missing_files:
                print("  Missing Files:")
                for missing in missing_files:
                    print(f"    - {missing}")
            if invalid_files:
                print("  Invalid Files:")
                for invalid in invalid_files:
                    print(f"    - {invalid}")
            print()  # Blank line for better readability


pass





import os
import glob
import numpy as np
import pandas as pd
import json
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from spectral import open_image
import warnings

# Suppress NotGeoreferencedWarning for clarity
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ----- Existing ENVI Processing Classes and Functions -----

class ENVIProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None  # This will hold the raster data array
        self.file_type = "envi"

    def load_data(self):
        """Loads the raster data from the file_path into self.data"""
        with rasterio.open(self.file_path) as src:
            self.data = src.read()  # Read all bands

    def get_chunk_from_extent(self, corrections=[], resample=False):
        self.load_data()  # Ensure data is loaded
        return self.data






def load_and_combine_rasters(raster_paths):
    """
    Loads and combines raster data from a list of file paths.
    Assumes each raster has shape (bands, rows, cols) and that
    all rasters can be concatenated along the band dimension.
    """
    chunks = []
    for path in raster_paths:
        processor = ENVIProcessor(path)
        chunk = processor.get_chunk_from_extent(corrections=['some_correction'], resample=False)
        chunks.append(chunk)
    combined_array = np.concatenate(chunks, axis=0)  # Combine along the first axis (bands)
    return combined_array

def process_and_flatten_array(array, json_dir='Resampling', original_bands=426, corrected_bands=426,
                              original_wavelengths=None, corrected_wavelengths=None, folder_name=None,
                              map_info=None):
    """
    Processes a 3D numpy array to a DataFrame, adds metadata columns, 
    renames columns dynamically based on JSON configuration, and adds Pixel_id.
    Uses provided wavelength lists to name original and corrected bands, and includes geocoordinates.

    Parameters:
    - array: A 3D numpy array of shape (bands, rows, cols).
    - json_dir: Directory containing the landsat_band_parameters.json file.
    - original_bands: Number of original bands expected.
    - corrected_bands: Number of corrected bands expected.
    - original_wavelengths: List of wavelengths for the original bands (floats).
    - corrected_wavelengths: List of wavelengths for the corrected bands (floats).
    - folder_name: Name of the subdirectory (flight line identifier).
    - map_info: The map info array from the metadata for georeferencing.

    Returns:
    - A pandas DataFrame with additional metadata columns and renamed band columns.
    """
    if len(array.shape) != 3:
        raise ValueError("Input array must be 3-dimensional. Expected (bands, rows, cols).")

    json_file = os.path.join(json_dir, 'landsat_band_parameters.json')
    if not os.path.isfile(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, 'r') as f:
        config = json.load(f)

    bands, rows, cols = array.shape
    print(f"[DEBUG] array shape: bands={bands}, rows={rows}, cols={cols}")

    reshaped_array = array.reshape(bands, -1).T  # (pixels, bands)
    pixel_indices = np.indices((rows, cols)).reshape(2, -1).T  # (pixels, 2)
    df = pd.DataFrame(reshaped_array, columns=[f'Band_{i+1}' for i in range(bands)])

    # Extract map info for georeferencing:
    # Format: [projection, x_pixel_start, y_pixel_start, map_x, map_y, x_res, y_res, ...]
    # Typically:
    #   x_pixel_start, y_pixel_start = 1,1 for upper-left pixel
    #   map_x, map_y = coordinates of that upper-left pixel
    #   x_res, y_res = pixel sizes (y_res should be positive but we assume north-down in ENVI)
    if map_info is not None and len(map_info) >= 7:
        projection = map_info[0]
        x_pixel_start = float(map_info[1])
        y_pixel_start = float(map_info[2])
        map_x = float(map_info[3])
        map_y = float(map_info[4])
        x_res = float(map_info[5])
        y_res = float(map_info[6])
    else:
        # Fallback if map_info is not provided
        projection = 'Unknown'
        x_pixel_start, y_pixel_start = 1.0, 1.0
        map_x, map_y = 0.0, 0.0
        x_res, y_res = 1.0, 1.0

    # Compute Easting, Northing
    # Pixel_row and Pixel_col are zero-based. 
    # According to ENVI conventions:
    # Easting = map_x + (pixel_col - (x_pixel_start - 1)) * x_res
    # Northing = map_y - (pixel_row - (y_pixel_start - 1)) * y_res
    pixel_row = pixel_indices[:, 0]
    pixel_col = pixel_indices[:, 1]
    Easting = map_x + (pixel_col - (x_pixel_start - 1)) * x_res
    Northing = map_y - (pixel_row - (y_pixel_start - 1)) * y_res

    # Insert Pixel info and coordinates
    df.insert(0, 'Pixel_Col', pixel_col)
    df.insert(0, 'Pixel_Row', pixel_row)
    df.insert(0, 'Pixel_id', np.arange(len(df)))
    df.insert(3, 'Easting', Easting)
    df.insert(4, 'Northing', Northing)

    # Check we have enough bands
    if bands < (original_bands + corrected_bands):
        raise ValueError(
            f"Not enough bands. Expected at least {original_bands + corrected_bands} (original+corrected), but got {bands}."
        )

    # Determine Corrected and Resampled flags
    remaining_bands = bands - (original_bands + corrected_bands)
    corrected_flag = "Yes" if corrected_bands > 0 else "No"
    resampled_flag = "Yes" if remaining_bands > 0 else "No"

    # Metadata columns: Subdirectory, Data_Source, Sensor_Type, Corrected, Resampled
    # Insert these at the very front
    df.insert(0, 'Resampled', resampled_flag)
    df.insert(0, 'Corrected', corrected_flag)
    df.insert(0, 'Sensor_Type', 'Hyperspectral')
    df.insert(0, 'Data_Source', 'Flight line')
    df.insert(0, 'Subdirectory', folder_name if folder_name else 'Unknown')

    # Rename bands with wavelengths
    band_names = []
    # Original bands
    if original_wavelengths is not None and len(original_wavelengths) >= original_bands:
        for i in range(original_bands):
            wl = original_wavelengths[i]
            band_names.append(f"Original_band_{i+1}_wl_{wl}nm")
    else:
        for i in range(1, original_bands + 1):
            band_names.append(f"Original_band_{i}")

    # Corrected bands
    if corrected_wavelengths is not None and len(corrected_wavelengths) >= corrected_bands:
        for i in range(corrected_bands):
            wl = corrected_wavelengths[i]
            band_names.append(f"Corrected_band_{i+1}_wl_{wl}nm")
    elif original_wavelengths is not None and len(original_wavelengths) >= corrected_bands:
        for i in range(corrected_bands):
            wl = original_wavelengths[i]
            band_names.append(f"Corrected_band_{i+1}_wl_{wl}nm")
    else:
        for i in range(1, corrected_bands + 1):
            band_names.append(f"Corrected_band_{i}")

    print(f"[DEBUG] remaining_bands for resampled sensors: {remaining_bands}")

    sensor_bands_assigned = 0
    for sensor, details in config.items():
        wavelengths = details.get('wavelengths', [])
        for i, wl in enumerate(wavelengths, start=1):
            if sensor_bands_assigned < remaining_bands:
                band_names.append(f"{sensor}_band_{i}_wl_{wl}nm")
                sensor_bands_assigned += 1
            else:
                break
        if sensor_bands_assigned >= remaining_bands:
            break

    if sensor_bands_assigned < remaining_bands:
        extra = remaining_bands - sensor_bands_assigned
        print(f"[DEBUG] {extra} leftover bands have no matching sensors/wavelengths in JSON. Naming them generically.")
        for i in range(1, extra + 1):
            band_names.append(f"Unassigned_band_{i}")

    # Now we have Pixel_id, Pixel_Row, Pixel_Col, Easting, Northing, and multiple metadata columns.
    # Determine how many leading metadata columns we have before bands:
    # Currently: Subdirectory, Data_Source, Sensor_Type, Corrected, Resampled, Pixel_id, Pixel_Row, Pixel_Col, Easting, Northing
    # That's 10 columns before bands start.
    metadata_count = 10

    new_columns = list(df.columns[:metadata_count]) + band_names
    if len(new_columns) != df.shape[1]:
        raise ValueError(
            f"Band naming mismatch: {len(new_columns)} columns assigned vs {df.shape[1]} in df. Check indexing."
        )

    df.columns = new_columns

    print(f"[DEBUG] Final DataFrame shape: {df.shape}")
    print("[DEBUG] Columns assigned successfully.")

    return df

def clean_data_and_write_to_csv(df, output_csv_path, chunk_size=100000):
    """
    Cleans a large DataFrame by processing it in chunks and then writes it to a CSV file.
    """
    total_rows = df.shape[0]
    num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

    print(f"Cleaning data and writing to CSV in {num_chunks} chunk(s).")

    first_chunk = True
    for i, start_row in enumerate(range(0, total_rows, chunk_size)):
        chunk = df.iloc[start_row:start_row + chunk_size].copy()
        non_pixel_cols = [col for col in chunk.columns if not col.startswith('Pixel') and 
                          col not in ['Subdirectory','Data_Source','Sensor_Type','Corrected','Resampled',
                                      'Easting','Northing']]

        # Replace -9999 values with NaN
        chunk[non_pixel_cols] = chunk[non_pixel_cols].apply(
            lambda x: np.where(np.isclose(x, -9999, atol=1), np.nan, x)
        )

        # Drop rows with all NaNs in non-pixel columns (spectral data)
        chunk.dropna(subset=non_pixel_cols, how='all', inplace=True)

        mode = 'w' if first_chunk else 'a'
        header = True if first_chunk else False
        chunk.to_csv(output_csv_path, mode=mode, header=header, index=False)

        print(f"Chunk {i+1}/{num_chunks} processed and written.")
        first_chunk = False

    print(f"Data cleaning complete. Output written to: {output_csv_path}")

def control_function(directory):
    """
    Orchestrates the finding, loading, processing of raster files found in a specified directory,
    cleans the processed data, and saves it to a CSV file in the same directory.
    """
    raster_paths = find_raster_files(directory)

    if not raster_paths:
        print(f"No matching raster files found in {directory}.")
        return

    # Assume original file name (without _envi etc.) is the directory name
    base_name = os.path.basename(os.path.normpath(directory))
    hdr_file = os.path.join(os.path.dirname(directory), base_name + '.hdr')
    if not os.path.isfile(hdr_file):
        hdr_file = os.path.join(directory, base_name + '.hdr')

    original_wavelengths = None
    map_info = None
    if os.path.isfile(hdr_file):
        img = open_image(hdr_file)
        original_wavelengths = img.metadata.get('wavelength', [])
        # Convert to float if they are strings
        original_wavelengths = [float(w) for w in original_wavelengths]
        map_info = img.metadata.get('map info', None)
    else:
        print(f"No HDR file found at {hdr_file}. Will use generic band names and no geocoords.")

    corrected_wavelengths = original_wavelengths

    # Load and combine raster data
    combined_array = load_and_combine_rasters(raster_paths)  
    print(f"Combined array shape for directory {directory}: {combined_array.shape}")

    # Attempt to process and flatten the array into a DataFrame
    try:
        df_processed = process_and_flatten_array(
            combined_array,
            json_dir='Resampling',
            original_bands=426,
            corrected_bands=426,
            original_wavelengths=original_wavelengths,
            corrected_wavelengths=corrected_wavelengths,
            folder_name=base_name,
            map_info=map_info
        )  
        print(f"DataFrame shape after flattening for directory {directory}: {df_processed.shape}")
    except ValueError as e:
        print(f"ValueError encountered during processing of {directory}: {e}")
        print("Check the number of bands vs. the expected column names in process_and_flatten_array().")
        return
    except Exception as e:
        print(f"An unexpected error occurred while processing {directory}: {e}")
        return

    # Extract the folder name from the directory path
    folder_name = os.path.basename(os.path.normpath(directory))
    output_csv_name = f"{folder_name}_spectral_data_all_sensors.csv"
    output_csv_path = os.path.join(directory, output_csv_name)

    # Always overwrite if CSV exists
    if os.path.exists(output_csv_path):
        print(f"CSV {output_csv_path} already exists and will be overwritten.")

    # Clean data and write to CSV
    clean_data_and_write_to_csv(df_processed, output_csv_path)  
    print(f"Processed and cleaned data saved to {output_csv_path}")

# ----- Masking Function Integration -----



# ----- New Function to Process All ENVI Files in a Directory -----

def process_all_envi_files_with_mask(
    directory,
    geojson_path,
    raster_crs_override=None,
    polygons_crs_override=None,
    output_masked_suffix="_masked",
    plot_output=False,  # Disable plotting by default when processing multiple files
    plot_filename_template="plot_results_{envi_basename}.png",
    dpi=300
):
    """
    Processes all ENVI raster files in the specified directory by masking them with the provided GeoJSON polygons.

    Parameters:
    -----------
    directory : str
        Path to the directory containing ENVI raster files.
    geojson_path : str
        Path to the GeoJSON file containing polygons for masking.
    raster_crs_override : str, optional
        CRS to assign to the raster if it's undefined (e.g., 'EPSG:4326').
    polygons_crs_override : str or CRS, optional
        CRS to assign to the polygons if they're undefined.
    output_masked_suffix : str, optional
        Suffix to append to each masked raster filename.
    plot_output : bool, optional
        Whether to generate and save plots for each masked raster.
    plot_filename_template : str, optional
        Template for the plot filenames. Use '{envi_basename}' as a placeholder for the ENVI file basename.
    dpi : int, optional
        Resolution of the saved plots in dots per inch.

    Returns:
    --------
    masked_raster_paths : list
        List of paths to the saved masked raster files.
    """
    # Find all relevant ENVI raster files in the directory
    envi_files = find_raster_files(directory)
    print(f"Found {len(envi_files)} ENVI raster files in {directory}.")

    if not envi_files:
        print("No ENVI raster files to process.")
        return []

    masked_raster_paths = []

    for envi_path in envi_files:
        try:
            print(f"\nProcessing ENVI file: {envi_path}")

            # Determine plot filename if plotting is enabled
            if plot_output:
                envi_basename = os.path.splitext(os.path.basename(envi_path))[0]
                plot_filename = plot_filename_template.format(envi_basename=envi_basename)
            else:
                plot_filename = None

            # Apply the masking function
            masked_raster = mask_raster_with_polygons(
                envi_path=envi_path,
                geojson_path=geojson_path,
                raster_crs_override=raster_crs_override,
                polygons_crs_override=polygons_crs_override,
                output_masked_suffix=output_masked_suffix,
                plot_output=plot_output,
                plot_filename=plot_filename,
                dpi=dpi
            )

            if masked_raster:
                masked_raster_paths.append(masked_raster)
                print(f"Successfully masked raster: {masked_raster}")
            else:
                print(f"Masking skipped for raster: {envi_path}")

        except Exception as e:
            print(f"Error processing {envi_path}: {e}")
            continue  # Continue with the next file

    print(f"\nAll processing complete. {len(masked_raster_paths)} rasters masked successfully.")
    return masked_raster_paths

# ----- Simplified Function: Only Folder Name and Polygon Layer as Required Parameters -----

def process_folder_with_polygons(
    folder_name,
    polygon_layer,
    raster_crs_override='EPSG:4326',
    polygons_crs_override='EPSG:4326',
    output_masked_suffix="_masked",
    plot_output=False,
    plot_filename_template="plot_results_{envi_basename}.png",
    dpi=300
):
    """
    Simplified function to process all ENVI raster files in a folder by masking them with a polygon layer.
    Only the folder name and polygon layer are required; all other parameters are optional with default values.

    Parameters:
    -----------
    folder_name : str
        Path to the directory containing ENVI raster files.
    polygon_layer : str
        Path to the GeoJSON file containing polygons for masking.
    raster_crs_override : str, optional
        CRS to assign to the raster if it's undefined (default: 'EPSG:4326').
    polygons_crs_override : str or CRS, optional
        CRS to assign to the polygons if they're undefined (default: 'EPSG:4326').
    output_masked_suffix : str, optional
        Suffix to append to each masked raster filename (default: "_masked").
    plot_output : bool, optional
        Whether to generate and save plots for each masked raster (default: False).
    plot_filename_template : str, optional
        Template for the plot filenames. Use '{envi_basename}' as a placeholder (default: "plot_results_{envi_basename}.png").
    dpi : int, optional
        Resolution of the saved plots in dots per inch (default: 300).

    Returns:
    --------
    masked_raster_paths : list
        List of paths to the saved masked raster files.
    """
    return process_all_envi_files_with_mask(
        directory=folder_name,
        geojson_path=polygon_layer,
        raster_crs_override=raster_crs_override,
        polygons_crs_override=polygons_crs_override,
        output_masked_suffix=output_masked_suffix,
        plot_output=plot_output,
        plot_filename_template=plot_filename_template,
        dpi=dpi
    )

import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from spectral import open_image
import psutil
from tqdm import tqdm
from pyproj import CRS


def print_memory_usage(context=""):
    """Prints memory usage to help debug potential crashes."""
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert bytes to GB
    print(f"[DEBUG] Memory usage {context}: {memory:.2f} GB")







pass












