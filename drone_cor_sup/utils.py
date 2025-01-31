from osgeo import gdal, osr
import re
from typing import Union
import h5py
import numpy as np
import os
import ephem
import rasterio
import h5py
# from osgeo import gdal
import seaborn as sns
from pyproj import Proj, transform
import ephem
import subprocess
import json
import glob

import geopandas as gpd
import rasterio.mask
import pandas as pd
from shapely.geometry import box
from shapely.geometry import mapping

from rasterio.transform import Affine

def get_neon_filename(reflectance_tiff_path: str):
    return f"NEON_D13_NIWO_test_{os.path.basename(reflectance_tiff_path)}"

def create_h5_file_from_dict(data, h5_file, group_name="/"):
    """Recursively create groups and datasets in an HDF5 file from a dictionary."""
    for key, value in data.items():
        if isinstance(value, dict):
            subgroup_name = f"{group_name}/{key}"
            _ = h5_file.create_group(subgroup_name)
            create_h5_file_from_dict(value, h5_file, subgroup_name)
        else:
            dataset_name = f"{group_name}/{key}"
            if os.path.basename(group_name) == 'Coordinate_System':
                dataset = h5_file.create_dataset(dataset_name, data=str(value), dtype=h5py.string_dtype())
            else:
                dataset = h5_file.create_dataset(dataset_name, data=value)
            if key in ['Wavelength', 'FWHM']:
                dataset.attrs['Units'] = 'nanometers'

def remove_non_numeric(input_string):
    """Remove non-numeric characters from a string."""
    return re.sub(r'\D', '', input_string)

def load_data(data: Union[str, np.array], expected_shape):
    """Loads data from file or ensures an array has the expected shape."""
    if isinstance(data, str):
        data_array = gdal.Open(data).ReadAsArray()
    else:
        data_array = np.array(data)
    
    if data_array.shape != expected_shape:
        raise ValueError(f"Data shape mismatch: Expected {expected_shape}, but got {data_array.shape}")
    
    return data_array

def tiff_to_h5(reflectance_tiff_path: str, slope_data: Union[str, np.array],
               aspect_data: Union[str, np.array], path_length_data: Union[str, np.array],
               solar_zenith_data: Union[str, np.array], solar_azimuth_data: Union[str, np.array], 
               wavelengths: list[int], fwhm: list[int], output_dir: str):
    """Converts a TIFF reflectance file into an HDF5 format with metadata and ancillary data."""

    # Open reflectance TIFF
    reflectance_ds = gdal.Open(reflectance_tiff_path)
    reflectance_data = reflectance_ds.ReadAsArray()[:10, :, :]  # Use first 10 bands
    img_shape = reflectance_data.shape[1:]  # (rows, cols)

    # Load ancillary data
    slope_data = load_data(slope_data, img_shape)
    aspect_data = load_data(aspect_data, img_shape)
    path_length_data = load_data(path_length_data, img_shape)
    solar_zenith_data = load_data(solar_zenith_data, img_shape)
    solar_azimuth_data = load_data(solar_azimuth_data, img_shape)

    # Set sensor angles to zero matrices matching image shape
    sensor_zenith_data = np.zeros(img_shape)
    sensor_azimuth_data = np.zeros(img_shape)

    # Extract spatial metadata
    proj = reflectance_ds.GetProjection()
    geo_transform = reflectance_ds.GetGeoTransform()
    
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(proj)
    epsg_code = spatial_ref.GetAuthorityCode(None)

    utm_zone = int(remove_non_numeric(proj.split("UTM zone ")[1].split(",")[0])) if "UTM zone" in proj else None
    map_info_string = f"UTM, 1.000, 1.000, {geo_transform[0]:.3f}, {geo_transform[3]:.3f}, {geo_transform[1]:.3f}, {geo_transform[5]:.3f}, {utm_zone}, North, WGS-84, units=Meters, 0"

    # HDF5 Structure
    h5_data = {
        'NIWO': {
            'Reflectance': {
                'Metadata': {
                    'Coordinate_System': {
                        'Coordinate_System_String': proj,
                        'EPSG Code': epsg_code,
                        'Map_Info': map_info_string,
                        'Proj4': spatial_ref.ExportToProj4()
                    },
                    "Ancillary_Imagery": {
                        "Path_Length": path_length_data,  # Now uses raster-based path length
                        "Slope": slope_data,
                        "Aspect": aspect_data
                    },
                    "Logs": {
                        "Solar_Azimuth_Angle": solar_azimuth_data,
                        "Solar_Zenith_Angle": solar_zenith_data
                    },
                    "to-sensor_Azimuth_Angle": sensor_azimuth_data,
                    "to-sensor_Zenith_Angle": sensor_zenith_data,
                    "Spectral_Data": {
                        'FWHM': fwhm,
                        'Wavelength': wavelengths
                    }
                },
                'Reflectance_Data': np.transpose(reflectance_data, axes=(1, 2, 0))  # Convert to (rows, cols, bands)
            }
        }
    }

    # Save to HDF5
    # h5_filename = output_dir + "/" + 'NEON_D13_NIWO_test' + "_" + os.path.basename(reflectance_tiff_path).replace('.tif', '.h5')
    h5_filename = f"{output_dir}/{get_neon_filename(reflectance_tiff_path).replace('.tif', '.h5')}"
    with h5py.File(h5_filename, "w") as hdf_file:
        create_h5_file_from_dict(h5_data, hdf_file)

    print(f"HDF5 file created: {h5_filename}")
    return h5_filename

def pixel_to_coord(file_path):
     with rasterio.open(file_path) as src:
            band1 = src.read(1)
            print('Band1 has shape', band1.shape)
            height = band1.shape[0]
            width = band1.shape[1]
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            eastings= np.array(xs)
            northings = np.array(ys)
            print('eastings shape', eastings.shape)
            p1 = Proj(src.crs)
            p2 = Proj(proj='latlong', datum='WGS84')
            lons, lats = transform(p1, p2, eastings, northings)
     return lons, lats, cols, rows,

def get_computed_azimuth_zenith(file_path: str, date_time_str: str, 
                                latitudes: Union[float, np.ndarray, list[float]], longitudes: Union[float, np.ndarray, list[float]]):
    with rasterio.open(file_path) as src:
        # Get the affine transform for the raster
        transform = src.transform
        
        # Create arrays to hold the azimuth and zenith values
        azimuth = np.zeros((src.height, src.width), dtype=np.float32)
        zenith = np.zeros((src.height, src.width), dtype=np.float32)
        
        # Assume a date and time for the Sun position calculation
        observer = ephem.Observer()
        observer.date = ephem.date(date_time_str)
        
        # Iterate over each pixel in the raster
        for row in range(latitudes.shape[0]):
            for col in range(latitudes.shape[1]):
                #lon, lat = pixel_to_coord(transform, row, col)
                observer.lat, observer.lon = latitudes[row,col], longitudes[row,col]
                
                sun = ephem.Sun(observer)
                
                # Convert azimuth and altitude (zenith angle is 90 - altitude) to degrees
                az = np.degrees(sun.az)
                #az = sun.az
                alt = np.degrees(sun.alt)
                zen = 90 - sun.alt
                
                azimuth[row, col] = az
                zenith[row, col] = zen
        return azimuth, zenith
    
def process_hdf5_with_neon2envi(image_path, neon_script_path, output_dir):
    """
    Runs the modified neon_script_path(neon2envi2_generic.py) script.
    """
    # Ensure we're using the correct Python script
    neon_script = os.path.abspath(neon_script_path)  # Get absolute path
    print(f"Using script: {neon_script}")

    # Define the command
    command = [
        "/opt/conda/envs/macrosystems/bin/python", neon_script,
        "-anc",
        "--images", image_path,
        "--output_dir", output_dir
    ]

    print("Executing command:", " ".join(command))  # Debugging line

    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                print(line, end='')  # Print output as it's received
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

def brdf_topo_correct_image(image_correction_script, json_path):
    """
    Runs the image_correction_script script.
    """
    # Ensure we're using the correct Python script
    py_script = os.path.abspath(image_correction_script)  # Get absolute path
    print(f"Using script: {py_script}")

    # Define the command
    command = [
        "/opt/conda/envs/macrosystems/bin/python", py_script,
        json_path
    ]

    print("Executing command:", " ".join(command))  # Debugging line

    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                print(line, end='')  # Print output as it's received
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

def generate_config_files(directory, bad_bands=None, file_type='envi', num_cpus=1):
    """
    Generate JSON configuration files for image processing.
    
    Parameters:
        directory (str): The directory containing the main image and ancillary files.
        bad_bands (list, optional): List of bad bands to exclude. Defaults to an empty list.
        file_type (str, optional): Type of file. Defaults to 'envi'.
        num_cpus (int, optional): Number of CPUs to use. Defaults to 1.
    
    Returns:
        List of generated config file paths.
    """

    if bad_bands is None:
        bad_bands = []

    main_image_name = os.path.basename(directory)
    main_image_file = os.path.join(directory, main_image_name)

    # Find ancillary files
    anc_files_pattern = os.path.join(directory, "*_ancillary*")
    anc_files = sorted(glob.glob(anc_files_pattern))

    aviris_anc_names = ['path_length', 'sensor_az', 'sensor_zn', 'solar_az', 'solar_zn',  'slope', 'aspect', 'phase', 'cosine_i']
    suffix_labels = ["reflectance", "anc"]  # Define suffixes for different types of files

    generated_files = []

    # Loop through each ancillary file and create a separate config file
    for i, anc_file in enumerate(anc_files):
        if i == 0:
            suffix_label = f"_reflectance"
        elif i == 1:
            suffix_label = f"_anc"
        else:
            suffix_label = f"_{i}"  # Fallback for unexpected 'i' values
        
        config_dict = {}

        config_dict['bad_bands'] = bad_bands
        config_dict['file_type'] = file_type
        config_dict["input_files"] = [main_image_file]
        
        config_dict["anc_files"] = {
            main_image_file: dict(zip(aviris_anc_names, [[anc_file, a] for a in range(len(aviris_anc_names))]))
        }

        # Export settings
        config_dict['export'] = {}
        config_dict['export']['coeffs'] = True
        config_dict['export']['image'] = True
        config_dict['export']['masks'] = True
        config_dict['export']['subset_waves'] = []
        config_dict['export']['output_dir'] = os.path.join(directory)
        config_dict['export']["suffix"] = suffix_label

        # Detailed settings for export options, TOPO and BRDF corrections
        # These settings include parameters like the type of correction, calculation and application of masks, 
        # coefficients to be used, output directory, suffixes for output files, and various specific parameters 
        # for TOPO and BRDF correction methods.
        # Input format: Nested dictionaries with specific keys and values as per the correction algorithm requirements
        # Example settings include:
        # - 'export': Dictionary of export settings like coefficients, image, masks, output directory, etc.
        # - 'topo': Dictionary of topographic correction settings including types, masks, coefficients, etc.
        # - 'brdf': Dictionary of BRDF correction settings including solar zenith type, geometric model, volume model, etc.

        # Additional settings can be added as needed for specific correction algorithms and export requirements.

        # TOPO Correction options
        config_dict["corrections"] = ['topo','brdf']
        config_dict["topo"] =  {}
        config_dict["topo"]['type'] = 'scs+c'
        config_dict["topo"]['calc_mask'] = [["ndi", {'band_1': 862,'band_2': 668,
                                                    'min': 0.1,'max': 1.0}],
                                            ['ancillary',{'name':'slope',
                                                        'min': np.radians(5),'max':'+inf' }],
                                            ['ancillary',{'name':'cosine_i',
                                                        'min': 0.12,'max':'+inf' }]]

        config_dict["topo"]['apply_mask'] = [["ndi", {'band_1': 862,'band_2': 668,
                                                    'min': 0.1,'max': 1.0}],
                                            ['ancillary',{'name':'slope',
                                                        'min': np.radians(5),'max':'+inf' }],
                                            ['ancillary',{'name':'cosine_i',
                                                        'min': 0.12,'max':'+inf' }]]
        config_dict["topo"]['c_fit_type'] = 'nnls'


        # config_dict["topo"]['type'] =  'precomputed'
        # config_dict["brdf"]['coeff_files'] =  {}

        # BRDF Correction options
        config_dict["brdf"] = {}
        config_dict["brdf"]['solar_zn_type'] ='scene'
        config_dict["brdf"]['type'] = 'flex'
        config_dict["brdf"]['grouped'] = True
        config_dict["brdf"]['sample_perc'] = 0.1
        config_dict["brdf"]['geometric'] = 'li_dense_r'
        config_dict["brdf"]['volume'] = 'ross_thick'
        config_dict["brdf"]["b/r"] = 10  #these may need updating. These constants pulled from literature. 
        config_dict["brdf"]["h/b"] = 2  # These may need updating. These contanstants pulled from literature.
        config_dict["brdf"]['interp_kind'] = 'linear'
        config_dict["brdf"]['calc_mask'] = [["ndi", {'band_1': 862, 'band_2': 668, 'min': 0.1, 'max': 1.0}]]
        config_dict["brdf"]['apply_mask'] = [["ndi", {'band_1': 862, 'band_2': 668, 'min': 0.1, 'max': 1.0}]]
        config_dict["brdf"]['diagnostic_plots'] = True
        config_dict["brdf"]['diagnostic_waves'] = [475, 862, 705, 668]

        # ## Flex dynamic NDVI params
        config_dict["brdf"]['bin_type'] = 'dynamic'
        config_dict["brdf"]['num_bins'] = 25
        config_dict["brdf"]['ndvi_bin_min'] = 0.05
        config_dict["brdf"]['ndvi_bin_max'] = 1.0
        config_dict["brdf"]['ndvi_perc_min'] = 10
        config_dict["brdf"]['ndvi_perc_max'] = 95

        # Define the number of CPUs to be used (considering the number of image-ancillary pairs)
        config_dict['num_cpus'] = num_cpus

        # Output path for configuration file
        # Assuming you want to include the suffix in the filename:
        suffix = config_dict['export']["suffix"]
        output_dir = config_dict['export']['output_dir']
        config_file_name = f"{suffix}.json"
        config_file_path = os.path.join(output_dir, config_file_name)

        config_dict["resample"]  = False
        config_dict["resampler"]  = {}
        config_dict["resampler"]['type'] =  'cubic'
        config_dict["resampler"]['out_waves'] = []
        config_dict["resampler"]['out_fwhm'] = []


        config_filename = f"{main_image_name}_config_{suffix_label}.json"
        config_file_path = os.path.join(directory, config_filename)

        with open(config_file_path, 'w') as outfile:
            json.dump(config_dict, outfile, indent=3)

        generated_files.append(config_file_path)

    return generated_files

def check_and_reproject(geojson_path, raster_path):
    """
    Checks if the GeoJSON polygons and the ENVI raster have the same CRS.
    If not, reprojects the polygons to match the raster's CRS.
    """
    # Load GeoJSON
    polygons = gpd.read_file(geojson_path)

    # Load raster CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    # Check if CRS matches
    if polygons.crs != raster_crs:
        print(f"🔄 Reprojecting GeoJSON from {polygons.crs} to {raster_crs}")
        polygons = polygons.to_crs(raster_crs)
    else:
        print("✅ GeoJSON and Raster have the same CRS")

    return polygons

def extract_pixel_reflectance(geojson_path, raster_path, output_csv):
    """
    Extracts reflectance values for each pixel inside each polygon and saves to CSV.
    Column headers use actual wavelengths instead of Band_1, Band_2, etc.
    """
    # Reproject GeoJSON if needed
    polygons = check_and_reproject(geojson_path, raster_path)

    # Open raster and get metadata
    with rasterio.open(raster_path) as src:
        wavelengths = src.descriptions  # Get actual wavelengths (e.g., ["444nm", "475nm", ...])
        raster_crs = src.crs
        raster_bounds = src.bounds
        raster_transform = src.transform  # Affine transform for pixel-to-geo mapping

    # Convert wavelengths to valid column names
    wavelengths = [w.replace(" ", "_") for w in wavelengths]  # Remove spaces if any

    # Convert raster bounds to a polygon
    raster_extent = box(*raster_bounds)

    # Filter polygons that are within the raster extent
    polygons = polygons[polygons.geometry.intersects(raster_extent)]
    
    # If no polygons remain, exit
    if polygons.empty:
        print("❌ No polygons found within the raster extent!")
        return

    print(f"✅ {len(polygons)} polygons found within raster extent.")

    # Prepare a list to store extracted data
    extracted_data = []

    # Loop through each polygon
    for poly_idx, polygon in polygons.iterrows():
        polygon_id = polygon.get("OBJECTID", poly_idx)  # Use an ID field if available

        # Mask the raster to extract pixel values within the polygon
        with rasterio.open(raster_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, [mapping(polygon.geometry)], crop=True)
            out_image = out_image.astype(np.float32)  # Convert to float for precision

            # Get pixel coordinates
            num_rows, num_cols = out_image.shape[1], out_image.shape[2]
            x_coords = np.arange(num_cols) * out_transform[0] + out_transform[2]
            y_coords = np.arange(num_rows) * out_transform[4] + out_transform[5]

            # Loop over each pixel
            pixel_id = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    # Extract reflectance for all bands
                    reflectance_values = out_image[:, i, j]

                    # Check if pixel is valid (not nodata)
                    if np.any(reflectance_values > 0):  # Assuming negative values are NoData
                        row = {
                            "Polygon_ID": polygon_id,
                            "Pixel_ID": f"{polygon_id}_{pixel_id}",  # Unique ID for each pixel
                            "X_Coordinate": x_coords[j],
                            "Y_Coordinate": y_coords[i],
                            **{wavelengths[b]: reflectance_values[b] for b in range(len(wavelengths))}
                        }
                        extracted_data.append(row)
                        pixel_id += 1  # Increment pixel counter

    # Convert to DataFrame
    df = pd.DataFrame(extracted_data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"📂 Reflectance data saved to: {output_csv}")

def get_tiff_transform(tiff_path):
    """
    Extract the transform (Affine) from a TIFF file.

    Parameters:
        tiff_path (str): Path to the TIFF file.

    Returns:
        Affine: The geotransformation of the TIFF file.
    """
    with rasterio.open(tiff_path) as src:
        return src.transform

def flip_envi_and_preserve_metadata(envi_path, output_path, tiff_transform):
    """
    Flip an ENVI image vertically, update its geotransformation, and preserve all metadata.
    
    Parameters:
        envi_path (str): Path to the ENVI file.
        output_path (str): Path to save the corrected ENVI file.
        tiff_transform (Affine): The correct transformation from the TIFF file.
    """
    with rasterio.open(envi_path) as src_envi:
        # Read ENVI data and metadata
        envi_data = src_envi.read()
        envi_meta = src_envi.meta.copy()
        
        # Extract additional metadata like wavelengths
        wavelengths = src_envi.descriptions  # Band descriptions, e.g., wavelengths
        wavelengths_units = src_envi.tags().get("wavelength units", "nanometers")  # Default to nanometers

    # Check if ENVI needs flipping (y-resolution positive)
    if envi_meta['transform'][5] > 0:
        print("Flipping ENVI image to match TIFF orientation...")
        # Flip the data vertically
        flipped_data = np.flip(envi_data, axis=1)

        # Update the transformation to match the TIFF file
        new_transform = Affine(
            tiff_transform.a, tiff_transform.b, tiff_transform.c,
            tiff_transform.d, tiff_transform.e, tiff_transform.f
        )
        envi_meta.update({"transform": new_transform})
    else:
        flipped_data = envi_data  # No flipping needed
        print("ENVI image does not need flipping.")

    # Save the corrected ENVI file
    with rasterio.open(output_path, "w", **envi_meta) as dst:
        dst.write(flipped_data)

        # Add wavelengths back to metadata
        if wavelengths:
            dst.descriptions = wavelengths

        # Add wavelength units if available
        if wavelengths_units:
            dst.update_tags(**{"wavelength units": wavelengths_units})

    print(f"Corrected ENVI file saved to {output_path} with preserved metadata.")

def fix_envi_orientation(envi_path, output_path):
    """
    Reflip the ENVI image vertically while preserving its corrected spatial position.
    
    Parameters:
        envi_path (str): Path to the ENVI file.
        output_path (str): Path to save the final corrected ENVI file.
    """
    with rasterio.open(envi_path) as src_envi:
        # Read ENVI data and metadata
        envi_data = src_envi.read()
        envi_meta = src_envi.meta.copy()
        
        # Extract additional metadata like wavelengths
        wavelengths = src_envi.descriptions  # Band descriptions, e.g., wavelengths
        wavelengths_units = src_envi.tags().get("wavelength units", "nanometers")  # Default to nanometers

        
        # envi_data = src.read()
        # envi_meta = src.meta.copy()

        # Reflip the ENVI image vertically
        print("Reflipping ENVI image to restore correct orientation...")
        reflipped_data = np.flip(envi_data, axis=1)  # Flip along the y-axis (rows)

    # Save the corrected ENVI file
    with rasterio.open(output_path, "w", **envi_meta) as dst:
        dst.write(reflipped_data)

        # Add wavelengths back to metadata
        if wavelengths:
            dst.descriptions = wavelengths

        # Add wavelength units if available
        if wavelengths_units:
            dst.update_tags(**{"wavelength units": wavelengths_units})

    print(f"Corrected ENVI file saved to {output_path} with preserved metadata.")