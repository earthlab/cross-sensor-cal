


import geopandas as gpd
import rasterio
from rasterio.mask import mask
import pandas as pd
import numpy as np
import hytools as ht
import numpy as np

import random
import hytools as ht
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.features import rasterize
from shapely.geometry import box
import numpy as np


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

# Define your list of filenames
filenames = [
    "export/resample_landsat5.img",
    "export/resample_landsat7.img",
    "export/resample_landsat8.img",
    "export/resample_landsat9.img",
    "export/ENVI__corrected_0",
    "output/ENVI"
]

pass



from shapely.geometry import box

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

# If module-level execution is needed, guard it:
if __name__ == "__main__":
    filenames = [ ... ]  # Define filenames here
    # Any other code to run when script is executed directly
