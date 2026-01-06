import os
import glob
from typing import Union

import numpy as np
from shapely.geometry import box

from ._optional import require_geopandas, require_matplotlib_pyplot, require_rasterio

from .file_types import NEONReflectanceENVIFile, NEONReflectanceBRDFCorrectedENVIFile, NEONReflectanceResampledENVIFile


def mask_raster_with_polygons(
    envi_file: Union[NEONReflectanceENVIFile, NEONReflectanceBRDFCorrectedENVIFile, NEONReflectanceResampledENVIFile],
    geojson_path,
    raster_crs_override=None,
    polygons_crs_override=None,
    plot_output=False,  # Disable plotting by default when processing multiple files
    plot_filename=None,  # Not used when plot_output is False
    dpi=300
):
    """
    Masks an ENVI raster using polygons from a GeoJSON file.

    Parameters:
    -----------
    envi_file : DataFile
        ENVI raster file type object.
    geojson_path : str
        Path to the GeoJSON file containing polygons.
    raster_crs_override : str, optional
        CRS to assign to the raster if it's undefined (e.g., 'EPSG:4326').
    polygons_crs_override : str or CRS, optional
        CRS to assign to the polygons if they're undefined.
    output_masked_suffix : str, optional
        Suffix to append to the masked raster filename.
    plot_output : bool, optional
        Whether to generate and save plots of the results.
    plot_filename : str, optional
        Filename for the saved plot.
    dpi : int, optional
        Resolution of the saved plot in dots per inch.

    Raises:
    -------
    FileNotFoundError
        If the ENVI raster or GeoJSON file does not exist.
    ValueError
        If CRS assignments are needed but not provided.

    Returns:
    --------
    masked_raster_path : str
        Path to the saved masked raster file.
    """
    def load_data(envi_path, geojson_path):
        """
        Loads the ENVI raster and GeoJSON polygons.
        """
        rasterio = require_rasterio()
        gpd = require_geopandas()

        if not os.path.exists(envi_path):
            raise FileNotFoundError(f"ENVI raster file not found at: {envi_path}")
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"GeoJSON file not found at: {geojson_path}")

        raster = rasterio.open(envi_path)
        polygons = gpd.read_file(geojson_path)
        return raster, polygons

    def assign_crs(raster, polygons, raster_crs_override=None, polygons_crs_override=None):
        """
        Determines the CRS for raster and polygons without modifying the raster object.
        Returns the raster CRS and the aligned polygons.
        """
        # Handle raster CRS
        if raster.crs is None:
            if raster_crs_override is not None:
                raster_crs = require_rasterio().crs.CRS.from_string(raster_crs_override)
                print(f"Assigned CRS {raster_crs_override} to raster.")
            else:
                raise ValueError("Raster CRS is undefined and no override provided.")
        else:
            raster_crs = raster.crs

        # Handle polygons CRS
        if polygons.crs is None:
            if polygons_crs_override is not None:
                polygons = polygons.set_crs(polygons_crs_override)
                print(f"Assigned CRS {polygons_crs_override} to polygons.")
            else:
                raise ValueError("Polygons CRS is undefined and no override provided.")

        return raster_crs, polygons

    def align_crs(raster_crs, polygons):
        """
        Reprojects polygons to match raster CRS.
        """
        print("Raster CRS:", raster_crs)
        print("Polygons CRS:", polygons.crs)

        if polygons.crs != raster_crs:
            print("Reprojecting polygons to match raster CRS...")
            polygons_aligned = polygons.to_crs(raster_crs)
            print("Reprojection complete.")
        else:
            print("CRS of raster and polygons already match.")
            polygons_aligned = polygons

        return polygons_aligned

    def clip_polygons(raster, polygons_aligned):
        """
        Clips polygons to raster bounds.
        """
        gpd = require_geopandas()

        print("Clipping polygons to raster bounds...")
        raster_bounds_geom = gpd.GeoDataFrame({'geometry': [box(*raster.bounds)]}, crs=raster.crs)
        clipped_polygons = gpd.overlay(polygons_aligned, raster_bounds_geom, how='intersection')
        if clipped_polygons.empty:
            print("No polygons overlap the raster extent.")
        else:
            print(f"Number of Clipped Polygons: {len(clipped_polygons)}")
            print("Clipped Polygons Bounds:", clipped_polygons.total_bounds)
        return clipped_polygons

    def create_mask(raster, polygons):
        """
        Creates a mask where pixels inside polygons are True and outside are False.
        """
        print("Creating mask from polygons...")
        rasterio = require_rasterio()
        mask = rasterio.features.rasterize(
            [(geom, 1) for geom in polygons.geometry],
            out_shape=(raster.height, raster.width),
            transform=raster.transform,
            fill=0,  # Background value
            dtype='uint8',
            all_touched=True  # Include all touched pixels
        )
        mask = mask.astype(bool)
        print(f"Mask created with shape {mask.shape}. Inside pixels: {np.sum(mask)}")

        # Additional debug: Check unique values
        unique_values = np.unique(mask)
        print(f"Unique values in mask: {unique_values}")

        return mask

    def apply_mask(raster, mask):
        """
        Applies the mask to raster data, setting areas outside polygons to nodata.
        """
        raster_data = raster.read()
        nodata_value = raster.nodata if raster.nodata is not None else -9999
        print(f"Using nodata value: {nodata_value}")

        if raster.count > 1:  # For multi-band rasters
            if mask.shape != raster.read(1).shape:
                raise ValueError("Mask shape does not match raster band shape.")
            mask_expanded = np.repeat(mask[np.newaxis, :, :], raster.count, axis=0)
            masked_data = np.where(mask_expanded, raster_data, nodata_value)
        else:  # For single-band rasters
            masked_data = np.where(mask, raster_data[0], nodata_value)

        print(f"Masked data shape: {masked_data.shape}")
        return masked_data

    def save_masked_raster(envi_path: Union[NEONReflectanceENVIFile, NEONReflectanceBRDFCorrectedENVIFile,
                            NEONReflectanceResampledENVIFile], masked_data, nodata, raster):
        """
        Saves the masked raster to a new file.
        """
        rasterio = require_rasterio()
        meta = raster.meta.copy()
        meta.update({
            'dtype': masked_data.dtype,
            'nodata': nodata,
            'count': raster.count if raster.count > 1 else 1
        })

        with rasterio.open(envi_path.masked_path.name, 'w', **meta) as dst:
            if raster.count > 1:
                for i in range(raster.count):
                    dst.write(masked_data[i], i + 1)
            else:
                dst.write(masked_data, 1)
        print(f"Masked raster saved to: {envi_path.masked_path.name}")
        return envi_path.masked_path.name

    def plot_results(raster, masked_data, nodata, clipped_polygons, plot_path):
        """
        Plots the original raster, clipped polygons, clipped polygons on raster, and masked raster.
        Saves the result as a high-resolution PNG and displays the plot.
        """
        print(f"Masked data stats: Min={masked_data.min()}, Max={masked_data.max()}, Unique={np.unique(masked_data)}")

        plt = require_matplotlib_pyplot()
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Original Raster
        original = raster.read(1)
        original = np.ma.masked_equal(original, nodata)
        axes[0, 0].imshow(
            original,
            cmap='gray',
            extent=(
                raster.bounds.left,
                raster.bounds.right,
                raster.bounds.bottom,
                raster.bounds.top
            )
        )
        axes[0, 0].set_title('Original Raster')
        axes[0, 0].axis('off')

        # Clipped Polygons
        clipped_polygons.plot(ax=axes[0, 1], facecolor='none', edgecolor='red')
        axes[0, 1].set_title('Clipped Polygons')
        axes[0, 1].set_xlim(raster.bounds.left, raster.bounds.right)
        axes[0, 1].set_ylim(raster.bounds.bottom, raster.bounds.top)
        axes[0, 1].axis('off')

        # Clipped Polygons on Raster
        axes[1, 0].imshow(
            original,
            cmap='gray',
            extent=(
                raster.bounds.left,
                raster.bounds.right,
                raster.bounds.bottom,
                raster.bounds.top
            )
        )
        clipped_polygons.plot(ax=axes[1, 0], facecolor='none', edgecolor='blue', linewidth=0.5)
        axes[1, 0].set_title('Clipped Polygons on Raster')
        axes[1, 0].set_xlim(raster.bounds.left, raster.bounds.right)
        axes[1, 0].set_ylim(raster.bounds.bottom, raster.bounds.top)
        axes[1, 0].axis('off')

        # Masked Raster
        # Find the first valid band to plot
        valid_band_found = False
        for band_index in range(masked_data.shape[0]):
            band = np.ma.masked_equal(masked_data[band_index], nodata)
            if np.any(band):  # Check if the band has non-nodata values
                print(f"Plotting band {band_index + 1} (valid band found).")
                valid_band_found = True
                axes[1, 1].imshow(
                    band,
                    cmap='Reds',
                    extent=(
                        raster.bounds.left,
                        raster.bounds.right,
                        raster.bounds.bottom,
                        raster.bounds.top
                    )
                )
                axes[1, 1].set_title(f'Masked Raster (Band {band_index + 1})')
                axes[1, 1].axis('off')
                break

        if not valid_band_found:
            print("No valid band found to plot.")
            axes[1, 1].text(
                0.5,
                0.5,
                "No valid band found",
                ha='center',
                va='center',
                transform=axes[1, 1].transAxes
            )

        plt.tight_layout()

        # Save the figure
        print(f"Saving plot to {plot_path}")
        plt.savefig(plot_path, dpi=dpi)  # Save as high-resolution PNG

        if plot_output:
            # Display the plot
            plt.show()
        else:
            plt.close(fig)

    if envi_file.masked_path.exists():
        print("Masked raster already exists; skipping generation.")
        return

    # Start of the masking function logic
    try:
        raster, polygons = load_data(envi_file.file_path, geojson_path)
    except FileNotFoundError as e:
        print(e)
        raise

    try:
        raster_crs, polygons = assign_crs(
            raster,
            polygons,
            raster_crs_override=raster_crs_override,
            polygons_crs_override=polygons_crs_override
        )
    except ValueError as e:
        print(e)
        raster.close()
        raise

    polygons_aligned = align_crs(raster_crs, polygons)
    clipped_polygons = clip_polygons(raster, polygons_aligned)

    if clipped_polygons.empty:
        print("No polygons overlap the raster extent. Skipping masking.")
        raster.close()
        return None

    # Check if raster has a valid transform
    if not raster.transform.is_identity:
        print("Raster has a valid geotransform.")
    else:
        print("Raster has an identity transform. Geospatial operations may be invalid.")
        # Depending on your data, you might want to skip or handle differently
        # For now, we'll proceed but be aware that spatial alignment may be incorrect

    mask = create_mask(raster, clipped_polygons)
    masked_data = apply_mask(raster, mask)

    # Handle rasters with no geotransform by informing the user
    if raster.transform.is_identity:
        print("Warning: Raster has an identity transform. Masked data may not be georeferenced correctly.")

    masked_raster_path = save_masked_raster(
        envi_file,
        masked_data,
        raster.nodata,
        raster
    )

    # Plot results if enabled
    if plot_output and plot_filename:
        plot_results(
            raster,
            masked_data,
            raster.nodata,
            polygons_aligned,
            clipped_polygons,
            plot_path=plot_filename
        )

    # Close the raster dataset
    raster.close()

    return masked_raster_path

def find_raster_files(directory):
    """
    Finds raster files ending with '_reflectance', '_envi', or '.img' while excluding
    files ending in '.json', '.hdr', or containing '_mask' or '_ancillary'.
    """
    pattern = os.path.join(directory, '**')
    all_files = glob.glob(pattern, recursive=True)

    # Filter the files based on naming conventions
    filtered_files = [
        file for file in all_files
        if (
            ('_reflectance' in os.path.basename(file) or
             '_envi' in os.path.basename(file) or
             file.endswith('.img')) and
            not file.endswith('.json') and
            not file.endswith('.hdr') and
            not file.endswith('.png') and
            '_mask' not in file and
            '_ancillary' not in file
        ) and not os.path.isdir(file)
    ]

    return filtered_files