import os
import glob
import re
from pathlib import Path
from typing import List

import pandas as pd
from shapely.geometry import box
from rasterio.features import rasterize
import rasterio
import geopandas as gpd
from tqdm import tqdm
import numpy as np
from rasterio.crs import CRS

from src.file_types import DataFile, NEONReflectanceENVIFile


def control_function_for_extraction(directory, polygon_path):
    """
    Finds and processes raster files in a directory.
    Processes data in chunks and saves output to CSV.
    """
    raster_paths = find_raster_files_for_extraction(directory)
    print(raster_paths)

    if not raster_paths:
        print(f"[DEBUG] No matching raster files found in {directory}.")
        return

    for raster_path in raster_paths:
        try:
            base_name = os.path.basename(raster_path).replace('.img', '').replace('.tif', '')
            output_csv_name = f"{base_name}_spectral_data.csv"
            output_csv_path = os.path.join(directory, output_csv_name)
            process_raster_in_chunks(raster_path, polygon_path, output_csv_path)
        except Exception as e:
            print(f"[ERROR] Error while processing raster file {raster_path}: {e}")


def find_best_reflectance_files(directory: Path) -> List[DataFile]:
    all_paths = directory.rglob("*.img")
    parsed = []

    for path in all_paths:
        try:
            file = NEONReflectanceENVIFile.from_filename(path)
            parsed.append(file)
        except ValueError:
            continue

    # Group by a unique ID (e.g., site + date + time)
    from collections import defaultdict

    file_groups = defaultdict(list)
    for f in parsed:
        key = (f.domain, f.site, f.date, f.time)
        file_groups[key].append(f)

    # Select best file per group
    selected = []
    for versions in file_groups.values():
        best = sorted(
            versions,
            key=lambda f: (
                not getattr(f, "is_masked", False),   # prefer masked
                f.suffix != "envi",                   # prefer envi
            )
        )[0]
        selected.append(best)

    return selected



def get_crs_from_hdr(hdr_path):
    """
    Reads an ENVI .hdr file and extracts the CRS as a Proj string or EPSG code.

    Parameters:
    - hdr_path (str): Path to the .hdr file.

    Returns:
    - crs (rasterio.crs.CRS or None): CRS object if found, else None.
    """
    try:
        # Open file with 'latin1' encoding and ignore errors to bypass problematic bytes.
        with open(hdr_path, 'r', encoding='latin1', errors='ignore') as f:
            lines = f.readlines()

        for line in lines:
            lower_line = line.lower()
            if "coordinate system string" in lower_line:
                proj_str = re.search(r'coordinate system string = (.*)', line, re.IGNORECASE)
                if proj_str:
                    wkt = proj_str.group(1).strip()
                    if wkt:
                        return CRS.from_wkt(wkt)  # Convert WKT to CRS

            elif "map info" in lower_line:
                map_info = re.search(r'map info = {(.*?)}', line, re.IGNORECASE)
                if map_info:
                    values = map_info.group(1).split(',')
                    try:
                        utm_zone = int(values[7])
                    except (IndexError, ValueError):
                        continue
                    hemisphere = values[8].strip().lower()
                    datum = values[9].strip() if len(values) > 9 else ""
                    if datum == "WGS-84":
                        return CRS.from_epsg(32600 + utm_zone) if hemisphere == "north" else CRS.from_epsg(32700 + utm_zone)

        return None  # Return None if no CRS is found

    except Exception as e:
        print(f"[ERROR] Could not extract CRS from {hdr_path}: {e}")
        return None


def process_raster_in_chunks(raster_path, polygon_path, output_csv_path, chunk_size=100000):
    """
    Processes a raster file in chunks, extracts CRS from an accompanying .hdr file if available,
    assigns CRS to the polygon layer if missing, and writes out spectral data to a CSV file.

    Additional metadata columns added:
    - Raster_File: Name of the raster file.
    - Polygon_File: Name of the polygon file.
    - Chunk_Number: Current chunk index.
    - CRS: Coordinate reference system of the raster.
    - Pixel_ID: Unique pixel identifier computed from the global row and column indices.
    - Pixel_X, Pixel_Y: X and Y coordinates of each pixel.

    Args:
        raster_path (str): Path to the raster file.
        polygon_path (str): Path to the polygon (vector) file.
        output_csv_path (str): Path where the output CSV will be saved.
        chunk_size (int, optional): Number of pixels per chunk. Defaults to 100000.
    """
    hdr_path = raster_path.replace(".img", ".hdr").replace(".tif", ".hdr")

    with rasterio.open(raster_path) as src:
        crs_from_hdr = None

        # Check for .hdr file and extract CRS if available
        if os.path.exists(hdr_path):
            crs_from_hdr = get_crs_from_hdr(hdr_path)

        polygons = gpd.read_file(polygon_path)

        # Assign CRS to polygons if missing
        if polygons.crs is None:
            print(f"[WARNING] {polygon_path} has no CRS. Assigning from .hdr file if available.")
            polygons = polygons.set_crs(crs_from_hdr if crs_from_hdr else "EPSG:4326")

        # If the raster's CRS is missing, assign from .hdr file if available
        if src.crs is None and crs_from_hdr:
            print(f"[INFO] Assigning CRS from .hdr file: {crs_from_hdr}")
            src = src.to_crs(crs_from_hdr)

        # Ensure both raster and polygons have the same CRS
        if polygons.crs != src.crs:
            polygons = polygons.to_crs(src.crs)

        # Rasterize the polygon layer to create an array of polygon IDs
        polygon_values = rasterize(
            [(geom, idx + 1) for idx, geom in enumerate(polygons.geometry)],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="int32"
        ).ravel()

        total_bands = src.count
        height, width = src.height, src.width
        num_chunks = (height * width // chunk_size) + (1 if (height * width) % chunk_size else 0)

        # Determine band prefix based on file naming
        filename = os.path.basename(raster_path)
        if "_masked" in filename:
            band_prefix = "Masked_band_"
        elif "_envi" in filename:
            band_prefix = "ENVI_band_"
        else:
            band_prefix = "Original_band_"
        print(f"[INFO] Processing {filename} as {band_prefix}")
        print(f"[INFO] Processing raster: {raster_path} with {total_bands} bands.")

        # Process raster in chunks
        with tqdm(total=num_chunks, desc=f"Processing {filename}", unit="chunk") as pbar:
            first_chunk = True
            for i in range(num_chunks):
                row_start = (i * chunk_size) // width
                row_end = min(((i + 1) * chunk_size) // width + 1, height)

                # Read a chunk of rows using a window
                data = src.read(window=((row_start, row_end), (0, width)))
                # Reshape data from (bands, rows, cols) to (pixels, bands)
                data_chunk = data.reshape(total_bands, -1).T

                # Generate row and column indices for the chunk
                row_indices, col_indices = np.meshgrid(
                    np.arange(row_start, row_end),
                    np.arange(width),
                    indexing='ij'
                )
                row_indices_flat = row_indices.flatten()
                col_indices_flat = col_indices.flatten()

                # Filter out rows with -9999 values
                valid_mask = ~np.any(data_chunk == -9999, axis=1)
                data_chunk = data_chunk[valid_mask]

                # Filter row and column indices for valid pixels
                valid_rows = row_indices_flat[valid_mask]
                valid_cols = col_indices_flat[valid_mask]

                # Calculate a unique Pixel ID using global row and column indices
                pixel_ids = valid_rows * width + valid_cols

                # Calculate pixel coordinates using the raster's affine transform
                transform = src.transform
                x_coords = transform.a * valid_cols + transform.b * valid_rows + transform.c
                y_coords = transform.d * valid_cols + transform.e * valid_rows + transform.f

                # Get corresponding polygon values for valid pixels
                polygon_chunk = polygon_values[row_start * width:row_end * width][valid_mask]

                # Create DataFrame for the spectral data
                chunk_df = pd.DataFrame(data_chunk, columns=[f'Band_{b + 1}' for b in range(total_bands)])

                # Add extra metadata columns
                chunk_df["Raster_File"] = os.path.basename(raster_path)
                chunk_df["Polygon_File"] = os.path.basename(polygon_path)
                chunk_df["Chunk_Number"] = i
                if src.crs is not None:
                    chunk_df["CRS"] = src.crs.to_string()

                # Add pixel ID and coordinate columns
                chunk_df["Pixel_ID"] = pixel_ids
                chunk_df["Pixel_X"] = x_coords
                chunk_df["Pixel_Y"] = y_coords

                # Add polygon ID column
                chunk_df["Polygon_ID"] = polygon_chunk

                # Rename band columns to include a prefix
                chunk_df.rename(columns={f"Band_{b + 1}": f"{band_prefix}{b + 1}" for b in range(total_bands)},
                                inplace=True)

                # Merge with polygon attributes based on Polygon_ID
                polygon_attributes = polygons.reset_index().rename(columns={'index': 'Polygon_ID'})
                chunk_df = pd.merge(chunk_df, polygon_attributes, on='Polygon_ID', how='left')

                # Write the chunk to CSV (using write mode for first chunk, then append)
                mode = 'w' if first_chunk else 'a'
                chunk_df.to_csv(output_csv_path, mode=mode, header=first_chunk, index=False)

                first_chunk = False
                pbar.update(1)