import os
import glob
import re
from pathlib import Path
from typing import List
from collections import defaultdict
from typing import List, Type

import pandas as pd
from shapely.geometry import box
from rasterio.features import rasterize
import rasterio
import geopandas as gpd
from tqdm import tqdm
import numpy as np
from rasterio.crs import CRS

from src.file_types import DataFile, NEONReflectanceENVIFile, NEONReflectanceBRDFCorrectedENVIFile, \
    NEONReflectanceResampledENVIFile, SpectralDataCSVFile


def control_function_for_extraction(directory, polygon_path):
    """
    Finds and processes raster files in a directory.
    Processes data in chunks and saves output to CSV.
    """
    raster_files = get_all_priority_rasters(directory, 'envi')

    if not raster_files:
        print(f"[DEBUG] No matching raster files found in {directory}.")
        return

    for raster_file in raster_files:
        try:
            spectral_csv_file = SpectralDataCSVFile.from_raster_file(raster_file)
            print(f"[DEBUG] Writing to {spectral_csv_file.path}")
            process_raster_in_chunks(raster_file, polygon_path, spectral_csv_file)
        except Exception as e:
            print(f"[ERROR] Error while processing raster file {raster_file.file_path}: {e}")


def select_best_files(files: List[DataFile]) -> List[DataFile]:
    grouped = defaultdict(list)
    for f in files:
        key = (f.domain, f.site, f.date, f.time)
        grouped[key].append(f)

    selected = []
    for group in grouped.values():
        # Sort using priority: masked + BRDF > masked > unmasked BRDF > unmasked
        best = sorted(
            group,
            key=lambda f: (
                not getattr(f, "is_masked", False),
                "brdf" not in f.path.name.lower(),
                getattr(f, "suffix", "") != "envi"
            )
        )[0]
        selected.append(best)

    return selected


def find_best_resampled_files(directory: Path, suffix: str) -> List[NEONReflectanceResampledENVIFile]:
    all_resampled = NEONReflectanceResampledENVIFile.find_all_sensors_in_directory(directory, suffix)
    return select_best_files(all_resampled)


def get_all_priority_rasters(base_dir: Path, suffix: str = 'envi') -> List[DataFile]:
    # 1. Get BRDF-corrected reflectance files
    brdf_files = NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(base_dir, suffix)

    # 2. Get original reflectance files
    raw_files = NEONReflectanceENVIFile.find_in_directory(base_dir)

    # 3. Combine and select best
    all_candidates = brdf_files + raw_files
    selected_originals = select_best_files(all_candidates)

    # 4. Get best resampled files
    selected_resampled = find_best_resampled_files(base_dir, suffix)

    return selected_originals + selected_resampled



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


def process_raster_in_chunks(raster_file: DataFile, polygon_path: Path, output_csv_file: DataFile, chunk_size=100000):
    """
    Processes a raster file in chunks, intersects pixels with a polygon, and writes extracted
    spectral and spatial data to a CSV file.
    """

    raster_path = raster_file.path
    output_csv_path = output_csv_file.path
    hdr_path = raster_path.with_suffix(".hdr")

    with rasterio.open(raster_path) as src:
        crs_from_hdr = None
        if hdr_path.exists():
            crs_from_hdr = get_crs_from_hdr(hdr_path)

        polygons = gpd.read_file(polygon_path)

        if polygons.crs is None:
            print(f"[WARNING] {polygon_path} has no CRS. Assigning from .hdr file if available.")
            polygons = polygons.set_crs(crs_from_hdr if crs_from_hdr else "EPSG:4326")

        if src.crs is None and crs_from_hdr:
            print(f"[INFO] Assigning CRS from .hdr file: {crs_from_hdr}")
            src = src.to_crs(crs_from_hdr)

        if polygons.crs != src.crs:
            polygons = polygons.to_crs(src.crs)

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

        # Smart prefixing
        if getattr(raster_file, "is_masked", False):
            band_prefix = "Masked_band_"
        elif getattr(raster_file, "suffix", "") == "envi":
            band_prefix = "ENVI_band_"
        else:
            band_prefix = "Original_band_"

        print(f"[INFO] Processing {raster_path.name} with {total_bands} bands as {band_prefix}")

        with tqdm(total=num_chunks, desc=f"Processing {raster_path.name}", unit="chunk") as pbar:
            first_chunk = True
            for i in range(num_chunks):
                row_start = (i * chunk_size) // width
                row_end = min(((i + 1) * chunk_size) // width + 1, height)

                data = src.read(window=((row_start, row_end), (0, width)))
                data_chunk = data.reshape(total_bands, -1).T

                row_indices, col_indices = np.meshgrid(
                    np.arange(row_start, row_end),
                    np.arange(width),
                    indexing='ij'
                )
                row_indices_flat = row_indices.flatten()
                col_indices_flat = col_indices.flatten()

                valid_mask = ~np.any(data_chunk == -9999, axis=1)
                data_chunk = data_chunk[valid_mask]

                valid_rows = row_indices_flat[valid_mask]
                valid_cols = col_indices_flat[valid_mask]
                pixel_ids = valid_rows * width + valid_cols

                transform = src.transform
                x_coords = transform.a * valid_cols + transform.b * valid_rows + transform.c
                y_coords = transform.d * valid_cols + transform.e * valid_rows + transform.f

                polygon_chunk = polygon_values[row_start * width:row_end * width][valid_mask]

                chunk_df = pd.DataFrame(data_chunk, columns=[f'Band_{b + 1}' for b in range(total_bands)])
                chunk_df["Raster_File"] = raster_path.name
                chunk_df["Polygon_File"] = polygon_path
                chunk_df["Chunk_Number"] = i
                chunk_df["Pixel_ID"] = pixel_ids
                chunk_df["Pixel_X"] = x_coords
                chunk_df["Pixel_Y"] = y_coords
                chunk_df["Polygon_ID"] = polygon_chunk
                if src.crs:
                    chunk_df["CRS"] = src.crs.to_string()

                chunk_df.rename(columns={f"Band_{b + 1}": f"{band_prefix}{b + 1}" for b in range(total_bands)},
                                inplace=True)

                polygon_attributes = polygons.reset_index().rename(columns={'index': 'Polygon_ID'})
                chunk_df = pd.merge(chunk_df, polygon_attributes, on='Polygon_ID', how='left')

                mode = 'w' if first_chunk else 'a'
                chunk_df.to_csv(output_csv_path, mode=mode, header=first_chunk, index=False)

                first_chunk = False
                pbar.update(1)