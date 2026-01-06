# DEPRECATED: We no longer export GeoTIFF deliverables.
# Final products are ENVI (.img/.hdr) plus optional .parquet.
# This file has been staged for removal.
import collections
from typing import List, Union

import requests
import zipfile

from rasterio.merge import merge
from scipy.optimize import nnls
import pandas as pd
import matplotlib.pyplot as plt
from unmixing.el_mesma import MesmaCore, MesmaModels
import itertools
import geopandas as gpd
from spectral.io import envi
from tqdm import tqdm
from rasterio.transform import Affine
from shapely.geometry import Point
from pathlib import Path

import glob
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from spectralbridge.file_types import MaskedSpectralCSVFile, EndmembersCSVFile, \
    UnmixingModelBestTIF, UnmixingModelFractionsTIF, UnmixingModelRMSETIF


PROJ_DIR = os.path.dirname(os.path.dirname(__file__))

def download_ecoregion():
	# Path where we expect the shapefile
	ecoregion_download = os.path.join(PROJ_DIR, 'data', 'Ecoregion', 'us_eco_l3.shp')

	os.makedirs(os.path.dirname(ecoregion_download), exist_ok=True)

	# Check if it already exists
	if not os.path.exists(ecoregion_download):
		# URL to download from
		url = "https://dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/us/us_eco_l3.zip"
		zip_path = "Ecoregion.zip"

		# Download the file
		print(f"Downloading {url}...")
		with requests.get(url, stream=True) as r:
			r.raise_for_status()
		with open(zip_path, 'wb') as f:
			for chunk in r.iter_content(chunk_size=8192):
				f.write(chunk)

		# Unzip into the correct folder
		os.makedirs(os.path.dirname(ecoregion_download), exist_ok=True)
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(os.path.join('data', 'Ecoregion'))

		# Delete the zip file
		os.remove(zip_path)

	# Check that the file now exists
	assert os.path.exists(ecoregion_download), f"Failed to find {ecoregion_download} after extraction."


def ies_to_envi(signatures, ies_indices: List[int],
                sli_path: str = 'signatures.sli',
                hdr_path: str = 'signatures.hdr'):
    """
    Export a subset of the signatures DataFrame to an ENVI spectral library (.sli/.hdr),
    selecting only the rows whose indices are in ies_indices.

    :param signatures: pandas.DataFrame containing spectral and metadata columns
    :param ies_indices: list of integer row indices to include in the output
    :param sli_path: output path for the .sli file
    :param hdr_path: output path for the .hdr header file
    """
    # 1) subset the DataFrame to only the IES-selected rows
    subset = signatures.iloc[ies_indices]

    # 2) identify spectral band columns
    band_cols = [c for c in subset.columns if c.startswith('Masked_band_')]

    # 3) extract spectral data as float32
    data = subset[band_cols].to_numpy(dtype=np.float32)

    # 4) extract class labels to annotate each spectrum
    classes = subset['cover_category'].astype(str).tolist()

    # 5) write the binary spectral library (.sli)
    data.tofile(sli_path)

    # 6) build and write the ENVI header (.hdr)
    hdr = {
        'samples':      data.shape[1],
        'lines':        data.shape[0],
        'bands':        1,
        'data type':    4,
        'interleave':   'bil',
        'byte order':   0,
        'band names':   band_cols,
        'spectra names': classes,
    }
    envi.write_envi_header(hdr_path, hdr)



def mask_band(band_data, transform, geometries, crs):
    with rasterio.Env():
        # Create a dummy rasterio dataset in memory
        with rasterio.MemoryFile() as memfile:
            profile = {
                'driver': 'GTiff',
                'height': band_data.shape[0],
                'width': band_data.shape[1],
                'count': 1,
                'dtype': band_data.dtype,
                'crs': crs,
                'transform': transform,
            }
            with memfile.open(**profile) as dataset:
                dataset.write(band_data, 1)
                out_image, out_transform = mask(dataset, geometries, crop=True)
    return out_image[0]  # because mask returns (bands, height, width)


def read_landsat_data(landsat_file: str, geometries):
    """
    Reads a single multi‐band (stacked) Landsat GeoTIFF, clips it to the input geometries,
    applies the raster’s nodata mask (converting nodata to np.nan), and returns:
      • clipped_stack: numpy.ndarray of shape (bands, H_clip, W_clip)
      • nan_max: float, the maximum over all non‐NaN pixels in the clipped stack

    Parameters
    ----------
    landsat_file : str
        Path to a multi‐band GeoTIFF (e.g. one of your "<tile>_clipped.tif" files).
    geometries : GeoDataFrame
        A GeoDataFrame containing one or more polygons. These will be reprojected
        to match the raster’s CRS before masking.
    """
    # 1) Open the stacked GeoTIFF and read metadata
    with rasterio.open(landsat_file) as src:
        # Read all bands into a 3D array of shape (bands, H, W), cast to float32
        full_stack = src.read().astype(np.float32)
        src_crs = src.crs
        src_transform = src.transform
        nodata_val = src.nodata

        # Reproject geometries to match the raster’s CRS
        geoms_proj = geometries.to_crs(src_crs)
        shapes = [mapping(geom) for geom in geoms_proj.geometry]

        # Clip (mask) the raster to those shapes; crop=True returns only the minimal window
        clipped_stack, clipped_transform = mask(
            dataset=src,
            shapes=shapes,
            crop=True,
            nodata=nodata_val,
            filled=True
        )
        # clipped_stack has shape (bands, H_clip, W_clip)

    # 2) Convert the nodata pixels to np.nan (only if nodata is defined)
    if nodata_val is not None:
        clipped_stack[clipped_stack == nodata_val] = np.nan

    # 3) Compute the maximum reflectance (ignoring NaNs)
    nan_max = float(np.nanmax(clipped_stack))

    return clipped_stack, nan_max


def read_landsat_data_with_transform(landsat_dir: str, geometries):
    """
    For each tile under `landsat_dir`, this function:
      1. Finds the date subfolder whose QA_PIXEL has the most “clear” pixels (bit 6 == 1).
      2. Reads & clips that QA_PIXEL to `geometries`, producing a boolean mask of “valid” pixels.
      3. Reads & clips each SR_B1…SR_B7 band to the same window, then applies the QA mask
         (setting non‐valid pixels to np.nan).
      4. Stacks the 7 clipped & masked bands into a single array of shape (7, H_clip, W_clip).
      5. Writes that 7‐band stack to a GeoTIFF named `<tile_name>_clipped.tif` inside the tile’s folder.
    """
    # Loop over each “tile” folder (e.g. "039030", "040026", …)
    for tile_name in sorted(os.listdir(landsat_dir)):
        tile_path = os.path.join(landsat_dir, tile_name)
        if not os.path.isdir(tile_path):
            continue

        # Step 1: find all date subfolders under this tile
        date_folders = sorted([
            d for d in os.listdir(tile_path)
            if os.path.isdir(os.path.join(tile_path, d))
        ])
        if not date_folders:
            continue

        best_date = None
        best_valid_count = -1
        best_qa_path = None

        # Step 2: For each date, open QA_PIXEL and count how many “clear” (bit 6 == 1) pixels
        for date_name in date_folders:
            date_path = os.path.join(tile_path, date_name)
            qa_files = glob.glob(os.path.join(date_path, "*QA_PIXEL.TIF"))
            if len(qa_files) != 1:
                # Skip if no QA_PIXEL or multiple QA_PIXEL
                continue
            qa_path = qa_files[0]
            with rasterio.open(qa_path) as qa_src:
                qa_data = qa_src.read(1)  # uint16 array
                # "Clear" pixel if bit 6 == 1
                clear_mask = ((qa_data >> 6) & 1) == 1
                valid_count = int(np.count_nonzero(clear_mask))

            if valid_count > best_valid_count:
                best_valid_count = valid_count
                best_date = date_name
                best_qa_path = qa_path

        # If no valid date found, skip this tile
        if best_date is None:
            continue

        # Step 3: Clip the chosen QA_PIXEL to geometries → get both array and transform
        with rasterio.open(best_qa_path) as qa_src:
            qa_crs = qa_src.crs
            # Reproject geometries to match QA’s CRS
            geoms_proj = geometries.to_crs(qa_crs)
            shapes = [mapping(g) for g in geoms_proj.geometry]

            # mask(..., crop=True) returns (clipped_array, clipped_transform)
            qa_clipped, qa_transform = mask(
                qa_src, shapes, crop=True, nodata=0, filled=True
            )
            # qa_clipped has shape (1, H_clip, W_clip)
            qa_clipped = qa_clipped[0]

        # Build boolean “clear pixel” mask from clipped QA:
        valid_mask_clipped = ((qa_clipped >> 6) & 1) == 1

        # Step 4: For SR_B1…SR_B7, clip & apply QA mask
        band_arrays = []
        for b in range(1, 8):
            pattern = os.path.join(
                tile_path, best_date, f"*SR_B{b}.TIF"
            )
            band_files = glob.glob(pattern)
            if len(band_files) != 1:
                raise RuntimeError(
                    f"Expected exactly one SR_B{b}.TIF in {tile_name}/{best_date}, found {band_files}"
                )
            band_path = band_files[0]

            with rasterio.open(band_path) as band_src:
                # Clip this band to the same shapes; use crop=True so the window matches QA’s
                band_clipped, _ = mask(
                    band_src,
                    shapes,
                    crop=True,
                    nodata=band_src.nodata,
                    filled=True
                )
                band_clipped = band_clipped[0].astype(np.float32)

            # Set non‐valid (cloudy) pixels to np.nan
            band_clipped[~valid_mask_clipped] = np.nan
            band_arrays.append(band_clipped)

        # Step 5: Stack the 7 bands into one array of shape (7, H_clip, W_clip)
        landsat_stack = np.stack(band_arrays, axis=0)
        nan_max = float(np.nanmax(landsat_stack))

        # Build metadata for writing the 7-band GeoTIFF
        # Use the QA’s transform & CRS (all bands share them)
        h_clip, w_clip = landsat_stack.shape[1:]
        out_meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 7,                # B1…B7
            "crs": qa_crs,
            "transform": qa_transform,
            "height": h_clip,
            "width": w_clip,
            # If you want compression, add e.g. "compress": "lzw"
        }

        # Step 6: Write the output file inside the tile’s folder
        out_filename = f"{tile_name}_clipped.tif"
        out_path = os.path.join(tile_path, out_filename)
        with rasterio.open(out_path, "w", **out_meta) as dst:
            for idx in range(7):
                dst.write(landsat_stack[idx], idx + 1)

        print(f"Wrote {out_path}  (max reflectance: {nan_max})")

    # Function writes files to disk and does not return anything
    return


def write_to_raster(tile_results):
    band_tiles = {b: [] for b in range(7)}  # 0→B1, 1→B2, …, 6→B7

    for tile_name, info in tile_results.items():
        stack = info["stack"]         # shape (7, H_clip, W_clip)
        tf = info["transform"]        # affine.Affine for that clipped window
        crs = info["crs"]             # common CRS for all tiles

        # For each of the 7 bands, append (array, transform).
        # Note: merge expects a list of single‐band 2D arrays, each with its own transform.
        for band_idx in range(7):
            band_tiles[band_idx].append((stack[band_idx], tf))

    # 3. For each band, do a mosaic via rasterio.merge.merge
    merged_bands = []
    merged_transform = None

    for band_idx in range(7):
        # `sources` is a list of (array, transform) → merge will mosaic them.
        sources = band_tiles[band_idx]  # list of (2D array, transform)
        # `merge` returns (mosaic_array, mosaic_transform):
        #    mosaic_array shape = (1, H_out, W_out)   (because each source is 2D)
        mosaic_arr, mosaic_tf = merge(sources)
        # Extract the 2D array
        merged_bands.append(mosaic_arr[0])
        # All bands share the same final mosaic grid, so save the transform once
        if merged_transform is None:
            merged_transform = mosaic_tf

    # 4. Build metadata for writing the final 7‐band GeoTIFF
    #    We assume all tiles shared the same CRS (they should, since they were all L2SP)
    out_crs = crs  # from last tile (all tiles must share CRS)
    out_height, out_width = merged_bands[0].shape

    out_meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 7,                     # 7 bands (B1…B7)
        "crs": out_crs,
        "transform": merged_transform,
        "height": out_height,
        "width": out_width,
        # You can add compression options if desired:
        # "compress": "lzw", "predictor": 2, etc.
    }

    # 5. Write the single output file
    output_path = "landsat_mosaic.tif"
    with rasterio.open(output_path, "w", **out_meta) as dst:
        for idx, band_arr in enumerate(merged_bands, start=1):
            # rasterio bands are 1‐indexed, so write at band=idx
            dst.write(band_arr, idx)

    print(f"Wrote merged mosaic → {output_path}")



def ies_from_library(spectral_library, num_endmembers, initial_selection="dist_mean", stop_threshold=0.01):
    if not isinstance(spectral_library, np.ndarray):
        raise ValueError("spectral_library must be a numpy array.")
    if np.any(~np.isfinite(spectral_library)):
        raise ValueError("spectral_library contains non-finite values.")

    n_samples, n_bands = spectral_library.shape
    if num_endmembers > n_samples:
        raise ValueError("num_endmembers cannot be greater than the number of spectra.")
    if initial_selection not in ["max_norm", "dist_mean", "ppi"]:
        raise ValueError("initial_selection must be 'max_norm' or 'dist_mean'.")

    selected_indices = []
    rmse_history = []
    avg_rmse_history = []
    stop_max_idx = None
    stop_mean_idx = None

    # --- Initial Endmember Selection ---
    if initial_selection == "max_norm":
        norms = np.linalg.norm(spectral_library, axis=1)
        first_idx = np.argmax(norms)
    else:  # "dist_mean"
        mean_spectrum = np.mean(spectral_library, axis=0)
        distances = np.linalg.norm(spectral_library - mean_spectrum, axis=1)
        first_idx = np.argmax(distances)

    selected_indices.append(first_idx)
    E = spectral_library[selected_indices, :]

    # --- Iterative Endmember Addition ---
    for i in range(1, num_endmembers):
        all_rmse = np.full(n_samples, np.nan)

        for j in range(n_samples):
            if j in selected_indices:
                all_rmse[j] = 0
                continue
            target = spectral_library[j]
            abundances, _ = nnls(E.T, target)
            reconstruction = np.dot(E.T, abundances)
            rmse = np.sqrt(np.mean((target - reconstruction) ** 2))
            all_rmse[j] = rmse

        # Candidates are spectra not already selected
        candidates = np.setdiff1d(np.where(np.isfinite(all_rmse))[0], selected_indices)

        if len(candidates) == 0:
            break

        next_idx = candidates[np.argmax(all_rmse[candidates])]
        selected_indices.append(next_idx)
        E = spectral_library[selected_indices, :]

        rmse_history.append(np.nanmax(all_rmse[candidates]))
        avg_rmse_history.append(np.nanmean(all_rmse[candidates]))

        if i >= 2:
            pct_drop_max = (rmse_history[-2] - rmse_history[-1]) / rmse_history[-2]
            pct_drop_mean = (avg_rmse_history[-2] - avg_rmse_history[-1]) / avg_rmse_history[-2]

            if stop_max_idx is None and pct_drop_max <= stop_threshold:
                stop_max_idx = i
            if stop_mean_idx is None and pct_drop_mean <= stop_threshold:
                stop_mean_idx = i

            if stop_max_idx is not None and stop_mean_idx is not None:
                break

    # --- Save RMSE Drop Plots ---
    iterations = np.arange(2, len(rmse_history) + 2)

    plt.figure()
    plt.plot(iterations[1:], np.diff(rmse_history) / np.array(rmse_history[:-1]) * 100, marker='o')
    plt.xlabel('Number of Endmembers')
    plt.ylabel('% Drop in Max RMSE')
    plt.title('Max RMSE % Decrease per Iteration')
    plt.grid(True)
    plt.savefig('max_rmse_percent_drop.png')
    plt.close()

    plt.figure()
    plt.plot(iterations[1:], np.diff(avg_rmse_history) / np.array(avg_rmse_history[:-1]) * 100, marker='o')
    plt.xlabel('Number of Endmembers')
    plt.ylabel('% Drop in Mean RMSE')
    plt.title('Mean RMSE % Decrease per Iteration')
    plt.grid(True)
    plt.savefig('mean_rmse_percent_drop.png')
    plt.close()

    return {
        'endmembers': spectral_library[selected_indices, :],
        'indices': selected_indices,
        'rmse_history': rmse_history,
        'avg_rmse_history': avg_rmse_history,
        'stop_max_idx': stop_max_idx,
        'stop_mean_idx': stop_mean_idx
    }


def build_look_up_table(endmember_classes):
    # Your input
    n_endmembers = len(endmember_classes)
    levels = [n_endmembers]

    # Initialize look-up table
    look_up_table = collections.defaultdict(dict)

    # Get unique class labels
    class_labels = np.unique(endmember_classes)

    # Map each class to the indices of its members
    class_to_indices = {
        label: np.where(endmember_classes == label)[0]
        for label in class_labels
    }

    # Build combinations for each class and level
    for level in levels:
        for class_label, indices in class_to_indices.items():
            if len(indices) >= level:
                combos = list(itertools.combinations(indices, level))
                look_up_table[level][class_label] = np.array(combos, dtype=np.int32)
            else:
                look_up_table[level][class_label] = np.empty((0, level), dtype=np.int32)

    return look_up_table


def main(signatures_paths: Union[str, List[str]], landsat_dir: str, ecoregion_mask: str, output_dir: str = None):
    """
    Main unmixing function.
    
    Parameters:
    - signatures_paths: Either a single path or list of paths to masked spectral CSV files
    - landsat_dir: Directory containing Landsat data
    - ecoregion_mask: Name of ecoregion to use as mask
    - output_dir: Output directory for results (defaults to PROJ_DIR/output)
    """
    download_ecoregion()

    if output_dir is None:
        output_dir = os.path.join(PROJ_DIR, 'output')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ecoregion_download = os.path.join(PROJ_DIR, 'data', 'Ecoregion', 'us_eco_l3.shp')
    ecoregions = gpd.read_file(ecoregion_download)
    geometries = ecoregions[ecoregions['US_L3NAME'] == ecoregion_mask]

    # Get transform and crs from first TIF file
    first_band_path = sorted(glob.glob(os.path.join(landsat_dir, "*SR_B*.TIF")))[0]
    with rasterio.open(first_band_path) as src:
        transform = src.transform
        crs = src.crs

    landsat, max_landsat = read_landsat_data(landsat_dir, geometries)

    # Handle single or multiple signature paths
    if isinstance(signatures_paths, str):
        signatures_paths = [signatures_paths]
    
    # Load and concatenate all signatures
    all_signatures = []
    signature_files = []
    
    for sig_path in signatures_paths:
        # Try to create MaskedSpectralCSVFile from path
        sig_file = MaskedSpectralCSVFile.from_filename(Path(sig_path))
        signature_files.append(sig_file)
        
        signatures_df = pd.read_csv(sig_path)
        all_signatures.append(signatures_df)
    
    # Concatenate all signatures
    signatures = pd.concat(all_signatures, ignore_index=True)
    
    # Use the first signature file for naming outputs
    primary_sig_file = signature_files[0]
    
    spectral_library = signatures.iloc[:, 0:7].to_numpy()

    ies_results = ies_from_library(spectral_library, len(signatures.values), initial_selection='dist_mean')
    indices = ies_results['indices']

    endmember_library = signatures.iloc[indices, [0, 1, 2, 3, 4, 5, 6]].copy()
    
    # Save endmembers with proper naming
    endmembers_file = EndmembersCSVFile.from_signatures_file(primary_sig_file, output_path)
    signatures.iloc[indices, [0, 1, 2, 3, 4, 5, 6, 22]].copy().to_csv(endmembers_file.path)
    
    max_endmember = np.nanmax(endmember_library)

    class_labels = signatures.iloc[indices, 22]
    class_list = np.asarray([str(x).lower() for x in class_labels])
    n_classes = len(np.unique(class_list))
    complexity_level = n_classes + 1

    landsat /= max_landsat
    endmember_library /= max_endmember

    models_object = MesmaModels()
    models_object.setup(class_list)

    for level in range(2, complexity_level):
        models_object.select_level(state=True, level=level)
        for i in np.arange(n_classes):
            models_object.select_class(state=True, index=i, level=level)

    mesma = MesmaCore(n_cores=1)

    # === Chunking here ===
    bands, height, width = landsat.shape
    total_pixels = height * width
    chunk_height = 5
    model_best = np.zeros((n_classes, height, width), dtype=np.uint32)
    model_fractions = np.zeros((n_classes+1, height, width), dtype=np.float32)  # +1 for shade which mesma adds
    model_rmse = np.zeros((height, width), dtype=np.float32)

    for start in tqdm(range(0, height, chunk_height), desc="Processing chunks"):

        chunk = landsat[:, start:start+chunk_height, :]

        best_all, fractions_all, rmse_all, _ = mesma.execute(
            image=chunk,
            library=np.float32(endmember_library).T,
            look_up_table=models_object.return_look_up_table(),
            em_per_class=models_object.em_per_class,
            residual_image=False
        )

        model_best[:, start:start+chunk_height, :] = best_all
        model_fractions[:, start:start+chunk_height, :] = fractions_all
        model_rmse[start:start+chunk_height, :] = rmse_all

    # === Write output arrays as rasters using rasterio ===

    from rasterio.transform import from_origin
    
    # Create file type objects for outputs
    model_best_file = UnmixingModelBestTIF.from_signatures_file(primary_sig_file, output_path)
    model_fractions_file = UnmixingModelFractionsTIF.from_signatures_file(primary_sig_file, output_path)
    model_rmse_file = UnmixingModelRMSETIF.from_signatures_file(primary_sig_file, output_path)
    
    # Write model_best raster (one band per class)
    with rasterio.open(
        model_best_file.path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=n_classes,
        dtype=model_best.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        for i in range(n_classes):
            dst.write(model_best[i, :, :], i + 1)
            dst.set_band_description(i + 1, str(np.unique(class_list)[i]))

    # Write model_fractions raster (one band per class + 1 for shade)
    with rasterio.open(
        model_fractions_file.path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=n_classes + 1,
        dtype=model_fractions.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        for i in range(n_classes):
            dst.write(model_fractions[i, :, :], i + 1)
            dst.set_band_description(i + 1, str(np.unique(class_list)[i]))
        dst.write(model_fractions[-1, :, :], n_classes + 1)
        dst.set_band_description(n_classes + 1, 'shade')

    # Write model_rmse raster
    with rasterio.open(
        model_rmse_file.path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=model_rmse.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(model_rmse, 1)
        dst.set_band_description(1, 'rmse')
    
    print(f"Unmixing complete!")
    print(f"  - Endmembers: {endmembers_file.path}")
    print(f"  - Model best: {model_best_file.path}")
    print(f"  - Model fractions: {model_fractions_file.path}")
    print(f"  - Model RMSE: {model_rmse_file.path}")
