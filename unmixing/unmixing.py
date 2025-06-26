import collections
from typing import List

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

import os


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
			zip_ref.extractall('data/Ecoregion')

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


import os
import glob
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

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


def random_unit_vectors(n_vectors, n_bands, seed=None):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n_vectors, n_bands))  # Gaussian sampling
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def ppi(spectral_library, n_samples, n_bands):
    ruv = random_unit_vectors(3000, n_bands)
    ppi_score = np.zeros(n_samples)

    for v in ruv:
        projections = np.dot(v, spectral_library.T)
        max_idx = np.argmax(projections)
        min_idx = np.argmin(projections)
        ppi_score[max_idx] += 1
        ppi_score[min_idx] += 1

    return ppi_score


# --- NFINDR Iterative Selection ---
def nfindr_iterative_selection(spectral_library, max_endmembers=21, max_iterations=3):
    """
    True NFINDR implementation using simplex volume maximization.

    Args:
        spectral_library: (n_samples, n_bands) array of spectral vectors
        max_endmembers: number of endmembers to extract
        max_iterations: number of full replacement passes

    Returns:
        dict containing selected endmembers, their indices, and volume history
    """
    from scipy.spatial import ConvexHull
    import random

    n_samples, n_bands = spectral_library.shape
    # if max_endmembers > n_bands + 1:
    #     raise ValueError("max_endmembers cannot exceed number of bands + 1")

    # Initialize by selecting random unique indices
    selected_indices = random.sample(range(n_samples), max_endmembers)
    current_endmembers = spectral_library[selected_indices, :]
    volume_history = []

    def compute_simplex_volume(endmembers):
        # Subtract mean to ensure correct affine volume
        centered = endmembers - np.mean(endmembers, axis=0)
        try:
            hull = ConvexHull(centered)
            return hull.volume
        except:
            return 0.0

    current_volume = compute_simplex_volume(current_endmembers)
    volume_history.append(current_volume)

    print('Running')
    for _ in range(max_iterations):
        improved = False
        for i in range(max_endmembers):
            for j in range(n_samples):
                if j in selected_indices:
                    continue
                trial_indices = selected_indices.copy()
                trial_indices[i] = j
                trial_endmembers = spectral_library[trial_indices, :]
                volume = compute_simplex_volume(trial_endmembers)
                if volume > current_volume:
                    selected_indices = trial_indices
                    current_volume = volume
                    improved = True
        volume_history.append(current_volume)
    print('Done')

    return {
        'endmembers': spectral_library[selected_indices, :],
        'indices': selected_indices,
        'volume_history': volume_history
    }


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
    elif initial_selection == 'ppi':
        ppi_score = ppi(spectral_library, n_samples, n_bands)
        first_idx = np.argmax(ppi_score)
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


def main(signatures_path: str, landsat_dir: str, ecoregion_mask: str):
    download_ecoregion()

    ecoregion_download = os.path.join(PROJ_DIR, 'data', 'Ecoregion', 'us_eco_l3.shp')
    ecoregions = gpd.read_file(ecoregion_download)
    geometries = ecoregions[ecoregions['US_L3NAME'] == ecoregion_mask]

    # Get transform and crs from first TIF file
    first_band_path = sorted(glob.glob(os.path.join(landsat_dir, "*SR_B*.TIF")))[0]
    with rasterio.open(first_band_path) as src:
        transform = src.transform
        crs = src.crs

    landsat, max_landsat = read_landsat_data(landsat_dir, geometries)

    signatures = pd.read_csv(signatures_path)
    spectral_library = signatures.iloc[:, 0:7].to_numpy()

    ies_results = ies_from_library(spectral_library, len(signatures.values), initial_selection='dist_mean')
    #ies_results = nfindr_iterative_selection(spectral_library, max_iterations=1000)
    indices = ies_results['indices']

    endmember_library = signatures.iloc[indices, [0, 1, 2, 3, 4, 5, 6]].copy()
    signatures.iloc[indices, [0, 1, 2, 3, 4, 5, 6, 22]].copy().to_csv('endmembers.csv')
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
    # Assume transform and crs are still available from read_landsat_data
    # transform and crs are already loaded in the local scope
    # Write model_best raster (one band per class)
    model_best_path = os.path.join(PROJ_DIR, 'output', 'model_best.tif')
    os.makedirs(os.path.dirname(model_best_path), exist_ok=True)
    with rasterio.open(
        model_best_path,
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
    model_fractions_path = os.path.join(PROJ_DIR, 'output', 'model_fractions.tif')
    os.makedirs(os.path.dirname(model_fractions_path), exist_ok=True)
    with rasterio.open(
        model_fractions_path,
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
    model_rmse_path = os.path.join(PROJ_DIR, 'output', 'model_rmse.tif')
    with rasterio.open(
        model_rmse_path,
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


def load_signatures_with_coords(hdr_dir: str) -> gpd.GeoDataFrame:
    """
    Reads every .img, masks invalid pixels, and for each valid pixel
    returns its 7-band spectrum *and* its projected x/y coordinate.
    """
    records = []
    for img_path in sorted(glob.glob(os.path.join(hdr_dir, "*masked_masked.img"))):
        with rasterio.open(img_path) as src:
            arr: np.ndarray = src.read()                # shape (7, H, W)
            transform: Affine = src.transform           # maps col,row to x,y
            mask = np.any(arr == -9999, axis=0)         # True = invalid
            # get all valid row,col indices
            rows, cols = np.where(~mask)
            # for each valid pixel, extract spectrum and compute x,y
            for r, c in zip(rows, cols):
                spectrum = arr[:, r, c]                 # length 7
                x, y = transform * (c, r)              # col,row → x,y
                rec = dict(
                    band_1 = float(spectrum[0]),
                    band_2 = float(spectrum[1]),
                    band_3 = float(spectrum[2]),
                    band_4 = float(spectrum[3]),
                    band_5 = float(spectrum[4]),
                    band_6 = float(spectrum[5]),
                    band_7 = float(spectrum[6]),
                    geometry = Point(x, y)
                )
                records.append(rec)

    # build GeoDataFrame
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=src.crs)
    return gdf


def main_from_images(resampled_dir, landsat_file: str, ecoregion_mask: str):
    # 1. download / read your ecoregions
    download_ecoregion()
    eco_shp = os.path.join(PROJ_DIR, 'data', 'Ecoregion', 'us_eco_l3.shp')
    ecoregions = gpd.read_file(eco_shp)
    geometries = ecoregions[ecoregions['US_L3NAME'] == ecoregion_mask]

    # 2. load the pixel signatures from your ENVI files
    sig_gdf = load_signatures_with_coords(resampled_dir)

    cover_polys = gpd.read_file('data/aop_macrosystems_data_1_7_25.geojson').to_crs(epsg=4326)
    if cover_polys.crs != sig_gdf.crs:
        cover_polys = cover_polys.to_crs(sig_gdf.crs)
    print('A')

    sig_with_cover = gpd.sjoin(
        sig_gdf,
        cover_polys[["cover_category", "geometry"]],
        how="left",
        predicate="within"
    )
    sig_with_cover = sig_with_cover.dropna(subset=["cover_category"]).reset_index(drop=True)
    print('B')
    # Now sig_with_cover has columns band_1...band_7 and cover_category

    # 5. Build your spectral library as before
    spectral_library = sig_with_cover[[f"band_{i}" for i in range(1, 8)]].to_numpy()

    print(len(spectral_library))

    ies_results = ies_from_library(
        spectral_library,
        num_endmembers=spectral_library.shape[0],
        initial_selection='dist_mean'
    )
    indices = ies_results["indices"]
    print("Selected endmember indices:", indices)

    # 6. Extract endmember spectra
    endmember_spectra = spectral_library[indices, :]  # shape (k,7)
    max_endmember = np.nanmax(endmember_spectra)

    # 7. Pull out their cover_category
    endmember_cats = sig_with_cover.iloc[indices]["cover_category"].values

    # 8. Build your final endmember library DataFrame
    df_end = pd.DataFrame(
        endmember_spectra,
        columns=[f"band_{i}" for i in range(1, 8)]
    )
    df_end["cover_category"] = endmember_cats

    endmember_library = df_end

    with rasterio.open(landsat_file) as src:
        transform = src.transform
        crs = src.crs

    # class labels: here we lowercase the array indices;
    # adapt if you have a separate label array
    class_list = df_end["cover_category"].values
    n_classes = len(np.unique(class_list))
    complexity_level = n_classes + 1

    print(class_list)

    # 4. Now read the entire Landsat stack for MESMA
    #    (assuming read_landsat_data returns the full cube and max value)
    landsat, max_landsat = read_landsat_data(landsat_file, geometries)

    # normalize
    landsat = landsat / max_landsat
    print(endmember_library, max_endmember)
    endmember_library_arr = endmember_library[['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_6', 'band_7']].to_numpy()
    endmember_library_arr /= max_endmember

    # 5. setup MESMA models
    models = MesmaModels()
    models.setup(class_list)
    for level in range(2, complexity_level):
        models.select_level(True, level)
        for i in range(n_classes):
            models.select_class(True, i, level)

    mesma = MesmaCore(n_cores=1)

    # 6. run MESMA in chunks (same as your original code)
    bands, height, width = landsat.shape
    chunk_h = 5
    model_best      = np.zeros((n_classes, height, width), dtype=np.uint32)
    model_fractions = np.zeros((n_classes+1, height, width), dtype=np.float32)
    model_rmse      = np.zeros((height, width),        dtype=np.float32)

    for row in tqdm(range(0, height, chunk_h), desc="MESMA chunking"):
        chunk = landsat[:, row:row+chunk_h, :]
        best, fracs, rmse, _ = mesma.execute(
            image=chunk,
            library=endmember_library_arr.T.astype(np.float32),
            look_up_table=models.return_look_up_table(),
            em_per_class=models.em_per_class,
            residual_image=False
        )
        model_best[:, row:row+chunk_h, :]      = best
        model_fractions[:, row:row+chunk_h, :] = fracs
        model_rmse[row:row+chunk_h, :]         = rmse

    from rasterio.transform import from_origin
    # Assume transform and crs are still available from read_landsat_data
    # transform and crs are already loaded in the local scope
    # Write model_best raster (one band per class)
    model_best_path = os.path.join(PROJ_DIR, 'output', 'model_best.tif')
    os.makedirs(os.path.dirname(model_best_path), exist_ok=True)
    with rasterio.open(
            model_best_path,
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
    model_fractions_path = os.path.join(PROJ_DIR, 'output', 'model_fractions.tif')
    os.makedirs(os.path.dirname(model_fractions_path), exist_ok=True)
    with rasterio.open(
            model_fractions_path,
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
    model_rmse_path = os.path.join(PROJ_DIR, 'output', 'model_rmse.tif')
    with rasterio.open(
            model_rmse_path,
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