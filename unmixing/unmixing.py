import collections
import os
import requests
import zipfile
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
import numpy as np
from scipy.optimize import nnls
import pandas as pd
import matplotlib.pyplot as plt
from unmixing.el_mesma import MesmaCore
import itertools
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import mapping


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


def read_landsat_data(landsat_dir: str, geometries):
    '''
    # TODO: Use scale function rather than dividing by 10
            Need to remove NAs
                Need to weed out low-quality bands
                    Use Landsat on Cyverse rather than downloading from Earth Explorer
    Returns:
    '''
    # 1. Find all Landsat bands matching "*SR_B*.TIF"
    all_landsat_bands = sorted(glob.glob(os.path.join(landsat_dir, "*SR_B*.TIF")))

    # 2. Read each band into a rasterio dataset and add 0 (convert to float)
    def read_raster(path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
        return data

    def read_raster_full(path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
        return data, transform, crs

    # Read bands individually
    L1, transform, crs = read_raster_full(all_landsat_bands[0])
    L2, transform, crs = read_raster_full(all_landsat_bands[1])
    L3, transform, crs = read_raster_full(all_landsat_bands[2])
    L4, transform, crs = read_raster_full(all_landsat_bands[3])
    L5, transform, crs = read_raster_full(all_landsat_bands[4])
    L6, transform, crs = read_raster_full(all_landsat_bands[5])

    geometries = geometries.to_crs(crs)
    geometries = [mapping(geom) for geom in geometries.geometry]

    L1_clipped = mask_band(L1, transform, geometries, crs)
    L2_clipped = mask_band(L2, transform, geometries, crs)
    L3_clipped = mask_band(L3, transform, geometries, crs)
    L4_clipped = mask_band(L4, transform, geometries, crs)
    L5_clipped = mask_band(L5, transform, geometries, crs)
    L6_clipped = mask_band(L6, transform, geometries, crs)

    # 3. Stack them into a 3D array (bands first)
    landsat_spatRas_scale = np.stack([L1_clipped, L2_clipped, L3_clipped, L4_clipped, L5_clipped, L6_clipped], axis=0)

    # 4. Check the range of band 3
    nan_max = np.nanmax(landsat_spatRas_scale)
    print("Maximum relfectance:", nan_max)

    # 5. Scale the data by dividing by 10
    landsat_spatRas = landsat_spatRas_scale

    return landsat_spatRas, nan_max


def ies_from_library(spectral_library, num_endmembers, initial_selection="dist_mean", stop_threshold=0.01):
    if not isinstance(spectral_library, np.ndarray):
        raise ValueError("spectral_library must be a numpy array.")
    if np.any(~np.isfinite(spectral_library)):
        raise ValueError("spectral_library contains non-finite values.")

    n_samples, n_bands = spectral_library.shape
    if num_endmembers > n_samples:
        raise ValueError("num_endmembers cannot be greater than the number of spectra.")
    if initial_selection not in ["max_norm", "dist_mean"]:
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


def build_look_up_table(class_labels, level=2):
    """
    Builds a MESMA look-up table.

    Args:
        class_labels (list): List of class labels (strings or IDs).
        level (int): How many endmembers to combine (e.g., 2 for pairs).

    Returns:
        dict: Valid look_up_table for MesmaCore.
    """
    class_indices = list(range(len(class_labels)))
    combinations = list(itertools.combinations(class_indices, level))

    lookup = {
        level: {combo: np.array([combo]) for combo in combinations}
    }

    return lookup


def main(signatures_path: str, landsat_dir: str):
    download_ecoregion()

    ecoregion_download = os.path.join(PROJ_DIR, 'data', 'Ecoregion', 'us_eco_l3.shp')
    ecoregions = gpd.read_file(ecoregion_download)
    geometries = ecoregions[ecoregions['US_L3NAME'] == 'Southern Rockies']

    signatures = pd.read_csv(signatures_path)
    landsat, max_landsat = read_landsat_data(landsat_dir, geometries)

    spectral_library = signatures.iloc[:, 0:7].to_numpy()
    ies_results = ies_from_library(spectral_library, len(signatures.values))

    endmember_library = signatures.iloc[ies_results['indices'], [0, 1, 2, 3, 4, 5]].copy()
    max_endmember = np.nanmax(endmember_library)

    landsat /= max(max_landsat, max_endmember)
    endmember_library /= max(max_landsat, max_endmember)

    # Rename the first column to "class"
    #endmember_library.columns = ['class'] + list(endmember_library.columns[1:])
    class_labels = list(signatures.iloc[ies_results['indices'], [22]].to_numpy())
    look_up_table = build_look_up_table(class_labels, level=len(endmember_library.values))
    em_per_class = collections.defaultdict(int)
    for cl in class_labels:
        em_per_class[cl[0]] += 1

    mesma = MesmaCore(n_cores=1)
    model_best, model_fractions, model_rmse = mesma.execute(
        image=landsat,
        library=endmember_library,
        look_up_table=look_up_table,
        em_per_class=em_per_class,
        residual_image=False
    )

    plt.imshow(model_rmse[:, :, 0], cmap='viridis')
    plt.colorbar(label='RMSE')
    plt.title('Best Model RMSE')
    plt.show()
