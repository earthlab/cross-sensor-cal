import numpy as np
import json
from spectral import open_image, envi
from spectral.io import envi
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

hyperspectral_hdr = 'NEON_D10_RMNP_DP1_20200701_153414_reflectance__envi.hdr'
resample_params_file = 'Resampling/landsat_band_parameters.json'
img = open_image(hyperspectral_hdr)
#subset_rows = 100
#subset_cols = 100
#hyperspectral_data = img.read_subregion((0, subset_rows), (0, subset_cols))
hyperspectral_data = img.load()
header = envi.read_envi_header(hyperspectral_hdr)
wavelengths = header.get('wavelength')
if wavelengths:
    wavelengths = [float(w) for w in wavelengths]

with open(resample_params_file, 'r') as f:
    all_sensor_params = json.load(f)

rows, cols, bands = hyperspectral_data.shape
row, col = rows // 2, cols // 2

plt.plot(wavelengths, hyperspectral_data[row, col, :].squeeze(), label='Hyperspectral')

def gaussian_rsr(wavelengths, center, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    rsr = norm.pdf(wavelengths, loc=center, scale=sigma)
    return rsr / np.sum(rsr)

for sensor_name, sensor_params in all_sensor_params.items():
    band_centers = np.array(sensor_params["wavelengths"])
    band_fwhms = np.array(sensor_params["fwhms"])

    rsr_matrix = np.array([gaussian_rsr(wavelengths, c, f) for c, f in zip(band_centers, band_fwhms)])
    n_out_bands = len(band_centers)
    resampled = np.zeros((rows, cols, n_out_bands), dtype=np.float32)

    for i in range(n_out_bands):
        weights = rsr_matrix[i]
        weighted = hyperspectral_data * weights[np.newaxis, np.newaxis, :]
        resampled[:, :, i] = np.sum(weighted, axis=2)

    plt.plot(band_centers, resampled[row, col, :], 'o-', label=f'{sensor_name}')

    resampled_hdr_path = f"{sensor_name.replace(' ', '_').replace('+', 'plus').replace('-', '_').lower()}_resampled.hdr"
    resampled_img_path = resampled_hdr_path.replace('.hdr', '.img')

    new_metadata = {
        'description': f'Resampled hyperspectral image using {sensor_name} RSR',
        'samples': cols,
        'lines': rows,
        'bands': n_out_bands,
        'data type': 4,
        'interleave': 'bsq',
        'byte order': 0,
        'sensor type': sensor_name,
        'wavelength units': 'nanometers',
        'wavelength': [str(w) for w in band_centers],
        'map info': header.get('map info'),
        'coordinate system string': header.get('coordinate system string'),
        'data ignore value': header.get('data ignore value'),
    }

    envi.save_image(resampled_hdr_path, resampled, metadata=new_metadata, force=True)
    print(f"Saved resampled image to {resampled_img_path}")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Spectral Convolution Comparison")
plt.legend()
plt.grid(True)
plt.savefig('downsampled.png')