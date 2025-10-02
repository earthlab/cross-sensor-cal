from pathlib import Path
import numpy as np
import json
from spectral import open_image
from spectral.io import envi
from scipy.stats import norm
import os

from .file_types import (
    NEONReflectanceResampledENVIFile,
    NEONReflectanceResampledHDRFile,
    NEONReflectanceBRDFCorrectedENVIHDRFile,
)

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))


def gaussian_rsr(wavelengths, center, fwhm):
    """Generate Gaussian Relative Spectral Response (RSR)"""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    rsr = norm.pdf(wavelengths, loc=center, scale=sigma)
    return rsr / np.sum(rsr)


def resample(input_dir: Path):
    """Perform convolutional resampling for BRDF+TOPO corrected hyperspectral data"""
    print(f"🚀 Starting convolutional resample for {input_dir}")
    brdf_corrected_hdr_files = (
        NEONReflectanceBRDFCorrectedENVIHDRFile.find_in_directory(input_dir)
    )

    for hdr_file in brdf_corrected_hdr_files:
        try:
            print(f"📂 Opening: {hdr_file.file_path}")
            img = open_image(hdr_file.file_path)
            hyperspectral_data = img.load()
        except Exception as e:
            print(f"❌ ERROR: Could not load {hdr_file.file_path}: {e}")
            continue

        header = envi.read_envi_header(hdr_file.file_path)
        wavelengths = header.get("wavelength")

        if wavelengths is None:
            with open(
                os.path.join(PROJ_DIR, "data", "hyperspectral_bands.json"), "r"
            ) as f:
                wavelengths = json.load(f).get("bands")

        if not wavelengths:
            print("❌ ERROR: No wavelengths found.")
            continue

        wavelengths = [float(w) for w in wavelengths]

        rows, cols, bands = hyperspectral_data.shape
        if len(wavelengths) != bands:
            print(
                f"❌ ERROR: Band mismatch ({len(wavelengths)} wavelengths vs {bands} bands)."
            )
            continue

        with open(
            os.path.join(PROJ_DIR, "data", "landsat_band_parameters.json"), "r"
        ) as f:
            all_sensor_params = json.load(f)

        for sensor_name, sensor_params in all_sensor_params.items():
            resampled_dir = (
                hdr_file.directory
                / f"Convolution_Reflectance_Resample_{sensor_name.replace(' ', '_')}"
            )
            os.makedirs(resampled_dir, exist_ok=True)

            # Build output file paths using corrected naming conventions
            resampled_hdr_file = NEONReflectanceResampledHDRFile.from_components(
                domain=hdr_file.domain,
                site=hdr_file.site,
                date=hdr_file.date,
                sensor=sensor_name,
                suffix=hdr_file.suffix,
                folder=resampled_dir,
                time=hdr_file.time,
                tile=hdr_file.tile,
                directional=hdr_file.directional,
            )

            resampled_img_file = NEONReflectanceResampledENVIFile.from_components(
                domain=hdr_file.domain,
                site=hdr_file.site,
                date=hdr_file.date,
                sensor=sensor_name,
                suffix=hdr_file.suffix,
                folder=resampled_dir,
                time=hdr_file.time,
                tile=hdr_file.tile,
                directional=hdr_file.directional,
            )

            if resampled_hdr_file.path.exists() and resampled_img_file.path.exists():
                print(f"⚠️ Skipping resampling for {sensor_name}: files already exist.")
                continue

            # Perform convolutional resampling
            band_centers = np.array(sensor_params["wavelengths"])
            band_fwhms = np.array(sensor_params["fwhms"])
            rsr_matrix = np.array(
                [
                    gaussian_rsr(wavelengths, c, f)
                    for c, f in zip(band_centers, band_fwhms)
                ]
            )
            n_out_bands = len(band_centers)
            resampled = np.zeros((rows, cols, n_out_bands), dtype=np.float32)

            for i in range(n_out_bands):
                weights = rsr_matrix[i]
                weighted = hyperspectral_data * weights[np.newaxis, np.newaxis, :]
                resampled[:, :, i] = np.sum(weighted, axis=2)

            new_metadata = {
                "description": f"Resampled hyperspectral image using {sensor_name} RSR",
                "samples": cols,
                "lines": rows,
                "bands": n_out_bands,
                "data type": 4,
                "interleave": "bsq",
                "byte order": 0,
                "sensor type": sensor_name,
                "wavelength units": "nanometers",
                "wavelength": [str(w) for w in band_centers],
                "map info": header.get("map info"),
                "coordinate system string": header.get("coordinate system string"),
                "data ignore value": header.get("data ignore value"),
            }

            envi.save_image(
                resampled_hdr_file.file_path,
                resampled,
                metadata=new_metadata,
                force=True,
            )
            print(f"✅ Resampled file saved: {resampled_hdr_file.file_path}")
