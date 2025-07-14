import glob

import json
import os.path
import warnings
from pathlib import Path

import ray
import re
import hytools as ht
from hytools.io.envi import *
from hytools.topo import calc_topo_coeffs
from hytools.brdf import calc_brdf_coeffs
from hytools.glint import set_glint_parameters
from hytools.masks import mask_create
from spectral import open_image
from spectral.io import envi

from src.file_types import (NEONReflectanceFile, NEONReflectanceENVIFile, NEONReflectanceConfigFile,
                            NEONReflectanceCoefficientsFile, NEONReflectanceAncillaryENVIFile,
                            NEONReflectanceBRDFCorrectedENVIFile, NEONReflectanceBRDFMaskENVIFile,
                            NEONReflectanceBRDFCorrectedENVIHDRFile)

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')


def topo_and_brdf_correction(config_file: str):
    """
    Main Function for Processing Geospatial Imagery

    This function orchestrates the process of reading, correcting, and exporting geospatial imagery
    based on the configuration settings specified in a JSON file. It utilizes parallel processing capabilities
    provided by Ray to efficiently handle multiple images.

    Process Flow:
    1. Load Configuration: Reads the JSON configuration file provided as a command line argument.
    2. Initialize Ray: Sets up Ray for parallel processing, using the number of CPUs specified in the config.
    3. Read Files: Depending on the file type ('envi' or 'neon'), reads the input files and ancillary data.
    4. Apply Corrections: Executes various correction algorithms (e.g., TOPO, BRDF, glint) as specified in the config.
    5. Export Data: Outputs corrected imagery and correction coefficients, again as specified in the config.

    Inputs:
    - A single command line argument specifying the path to the JSON configuration file.

    The configuration file should contain:
    - Input file paths.
    - Number of CPUs to use for processing.
    - File type (e.g., 'envi', 'neon').
    - Correction parameters and settings.
    - Export settings for the corrected images and coefficients.

    Output:
    - Corrected geospatial imagery files.
    - Correction coefficients files, if specified in the configuration.

    Usage:
    - This function is typically used as part of a larger workflow for processing and analyzing geospatial imagery,
      particularly when dealing with large datasets or requiring specialized correction techniques.
    - To execute, run this script with Python, providing the path to the configuration file as an argument.
    """
    with open(config_file, 'r') as outfile:
        config_dict = json.load(outfile)

    images = config_dict["input_files"]

    if len(images) != 1:
        raise ValueError('Length of images in config file not equal to 1')

    image = images[0]
    reflectance_file = NEONReflectanceENVIFile.from_filename(Path(image))

    if ray.is_initialized():
        ray.shutdown()
    print("Using %s CPUs." % config_dict['num_cpus'])
    ray.init(num_cpus=config_dict['num_cpus'])

    HyTools = ray.remote(ht.HyTools)
    actors = [HyTools.remote() for _ in images]

    if config_dict['file_type'] == 'envi':
        anc_files = config_dict["anc_files"]
        _ = ray.get([a.read_file.remote(image, config_dict['file_type'],
                                        anc_files[image]) for a, image in zip(actors, images)])

    elif config_dict['file_type'] == 'neon':
        _ = ray.get([a.read_file.remote(image, config_dict['file_type']) for a, image in zip(actors, images)])

    _ = ray.get([a.create_bad_bands.remote(config_dict['bad_bands']) for a in actors])

    brdf_corrected_file = NEONReflectanceBRDFCorrectedENVIFile.from_components(
        reflectance_file.domain,
        reflectance_file.site,
        reflectance_file.date,
        reflectance_file.time,
        config_dict['export']["suffix"],
        Path(config_dict['export']['output_dir'])
    )

    for correction in config_dict["corrections"]:
        coefficients_file = NEONReflectanceCoefficientsFile.from_components(
            reflectance_file.domain,
            reflectance_file.site,
            reflectance_file.date,
            reflectance_file.time,
            correction,
            config_dict['export']["suffix"],
            Path(config_dict['export']['output_dir'])
        )
        print(brdf_corrected_file.file_path, coefficients_file.file_path)
        if brdf_corrected_file.path.exists() and coefficients_file.path.exists():
            continue
        if correction == 'topo':
            calc_topo_coeffs(actors, config_dict['topo'])
        elif correction == 'brdf':
            calc_brdf_coeffs(actors, config_dict)
        elif correction == 'glint':
            set_glint_parameters(actors, config_dict)

    if config_dict['export']['coeffs'] and len(config_dict["corrections"]) > 0:
        print("Exporting correction coefficients.")
        _ = ray.get([a.do.remote(export_coeffs, config_dict['export']) for a in actors])

    if config_dict['export']['image']:
        print("Exporting corrected images.")
        _ = ray.get([a.do.remote(apply_corrections, config_dict) for a in actors])

    ray.shutdown()


def export_coeffs(hy_obj, export_dict):
    """
    Exports Correction Coefficients to Files

    This function exports the correction coefficients for various corrections applied to geospatial imagery.
    It generates a separate file for each type of correction, storing the coefficients in JSON format.

    Inputs:
    - hy_obj: An object representing the processed imagery data, which includes correction coefficients.
    - export_dict: A dictionary containing export settings, such as the output directory and file suffix.

    Process:
    - Iterates through each correction type present in the 'hy_obj'.
    - Constructs a file path for each correction's coefficients based on the settings in 'export_dict'.
    - Exports the coefficients to a JSON file.

    Supported Corrections:
    - Topographic ('topo')
    - Glint (currently skipped in this function)
    - BRDF ('brdf')

    Output:
    - JSON files containing correction coefficients for each correction type.
      The files are named based on the correction type and include a suffix from 'export_dict'.

    Usage:
    - Typically used after applying corrections to geospatial imagery data.
    - Helps in documenting the coefficients used for corrections, which can be useful for analysis, reproducibility, and auditing.

    Note:
    - The function currently skips exporting coefficients for 'glint' correction.
    """
    reflectance_file = NEONReflectanceENVIFile.from_filename(Path(hy_obj.file_name))
    for correction in hy_obj.corrections:
        coefficients_file = NEONReflectanceCoefficientsFile.from_components(
            reflectance_file.domain,
            reflectance_file.site,
            reflectance_file.date,
            reflectance_file.time,
            correction,
            export_dict["suffix"],
            Path(export_dict["output_dir"])
        )

        if coefficients_file.path.exists():
            continue

        with open(coefficients_file.file_path, 'w') as outfile:
            if correction == 'topo':
                corr_dict = hy_obj.topo
            elif correction == 'glint':
                continue
            else:
                corr_dict = hy_obj.brdf
            json.dump(corr_dict, outfile)


def apply_corrections(hy_obj, config_dict):
    """
    Applies specified corrections to geospatial imagery data and exports the corrected images.

    Inputs:
    - hy_obj: An object representing the processed imagery data.
    - config_dict: A dictionary containing configuration settings for corrections and exports.

    Process Overview:
    1. Update Header: Modifies the header dictionary with relevant details like 'data ignore value' and 'data type'.
    2. Output File Naming: Constructs the output file path based on the provided configuration.
    3. Correction and Export:
       a. If 'subset_waves' is empty, exports all wavelengths, with optional resampling.
       b. If 'subset_waves' contains specific wavelengths, exports only the selected bands.
    4. Mask Export: Optionally, exports masks associated with the applied corrections.
    5. File Writing: Utilizes the WriteENVI utility to write the corrected data to ENVI format files.

    Outputs:
    - Corrected imagery files in ENVI format.
    - Optionally, mask files indicating areas affected by specific corrections.

    Usage:
    - This function is used in workflows where geospatial imagery requires corrections like TOPO, BRDF, etc.
    - It is a part of a larger processing pipeline, following the application of corrections to the data.

    Note:
    - The function assumes that the 'hy_obj' has methods like 'get_header', 'iterate', and 'get_band' and attributes like 'corrections' and 'wavelengths'.
    - The 'WriteENVI' utility is used for writing the output, which should be predefined or imported.
    """
    print(config_dict["corrections"], "CORRECTIONS")
    header_dict = hy_obj.get_header()
    header_dict['data ignore value'] = hy_obj.no_data
    header_dict['data type'] = 4

    reflectance_file = NEONReflectanceENVIFile.from_filename(Path(os.path.basename(hy_obj.file_name)))
    brdf_corrected_file = NEONReflectanceBRDFCorrectedENVIFile.from_components(
        reflectance_file.domain,
        reflectance_file.site,
        reflectance_file.date,
        reflectance_file.time,
        config_dict['export']["suffix"],
        Path(config_dict['export']['output_dir'])
    )

    if not brdf_corrected_file.path.exists():
        # Export all wavelengths
        if len(config_dict['export']['subset_waves']) == 0:
            if config_dict["resample"]:
                hy_obj.resampler = config_dict['resampler']
                waves = hy_obj.resampler['out_waves']
            else:
                waves = hy_obj.wavelengths

            header_dict['bands'] = len(waves)
            header_dict['wavelength'] = waves

            writer = WriteENVI(brdf_corrected_file.file_path, header_dict)
            iterator = hy_obj.iterate(by='line', corrections=hy_obj.corrections, resample=config_dict['resample'])
            while not iterator.complete:
                line = iterator.read_next()
                writer.write_line(line, iterator.current_line)
            writer.close()

        # Export subset of wavelengths
        else:
            waves = config_dict['export']['subset_waves']
            bands = [hy_obj.wave_to_band(x) for x in waves]
            waves = [round(hy_obj.wavelengths[x], 2) for x in bands]
            header_dict['bands'] = len(bands)
            header_dict['wavelength'] = waves

            writer = WriteENVI(brdf_corrected_file.file_path, header_dict)
            for b, band_num in enumerate(bands):
                band = hy_obj.get_band(band_num, corrections=hy_obj.corrections)
                writer.write_band(band, b)
            writer.close()

    # Export masks
    brdf_corrected_masked_file = NEONReflectanceBRDFMaskENVIFile.from_components(
        reflectance_file.domain,
        reflectance_file.site,
        reflectance_file.date,
        reflectance_file.time,
        config_dict['export']["suffix"],
        Path(config_dict['export']['output_dir'])
    )

    if (config_dict['export']['masks'] and len(config_dict["corrections"]) > 0 and
            not brdf_corrected_masked_file.path.exists()):
        masks = []
        mask_names = []

        for correction in config_dict["corrections"]:
            for mask_type in config_dict[correction]['apply_mask']:
                mask_names.append(correction + '_' + mask_type[0])
                masks.append(mask_create(hy_obj, [mask_type]))

        header_dict['data type'] = 1
        header_dict['bands'] = len(masks)
        header_dict['band names'] = mask_names
        header_dict['samples'] = hy_obj.columns
        header_dict['lines'] = hy_obj.lines
        header_dict['wavelength'] = []
        header_dict['fwhm'] = []
        header_dict['wavelength units'] = ''
        header_dict['data ignore value'] = 255

        writer = WriteENVI(brdf_corrected_masked_file.file_path, header_dict)

        for band_num, mask in enumerate(masks):
            mask = mask.astype(int)
            mask[~hy_obj.mask['no_data']] = 255
            writer.write_band(mask, band_num)

        del masks

def generate_correction_configs_for_directory(reflectance_file: NEONReflectanceFile):
    """
    Generates configuration files for TOPO and BRDF corrections for all ancillary files in a given directory.

    Args:
    - directory (str): The directory containing the main image and its ancillary files.
    """
    # Define your configuration settings
    bad_bands = [[300, 400], [1337, 1430], [1800, 1960], [2450, 2600]]
    file_type = 'envi'

    anc_files = NEONReflectanceAncillaryENVIFile.find_in_directory(reflectance_file.directory)
    print(f'ANC files {anc_files}')
    anc_files.sort()

    aviris_anc_names = ['path_length', 'sensor_az', 'sensor_zn', 'solar_az', 'solar_zn', 'slope', 'aspect', 'phase',
                        'cosine_i']

    # Loop through each ancillary file and create a separate config file
    for i, anc_file in enumerate(anc_files):
        if i == 0:
            suffix_label = f"envi"
        elif i == 1:
            suffix_label = f"anc"
        else:
            suffix_label = f"{i}"

        config_dict = {}

        config_dict['bad_bands'] = bad_bands
        config_dict['file_type'] = file_type
        config_dict["input_files"] = [reflectance_file.file_path]

        config_dict["anc_files"] = {
            reflectance_file.file_path: dict(zip(aviris_anc_names, [[anc_file.file_path, a] for a in range(len(aviris_anc_names))]))
        }

        # Export settings
        config_dict['export'] = {}
        config_dict['export']['coeffs'] = True
        config_dict['export']['image'] = True
        config_dict['export']['masks'] = True
        config_dict['export']['subset_waves'] = []
        config_dict['export']['output_dir'] = str(reflectance_file.directory)
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
        config_dict["corrections"] = ['topo', 'brdf']
        config_dict["topo"] = {}
        config_dict["topo"]['type'] = 'scs+c'
        config_dict["topo"]['calc_mask'] = [["ndi", {'band_1': 850, 'band_2': 660,
                                                     'min': 0.1, 'max': 1.0}],
                                            ['ancillary', {'name': 'slope',
                                                           'min': np.radians(5), 'max': '+inf'}],
                                            ['ancillary', {'name': 'cosine_i',
                                                           'min': 0.12, 'max': '+inf'}],
                                            ['cloud', {'method': 'zhai_2018',
                                                       'cloud': True, 'shadow': True,
                                                       'T1': 0.01, 't2': 1 / 10, 't3': 1 / 4,
                                                       't4': 1 / 2, 'T7': 9, 'T8': 9}]]

        config_dict["topo"]['apply_mask'] = [["ndi", {'band_1': 850, 'band_2': 660,
                                                      'min': 0.1, 'max': 1.0}],
                                             ['ancillary', {'name': 'slope',
                                                            'min': np.radians(5), 'max': '+inf'}],
                                             ['ancillary', {'name': 'cosine_i',
                                                            'min': 0.12, 'max': '+inf'}]]
        config_dict["topo"]['c_fit_type'] = 'nnls'

        # config_dict["topo"]['type'] =  'precomputed'
        # config_dict["brdf"]['coeff_files'] =  {}

        # BRDF Correction options
        config_dict["brdf"] = {}
        config_dict["brdf"]['solar_zn_type'] = 'scene'
        config_dict["brdf"]['type'] = 'flex'
        config_dict["brdf"]['grouped'] = True
        config_dict["brdf"]['sample_perc'] = 0.1
        config_dict["brdf"]['geometric'] = 'li_dense_r'
        config_dict["brdf"]['volume'] = 'ross_thick'
        config_dict["brdf"]["b/r"] = 10  # these may need updating. These constants pulled from literature.
        config_dict["brdf"]["h/b"] = 2  # These may need updating. These contanstants pulled from literature.
        config_dict["brdf"]['interp_kind'] = 'linear'
        config_dict["brdf"]['calc_mask'] = [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]]
        config_dict["brdf"]['apply_mask'] = [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]]
        config_dict["brdf"]['diagnostic_plots'] = True
        config_dict["brdf"]['diagnostic_waves'] = [448, 849, 1660, 2201]

        # ## Flex dynamic NDVI params
        config_dict["brdf"]['bin_type'] = 'dynamic'
        config_dict["brdf"]['num_bins'] = 25
        config_dict["brdf"]['ndvi_bin_min'] = 0.05
        config_dict["brdf"]['ndvi_bin_max'] = 1.0
        config_dict["brdf"]['ndvi_perc_min'] = 10
        config_dict["brdf"]['ndvi_perc_max'] = 95

        # Define the number of CPUs to be used (considering the number of image-ancillary pairs)
        config_dict['num_cpus'] = 8

        # Output path for configuration file
        # Assuming you want to include the suffix in the filename:
        suffix = config_dict['export']["suffix"]

        config_dict["resample"] = False
        config_dict["resampler"] = {}
        config_dict["resampler"]['type'] = 'cubic'
        config_dict["resampler"]['out_waves'] = []
        config_dict["resampler"]['out_fwhm'] = []

        # Remove bad bands from output waves
        for wavelength in range(450, 660, 100):
            bad = False
            for start, end in config_dict['bad_bands']:
                bad = ((wavelength >= start) & (wavelength <= end)) or bad
            if not bad:
                config_dict["resampler"]['out_waves'].append(wavelength)

        # Construct the filename for the configuration JSON
        config_file = NEONReflectanceConfigFile.from_components(
            reflectance_file.domain, reflectance_file.site, reflectance_file.date, reflectance_file.time, suffix,
            reflectance_file.directory)
        print(f'CONFIG file {config_file.file_path}')
        # Save the configuration to a JSON file
        with open(config_file.file_path, 'w+') as outfile:
            print(config_dict, outfile)
            json.dump(config_dict, outfile, indent=3)

        print(f"Configuration saved to {config_file.file_path}")


def generate_config_json(parent_directory):
    """
    Loops through each subdirectory within the given parent directory and generates configuration files for each,
    excluding certain unwanted directories like '.ipynb_checkpoints'.

    Args:
    - parent_directory (str): The parent directory containing multiple subdirectories for which to generate configurations.
    """
    # Find all subdirectories within the parent directory, excluding the ones in `exclude_dirs`
    reflectance_files = NEONReflectanceENVIFile.find_in_directory(Path(parent_directory))
    print(f"Reflectance files: {reflectance_files}")

    # Loop through each subdirectory and generate correction configurations
    for reflectance_file in reflectance_files:
        print(type(reflectance_file))
        print(reflectance_file.file_path, reflectance_file.directory)
        print(f"Generating configuration files for directory: {os.path.dirname(reflectance_file.file_path)}")
        generate_correction_configs_for_directory(reflectance_file)
        print("Configuration files generation completed.\n")


def apply_offset_to_envi(input_dir: Path, offset: float):
    """
    Applies a constant offset to all valid pixels in an ENVI image.
    Invalid/masked pixels (NaN or -9999) are preserved. Values are clipped to a minimum of 0.

    Args:
        offset (float): Value to add to each pixel.
    """
    brdf_corrected_header_files = NEONReflectanceBRDFCorrectedENVIHDRFile.find_in_directory(input_dir, 'envi')

    for brdf_corrected_header_file in brdf_corrected_header_files:
        print(f'Applying offset of {offset} to {brdf_corrected_header_file.file_path}')
        img = open_image(brdf_corrected_header_file.file_path)
        data = img.load().copy()  # Load into memory so we can overwrite

        # Create mask for invalid values (NaN or -9999)
        mask = np.isnan(data) | (data == -9999)

        # Apply offset where valid
        data[~mask] += offset

        # Clip values to be >= 0
        data = np.clip(data, 0, None)

        # Restore mask values
        data[mask] = -9999  # Or np.nan if you prefer

        # Overwrite original ENVI file
        envi.save_image(brdf_corrected_header_file.file_path, data, interleave=str(img.interleave).lower(),
                        byte_order=str(img.byte_order).lower(), metadata=img.metadata, force=True)
