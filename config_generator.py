import json
import glob
import numpy as np
import os
"""
Configuration Generator for TOPO and BRDF Correction

This script generates configuration files for TOPO and BRDF corrections of geospatial imagery. 
It is tailored to handle specific input formats and ancillary data, and outputs configurations 
for each ancillary file associated with the main image.

Key Components:
1. Bad Bands: Specify wavelength ranges that are not suitable for processing.
2. File Type: Define the type of the input image files.
3. Main Image File: Path to the primary image file for which corrections are to be applied.
4. Ancillary Files: Automatically fetches ancillary files related to the main image.
5. Ancillary Names: Defines the names of ancillary datasets for referencing in the configuration.

Required Inputs:
- Bad band ranges: List of [start, end] wavelength ranges that should be excluded.
- File type: String specifying the format of the image data (e.g., 'envi').
- Main image file path: String indicating the full path to the main image file.
- Ancillary file pattern: String with a glob pattern to identify ancillary files.

Process Overview:
- The script iterates through each ancillary file, creating a separate configuration file.
- For each configuration, it includes settings for both TOPO and BRDF corrections, export options, 
  and mappings between ancillary datasets and the main image file.
- The configuration details include various parameters and settings specific to the correction algorithms.
- Each configuration is saved as a JSON file in a specified output directory.

Usage:
- This script is typically used in workflows where geospatial imagery requires correction for topographic and 
  bidirectional reflectance distribution function (BRDF) effects.
- It is especially useful when processing large datasets or multiple images with similar correction requirements.

Output:
- JSON configuration files for each ancillary file, containing all necessary settings for TOPO and BRDF corrections.
"""


# Only coefficients for good bands will be calculated
bad_bands = [[300, 400], [1337, 1430], [1800, 1960], [2450, 2600]]

# Input data settings for ENVI
file_type = 'envi'

main_image_file = r"output/NEON_D13_NIWO_DP1_20200731_155024_reflectance"
main_image_name = os.path.basename(main_image_file).split('.')[0]  # Removes the file extension

# Assuming all ancillary files are in the same directory as the main file
anc_files = glob.glob("output/NEON_D13_NIWO_DP1_20200731_155024_reflectance/*_ancillary*")
anc_files.sort()

#print(anc_files)

# Ancillary files related to each image file
aviris_anc_names = ['path_length', 'sensor_az', 'sensor_zn',
                    'solar_az', 'solar_zn', 'phase', 'slope',
                    'aspect', 'cosine_i']


# Section for topo configuration

# Loop through each ancillary file and create a separate config file
for i, anc_file in enumerate(anc_files):
    if i == 0:
        suffix_label = f"{main_image_name}_corrected_envi"
    elif i == 1:
        suffix_label = f"{main_image_name}_corrected_anc"
    else:
        suffix_label = f"{main_image_name}_corrected_{i}"  # Fallback for unexpected 'i' values

    
    
    config_dict = {}

    config_dict['bad_bands'] = bad_bands
    config_dict['file_type'] = file_type
    config_dict["input_files"] = [main_image_file]
    
    config_dict["anc_files"] = {
        main_image_file: dict(zip(aviris_anc_names, [[anc_file, a] for a in range(len(aviris_anc_names))]))
    }

    # Export settings
    config_dict['export'] = {}
    config_dict['export']['coeffs'] = True
    config_dict['export']['image'] = True
    config_dict['export']['masks'] = True
    config_dict['export']['subset_waves'] = []
    config_dict['export']['output_dir'] = main_image_file
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
    config_dict["corrections"] = ['topo','brdf']
    config_dict["topo"] =  {}
    config_dict["topo"]['type'] = 'scs+c'
    config_dict["topo"]['calc_mask'] = [["ndi", {'band_1': 850,'band_2': 660,
                                                'min': 0.1,'max': 1.0}],
                                        ['ancillary',{'name':'slope',
                                                    'min': np.radians(5),'max':'+inf' }],
                                        ['ancillary',{'name':'cosine_i',
                                                    'min': 0.12,'max':'+inf' }],
                                        ['cloud',{'method':'zhai_2018',
                                                'cloud':True,'shadow':True,
                                                'T1': 0.01,'t2': 1/10,'t3': 1/4,
                                                't4': 1/2,'T7': 9,'T8': 9}]]

    config_dict["topo"]['apply_mask'] = [["ndi", {'band_1': 850,'band_2': 660,
                                                'min': 0.1,'max': 1.0}],
                                        ['ancillary',{'name':'slope',
                                                    'min': np.radians(5),'max':'+inf' }],
                                        ['ancillary',{'name':'cosine_i',
                                                    'min': 0.12,'max':'+inf' }]]
    config_dict["topo"]['c_fit_type'] = 'nnls'

    # config_dict["topo"]['type'] =  'precomputed'
    # config_dict["brdf"]['coeff_files'] =  {}

    # BRDF Correction options
    config_dict["brdf"] = {}
    config_dict["brdf"]['solar_zn_type'] ='scene'
    config_dict["brdf"]['type'] = 'flex'
    config_dict["brdf"]['grouped'] = True
    config_dict["brdf"]['sample_perc'] = 0.1
    config_dict["brdf"]['geometric'] = 'li_dense_r'
    config_dict["brdf"]['volume'] = 'ross_thick'
    config_dict["brdf"]["b/r"] = 10  #these may need updating. These constants pulled from literature. 
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
    config_dict['num_cpus'] = 1
    
    # Output path for configuration file
    # Assuming you want to include the suffix in the filename:
    suffix = config_dict['export']["suffix"]
    output_dir = config_dict['export']['output_dir']
    config_file_name = f"{suffix}.json"
    config_file_path = os.path.join(output_dir, config_file_name)

    config_dict["resample"]  = False
    config_dict["resampler"]  = {}
    config_dict["resampler"]['type'] =  'cubic'
    config_dict["resampler"]['out_waves'] = []
    config_dict["resampler"]['out_fwhm'] = []

    # Remove bad bands from output waves
    for wavelength in range(450,660,100):
        bad=False
        for start,end in config_dict['bad_bands']:
            bad = ((wavelength >= start) & (wavelength <=end)) or bad
        if not bad:
            config_dict["resampler"]['out_waves'].append(wavelength)

    # Save the configuration to a JSON file
    with open(config_file_path, 'w') as outfile:
        json.dump(config_dict, outfile, indent=3)


