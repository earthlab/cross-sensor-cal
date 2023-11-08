import json
import glob
import numpy as np

# '''
# Questions:
# 1) what are bad bands and what values to put

# '''

# Only coefficients for good bands will be calculated
bad_bands = [[300, 400], [1337, 1430], [1800, 1960], [2450, 2600]]

# Input data settings for ENVI
file_type = 'envi'
main_image_file = "output/neon"

# Assuming all ancillary files are in the same directory as the main file
anc_files = glob.glob("output/neon_*_ancillary")
anc_files.sort()

# Ancillary files related to each image file
aviris_anc_names = ['path_length', 'sensor_az', 'sensor_zn',
                    'solar_az', 'solar_zn', 'phase', 'slope',
                    'aspect', 'cosine_i']


# Section for topo configuration

# Loop through each ancillary file and create a separate config file
for i, anc_file in enumerate(anc_files):
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
    config_dict['export']['output_dir'] = "export/"
    config_dict['export']["suffix"] = f'_corrected_{i}'

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
    config_dict["brdf"]["b/r"] = 10
    config_dict["brdf"]["h/b"] = 2
    config_dict["brdf"]['interp_kind'] = 'linear'
    config_dict["brdf"]['calc_mask'] = [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]]
    config_dict["brdf"]['apply_mask'] = [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]]
    config_dict["brdf"]['diagnostic_plots'] = True
    config_dict["brdf"]['diagnostic_waves'] = [440, 550, 660, 850]

    # ## Flex dynamic NDVI params
    config_dict["brdf"]['bin_type'] = 'dynamic'
    config_dict["brdf"]['num_bins'] = 18
    config_dict["brdf"]['ndvi_bin_min'] = 0.05
    config_dict["brdf"]['ndvi_bin_max'] = 1.0
    config_dict["brdf"]['ndvi_perc_min'] = 10
    config_dict["brdf"]['ndvi_perc_max'] = 95

    # Define the number of CPUs to be used (considering the number of image-ancillary pairs)
    config_dict['num_cpus'] = 1
    
    # Output path for configuration file
    config_file = f"output/config_{i}.json"

    config_dict["resample"]  = False

    # Save the configuration to a JSON file
    with open(config_file, 'w') as outfile:
        json.dump(config_dict, outfile, indent=3)

# Section for brdf configuration

# Loop through each ancillary file and create a separate config file
# for i, anc_file in enumerate(anc_files):
#     config_dict = {}

#     config_dict['bad_bands'] = bad_bands
#     config_dict['file_type'] = file_type
#     config_dict["input_files"] = [main_image_file]
    
#     config_dict["anc_files"] = {
#         main_image_file: dict(zip(aviris_anc_names, [[anc_file, a] for a in range(len(aviris_anc_names))]))
#     }

#     # Export settings
#     config_dict['export'] = {}
#     config_dict['export']['coeffs'] = True
#     config_dict['export']['image'] = True
#     config_dict['export']['masks'] = True
#     config_dict['export']['subset_waves'] = []
#     config_dict['export']['output_dir'] = "export/"
#     config_dict['export']["suffix"] = f'brdf_corrected_{i}'

#     # BRDF Correction options
#     config_dict["corrections"] = ['brdf']
#     config_dict["brdf"] = {}
#     config_dict["brdf"]['solar_zn_type'] ='scene'
#     config_dict["brdf"]['type'] = 'universal'
#     config_dict["brdf"]['grouped'] = True
#     config_dict["brdf"]['sample_perc'] = 0.1
#     config_dict["brdf"]['geometric'] = 'li_dense_r'
#     config_dict["brdf"]['volume'] = 'ross_thick'
#     config_dict["brdf"]["b/r"] = 2.5
#     config_dict["brdf"]["h/b"] = 2
#     config_dict["brdf"]['calc_mask'] = [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]]
#     config_dict["brdf"]['apply_mask'] = [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]]
#     config_dict["brdf"]['diagnostic_plots'] = True
#     config_dict["brdf"]['diagnostic_waves'] = [440, 550, 660, 850]

#     # Define the number of CPUs to be used (considering the number of image-ancillary pairs)
#     config_dict['num_cpus'] = 1
    
#     # Output path for configuration file
#     config_file = f"output/config_{i}.json"

#     config_dict["resample"]  = False

#     # Save the configuration to a JSON file
#     with open(config_file, 'w') as outfile:
#         json.dump(config_dict, outfile, indent=3)

#########################################################################################################
# Version Second brdf

# import json
# import glob
# import os

# # Folder name where the files are located
# folder_name = "output"

# # Only coefficients for good bands will be calculated
# bad_bands = [[300, 400], [1337, 1430], [1800, 1960], [2450, 2600]]

# # Input data settings for ENVI
# file_type = 'envi'

# # Assuming all ancillary files are in the same directory as the main file
# anc_files = glob.glob(f"{folder_name}/neon_*_ancillary")
# anc_files.sort()

# # Ancillary files related to each image file
# aviris_anc_names = ['path_length', 'sensor_az', 'sensor_zn',
#                     'solar_az', 'solar_zn', 'phase', 'slope',
#                     'aspect']

# # Create the main configuration dictionary
# config_dict = {}
# config_dict['bad_bands'] = bad_bands
# config_dict['file_type'] = file_type
# config_dict["anc_files"] = {}

# input_files = []

# for anc_file in anc_files:
#     symlink_name = os.path.basename(anc_file).split('_ancillary')[0]
#     input_files.append(f"{folder_name}/{symlink_name}")
#     config_dict["anc_files"][f"{folder_name}/{symlink_name}"] = dict(zip(aviris_anc_names, [[anc_file, a] for a in range(len(aviris_anc_names))]))

# config_dict["input_files"] = input_files

# # Export settings
# config_dict['export'] = {
#     "coeffs": True,
#     "image": True,
#     "masks": True,
#     "subset_waves": [],
#     "output_dir": "export/",
#     "suffix": "brdf_corrected"
# }

# # BRDF Correction options
# config_dict["corrections"] = ['brdf']
# config_dict["brdf"] = {
#     'solar_zn_type': 'scene',
#     'type': 'universal',
#     'grouped': True,
#     'sample_perc': 0.1,
#     'geometric': 'li_dense_r',
#     'volume': 'ross_thick',
#     "b/r": 2.5,
#     "h/b": 2,
#     'calc_mask': [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]],
#     'apply_mask': [["ndi", {'band_1': 850, 'band_2': 660, 'min': 0.1, 'max': 1.0}]],
#     'diagnostic_plots': True,
#     'diagnostic_waves': [440, 550, 660, 850]
# }

# # Define the number of CPUs to be used (considering the number of image-ancillary pairs)
# config_dict['num_cpus'] = 1

# config_dict["resample"] = False

# # Output path for configuration file
# config_file = f"{folder_name}/config.json"

# # Save the configuration to a JSON file
# with open(config_file, 'w') as outfile:
#     json.dump(config_dict, outfile, indent=3)
