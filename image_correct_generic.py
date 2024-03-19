import json
import os
import warnings
import sys
import ray
import numpy as np
import hytools as ht
from hytools.io.envi import *
from hytools.topo import calc_topo_coeffs
from hytools.brdf import calc_brdf_coeffs
from hytools.glint import set_glint_parameters
from hytools.masks import mask_create


warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

def main():
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

    config_file = sys.argv[1]
    print(config_file)
    

    with open(config_file, 'r') as outfile:
        config_dict = json.load(outfile)

    #print(config_dict)
    
    images = config_dict["input_files"]

    #print(images)
    #print(config_dict['file_type'])
    
    if ray.is_initialized():
        ray.shutdown()
    print("Using %s CPUs." % config_dict['num_cpus'])
    ray.init(num_cpus = config_dict['num_cpus'])

    HyTools = ray.remote(ht.HyTools)
    actors = [HyTools.remote() for image in images]

    if config_dict['file_type'] == 'envi':
        anc_files = config_dict["anc_files"]
        
       # print([(image,config_dict['file_type'], anc_files[image]) for a,image in zip(actors,images)])
        
        _ = ray.get([a.read_file.remote(image,config_dict['file_type'],
                                         anc_files[image]) for a,image in zip(actors,images)])

    elif config_dict['file_type'] == 'neon':
        _ = ray.get([a.read_file.remote(image,config_dict['file_type']) for a,image in zip(actors,images)])

    _ = ray.get([a.create_bad_bands.remote(config_dict['bad_bands']) for a in actors])

    for correction in config_dict["corrections"]:
        if correction =='topo':
            calc_topo_coeffs(actors,config_dict['topo'])
        elif correction == 'brdf':
            calc_brdf_coeffs(actors,config_dict)
        elif correction == 'glint':
            set_glint_parameters(actors,config_dict)

    if config_dict['export']['coeffs'] and len(config_dict["corrections"]) > 0:
        print("Exporting correction coefficients.")
        _ = ray.get([a.do.remote(export_coeffs,config_dict['export']) for a in actors])

    if config_dict['export']['image']:
        print("Exporting corrected images.")
        _ = ray.get([a.do.remote(apply_corrections,config_dict) for a in actors])

    ray.shutdown()

def export_coeffs(hy_obj,export_dict):
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

    for correction in hy_obj.corrections:
        coeff_file = export_dict['output_dir']
        coeff_file += os.path.splitext(os.path.basename(hy_obj.file_name))[0]
        print(coeff_file)
        coeff_file += "_%s_coeffs_%s.json" % (correction,export_dict["suffix"])

        with open(coeff_file, 'w') as outfile:
            if correction == 'topo':
                corr_dict = hy_obj.topo
            elif correction == 'glint':
                continue
            else:
                corr_dict = hy_obj.brdf
            json.dump(corr_dict,outfile)

def apply_corrections(hy_obj,config_dict):
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

    header_dict = hy_obj.get_header()
    header_dict['data ignore value'] = hy_obj.no_data
    header_dict['data type'] = 4

    output_name = config_dict['export']['output_dir']
    output_name += os.path.splitext(os.path.basename(hy_obj.file_name))[0]
    output_name +=  "_%s" % config_dict['export']["suffix"]

    #Export all wavelengths
    if len(config_dict['export']['subset_waves']) == 0:

        if config_dict["resample"] == True:
            hy_obj.resampler = config_dict['resampler']
            waves= hy_obj.resampler['out_waves']
        else:
            waves = hy_obj.wavelengths

        header_dict['bands'] = len(waves)
        header_dict['wavelength'] = waves

        writer = WriteENVI(output_name,header_dict)
        iterator = hy_obj.iterate(by='line', corrections=hy_obj.corrections,
                                  resample=config_dict['resample'])
        while not iterator.complete:
            line = iterator.read_next()
            writer.write_line(line,iterator.current_line)
        writer.close()

    #Export subset of wavelengths
    else:
        waves = config_dict['export']['subset_waves']
        bands = [hy_obj.wave_to_band(x) for x in waves]
        waves = [round(hy_obj.wavelengths[x],2) for x in bands]
        header_dict['bands'] = len(bands)
        header_dict['wavelength'] = waves

        writer = WriteENVI(output_name,header_dict)
        for b,band_num in enumerate(bands):
            band = hy_obj.get_band(band_num,
                                   corrections=hy_obj.corrections)
            writer.write_band(band, b)
        writer.close()

    #Export masks
    if (config_dict['export']['masks']) and (len(config_dict["corrections"]) > 0):
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


        output_name = config_dict['export']['output_dir']
        output_name += os.path.splitext(os.path.basename(hy_obj.file_name))[0]
        output_name +=  "_%s_mask" % config_dict['export']["suffix"]

        writer = WriteENVI(output_name,header_dict)

        for band_num,mask in enumerate(masks):
            mask = mask.astype(int)
            mask[~hy_obj.mask['no_data']] = 255
            writer.write_band(mask,band_num)

        del masks

if __name__== "__main__":
    main()