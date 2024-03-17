import argparse
import os
import h5py
import ray
import numpy as np
import hytools as ht
from hytools.io.envi import WriteENVI

def get_actual_key(h5_file, expected_key):
    """
    Finds the actual key in an HDF5 file that corresponds to an expected key, accounting for case sensitivity.

    Inputs:
    - h5_file: The HDF5 file object.
    - expected_key: The expected key (string) whose actual key in the file is to be found.

    Output:
    - Returns the actual key from the HDF5 file that matches the expected key, regardless of case sensitivity.
    """
    actual_keys = {key.lower(): key for key in h5_file.keys()}
    return actual_keys.get(expected_key.lower(), expected_key)

def get_all_solar_angles(logs_group):
    """
        Extracts all solar azimuth and zenith angles from a logs group in an HDF5 file.

        Input:
        - logs_group: The group within the HDF5 file containing logs data.

        Output:
        - Returns an array of tuples containing solar azimuth and zenith angles.
    """
    
    angles = []
    for log_entry in logs_group.keys():
        angles.append((
            logs_group[log_entry]["Solar_Azimuth_Angle"][()],
            logs_group[log_entry]["Solar_Zenith_Angle"][()]
        ))
    return np.array(angles)

import argparse
import os
import h5py
import numpy as np
import hytools as ht
from hytools.io.envi import WriteENVI

# Your get_actual_key and get_all_solar_angles functions remain the same.

def export_anc(args):
    """
    Modifies the function to take a dictionary of arguments for flexibility.
    Expects 'hy_obj', 'h5_file_path', 'site_code', and 'output_dir' in args.
    """
    h5_file_path = args['h5_file_path']
    site_code = args['site_code']
    output_dir = args['output_dir']

    with h5py.File(h5_file_path, 'r') as h5_file:
        base_path = f"/{site_code}/Reflectance/Metadata/Ancillary_Imagery/"
        path_length_data = h5_file[base_path + "Path_Length"][...]
        slope_data = h5_file[base_path + "Slope"][...]
        aspect_data = h5_file[base_path + "Aspect"][...]
        # Continues as before...

def main():
    parser = argparse.ArgumentParser(description="Convert NEON AOP H5 to ENVI format and optionally export ancillary data.")
    parser.add_argument('images', nargs='*', help="Input image pathnames")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--site_code', required=True, help="NEON site code (e.g., 'NIWO')")
    parser.add_argument("-anc", action='store_true', help="Output ancillary", required=False)

    args = parser.parse_args()

    if not args.output_dir.endswith("/"):
        args.output_dir += "/"

    # Initialize Ray if needed and process each image.
    if args.anc:
        # Adjusted to use a dictionary for passing multiple arguments easily.
        anc_args = {
            'h5_file_path': None,  # This will be set per image in the loop
            'site_code': args.site_code,
            'output_dir': args.output_dir,
        }
        for image_path in args.images:
            print(f"Processing {image_path}")
            anc_args['h5_file_path'] = image_path
            export_anc(anc_args)

    print("Export complete.")

if __name__ == "__main__":
    main()

