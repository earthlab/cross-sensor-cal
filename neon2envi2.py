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

def export_anc(hy_obj, args_and_image):
    """
    Exports ancillary data from an HDF5 file to ENVI format.

    Inputs:
    - hy_obj: An object representing the HDF5 file.
    - args_and_image: A tuple containing command line arguments and the image file path.

    This function reads various ancillary data from the HDF5 file, such as path length, slope, aspect, etc., and exports them to ENVI format.
    """
    h5_file_path, args = args_and_image

    with h5py.File(hy_obj.file_name, 'r') as h5_file:
        # Fetching data from the provided h5 file structure
        path_length_data = h5_file['/NIWO/Reflectance/Metadata/Ancillary_Imagery/Path_Length'][...]
        slope_data = h5_file['/NIWO/Reflectance/Metadata/Ancillary_Imagery/Slope'][...]
        aspect_data = h5_file['/NIWO/Reflectance/Metadata/Ancillary_Imagery/Aspect'][...]
        # Add other ancillary datasets as needed

        illumination_factor = h5_file['/NIWO/Reflectance/Metadata/Ancillary_Imagery/Illumination_Factor'][...]
        to_sensor_azimuth_angle_data = h5_file['/NIWO/Reflectance/Metadata/to-sensor_Azimuth_Angle'][...]
        to_sensor_zenith_angle_data = h5_file['/NIWO/Reflectance/Metadata/to-sensor_Zenith_Angle'][...]
        
        # Assuming solar angles are stored per log entry
        logs_group = h5_file['/NIWO/Reflectance/Metadata/Logs']
        solar_azimuth_angle = logs_group["Solar_Azimuth_Angle"][()]
        solar_zenith_angle = logs_group["Solar_Zenith_Angle"][()]

        # Prepare ancillary header
        anc_header = hy_obj.get_header()
        anc_header['bands'] = 8  # Update this number based on the actual number of bands you are including
        anc_header['band_names'] = ['path length', 'to-sensor azimuth', 'to-sensor zenith', 'solar azimuth', 
                                    'solar zenith', 'slope', 'aspect', 'cosine_i']
        anc_header['wavelength units'] = 'Unknown'
        anc_header['wavelength'] = np.nan
        anc_header['data type'] = 4  # Assuming float32 data type

        #output_name = f"{args.output_dir}{os.path.basename(os.path.splitext(hy_obj.file_name)[0])}_ancillary"

        output_name = f"{args.output_dir}_ancillary"
        
        writer = WriteENVI(output_name, anc_header)
        
        # Write bands
        writer.write_band(path_length_data, 0)
        writer.write_band(to_sensor_azimuth_angle_data, 1)
        writer.write_band(to_sensor_zenith_angle_data, 2)
        writer.write_band(solar_azimuth_angle, 3)
        writer.write_band(solar_zenith_angle, 4)
        writer.write_band(slope_data, 5)
        writer.write_band(aspect_data, 6)
        writer.write_band(illumination_factor, 7)
        # Add other bands as needed
        
        writer.close()


def main():
    """
    Main function to convert NEON AOP H5 files to ENVI format and optionally export ancillary data.

    This function uses command line arguments to specify input images and output directory. It supports parallel processing using Ray for efficiency.
    """
    parser = argparse.ArgumentParser(description = "Convert NEON AOP H5 to ENVI format")
    parser.add_argument('images', help="Input image pathnames", nargs='*')
    parser.add_argument('output_dir', help="Output directory", type=str)
    parser.add_argument("-anc", help="Output ancillary", required=False, action='store_true')

    args = parser.parse_args()

    if not args.output_dir.endswith("/"):
        args.output_dir += "/"

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=len(args.images))

    hytool = ray.remote(ht.HyTools)
    actors = [hytool.remote() for image in args.images]
    _ = ray.get([a.read_file.remote(image, 'neon') for a, image in zip(actors, args.images)])

    def neon_to_envi(hy_obj):
        #basemame = os.path.basename(os.path.splitext(hy_obj.file_name)[0])
        basemame = "ENVI"
        print("Exporting %s " % basemame)
        output_name = args.output_dir + basemame
        writer = WriteENVI(output_name, hy_obj.get_header())
        iterator = hy_obj.iterate(by='chunk')
        pixels_processed = 0
        while not iterator.complete:
            chunk = iterator.read_next()
            pixels_processed += chunk.shape[0] * chunk.shape[1]
            writer.write_chunk(chunk, iterator.current_line, iterator.current_column)
            if iterator.complete:
                writer.close()

    _ = ray.get([a.do.remote(neon_to_envi) for a in actors])

    if args.anc:
        print("\nExporting ancillary data")
        _ = ray.get([a.do.remote(export_anc, (image, args)) for a, image in zip(actors, args.images)])

    print("Export complete.")

if __name__ == "__main__":
    main()
