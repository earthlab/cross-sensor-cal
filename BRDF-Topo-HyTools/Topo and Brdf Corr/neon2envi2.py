import argparse
import os
import h5py
import ray
import numpy as np
import hytools as ht
from hytools.io.envi import WriteENVI

def get_actual_key(h5_file, expected_key):
    actual_keys = {key.lower(): key for key in h5_file.keys()}
    return actual_keys.get(expected_key.lower(), expected_key)

def get_all_solar_angles(logs_group):
    angles = []
    for log_entry in logs_group.keys():
        angles.append((
            logs_group[log_entry]["Solar_Azimuth_Angle"][()],
            logs_group[log_entry]["Solar_Zenith_Angle"][()]
        ))
    return np.array(angles)


def export_anc(hy_obj, args_and_image):
    h5_file_path, args = args_and_image

    with h5py.File(hy_obj.file_name, 'r') as h5_file:
        anc_imagery_path = '/NIWO/Reflectance/Metadata/Ancillary_Imagery/'
        
        # Fetching data from the provided h5 file structure
        path_length_data = h5_file[anc_imagery_path + 'Path_Length'][...]
        slope_data = h5_file[anc_imagery_path + 'Slope'][...]
        aspect_data = h5_file[anc_imagery_path + 'Aspect'][...]

        illumination_factor = h5_file[anc_imagery_path + 'Illumination_Factor'][...]
        to_sensor_azimuth_angle_data = h5_file['/NIWO/Reflectance/Metadata/to-sensor_azimuth_angle'][...]
        to_sensor_zenith_angle_data = h5_file['/NIWO/Reflectance/Metadata/to-sensor_zenith_angle'][...]

        logs_group = h5_file['/NIWO/Reflectance/Metadata/Logs']
        for log_entry in logs_group.keys():
            solar_azimuth_angle = logs_group[log_entry]["Solar_Azimuth_Angle"][()]
            solar_zenith_angle = logs_group[log_entry]["Solar_Zenith_Angle"][()]
            
            # Prepare ancillary header
            anc_header = hy_obj.get_header()
            anc_header['bands'] = 8
            anc_header['band_names'] = ['path length', 'to-sensor azimuth', 'to-sensor zenith', 'solar azimuth', 'solar zenith',
                                        'phase', 'slope', 'aspec    t', 'cosine_i']
            anc_header['wavelength units'] = np.nan
            anc_header['wavelength'] = np.nan
            anc_header['data type'] = 4

            output_name = f"{args.output_dir}{os.path.basename(os.path.splitext(hy_obj.file_name)[0])}_{log_entry}_ancillary"
            writer = WriteENVI(output_name, anc_header)
            
            # Write bands
            writer.write_band(path_length_data, 0)
            writer.write_band(to_sensor_azimuth_angle_data, 1)
            writer.write_band(to_sensor_zenith_angle_data, 2)
            writer.write_band(solar_azimuth_angle, 3)
            writer.write_band(solar_zenith_angle, 4)
            writer.write_band(slope_data, 5)
            writer.write_band(aspect_data, 6)
            writer.write_band(illumination_factor,7)
            
            writer.close()

            # # Create symlink for the main file
            # main_file_path = os.path.join(args.output_dir, os.path.basename(os.path.splitext(hy_obj.file_name)[0]))
            # print(main_file_path)
            # symlink_name = os.path.join(args.output_dir, os.path.basename(os.path.splitext(hy_obj.file_name)[0]) + f"_{log_entry}")
            # if not os.path.exists(symlink_name):
            #     os.symlink(main_file_path, symlink_name)

            # # Create symlink for the main hdr file
            # main_hdr_file_path = os.path.join(args.output_dir, os.path.basename(os.path.splitext(hy_obj.file_name)[0]) + ".hdr")
            # print(main_hdr_file_path)
            # symlink_name_hdr = os.path.join(args.output_dir, os.path.basename(os.path.splitext(hy_obj.file_name)[0]) + f"_{log_entry}.hdr")
            # if not os.path.exists(symlink_name_hdr):
            #     os.symlink(main_hdr_file_path, symlink_name_hdr)

def main():
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
        basemame = os.path.basename(os.path.splitext(hy_obj.file_name)[0])
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