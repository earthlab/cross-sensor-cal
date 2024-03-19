import argparse
import os
import h5py
from h5py import Dataset
import ray
import numpy as np
import hytools as ht
from hytools.io.envi import WriteENVI
import re


def get_all_keys(group):
    if isinstance(group, Dataset):
        return [group.name]

    all_keys = []
    for key in group.keys():
        all_keys += get_all_keys(group[key])

    return all_keys


def get_actual_key(h5_file, expected_key):
    """Attempt to find a key in the HDF5 file that matches the expected key, regardless of case sensitivity."""
    actual_keys = {key.lower(): key for key in get_all_keys(h5_file)}
    return actual_keys.get(expected_key.lower())


def get_all_solar_angles(logs_group):
    angles = []
    for log_entry in logs_group.keys():
        angles.append((logs_group[log_entry]["Solar_Azimuth_Angle"][()], logs_group[log_entry]["Solar_Zenith_Angle"][()]))
    return np.array(angles)


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def print_all_keys(h5_file, indent=0):
    """Recursively prints all keys in an HDF5 file or group."""
    for key in h5_file.keys():
        print('  ' * indent + key)
        if isinstance(h5_file[key], h5py.Group):
            print_all_keys(h5_file[key], indent + 1)


def extract_site_code_from_filename(filename):
    """Extracts the NEON site code from the given filename."""
    match = re.search(r"NEON_D\d+_([A-Z]+)_", filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Site code not found in filename: {filename}")


def neon_to_envi(image_path, output_dir):
    # Create HyTools object and load the image
    hy_obj = ht.HyTools()
    hy_obj.read_file(image_path, 'neon')

    # Construct output file name
    basename = os.path.basename(os.path.splitext(image_path)[0])
    output_name = os.path.join(output_dir, basename)
    
    # Initialize ENVI writer with header from HyTools object
    writer = WriteENVI(output_name, hy_obj.get_header())
    
    # Process and write chunks
    pixels_processed = 0
    iterator = hy_obj.iterate(by='chunk')
    while not iterator.complete:
        chunk = iterator.read_next()
        pixels_processed += chunk.shape[0] * chunk.shape[1]
        writer.write_chunk(chunk, iterator.current_line, iterator.current_column)
        if iterator.complete:
            writer.close()

    print(f"Exported {basename} to ENVI format.")


def envi_header_for_ancillary(hy_obj, ancillary_attributes, interleave='bil'):
    """
    Create an ENVI header dictionary specifically for ancillary data derived from NEON metadata.

    Args:
        hy_obj (Hytools object): Populated HyTools file object, used for base metadata.
        ancillary_attributes (dict): Attributes specific to the ancillary data.
        interleave (str, optional): Data interleave type. Defaults to 'bil'.

    Returns:
        dict: Populated ENVI header dictionary for ancillary data.
    """
    header_dict = dict()
    header_dict["ENVI description"] = "Ancillary data"
    header_dict["samples"] = ancillary_attributes['samples']  # Expected to be provided
    header_dict["lines"] = ancillary_attributes['lines']  # Expected to be provided
    header_dict["bands"] = len(ancillary_attributes['band names'])  # Based on provided band names
    header_dict["header offset"] = 0
    header_dict["file type"] = "ENVI Standard"
    header_dict["data type"] = ancillary_attributes['data type']  # Should match the ancillary data
    header_dict["interleave"] = interleave
    header_dict["byte order"] = 0  # Assuming little endian; adjust as necessary
    # These could be derived from hy_obj if applicable or set as needed
    header_dict["map info"] = ancillary_attributes.get('map info', "")
    header_dict["coordinate system string"] = ancillary_attributes.get('coordinate system string', "")
    header_dict["wavelength units"] = "Unknown"  # Ancillary data might not have wavelength information
    header_dict["band names"] = ancillary_attributes['band names']  # Should be provided
    # Include any other necessary fields...

    return header_dict


def export_anc(h5_file_path, site_code, output_dir):
    print("EXP ANC")
    print(h5_file_path)
    """Exports ancillary data from an HDF5 file to ENVI format."""
    with h5py.File(h5_file_path, 'r') as h5_file:
        base_path = f"/{site_code}/Reflectance/Metadata/Ancillary_Imagery/"
        print(base_path)

        path_length_key = get_actual_key(h5_file, f"{base_path}Path_Length")
        print(path_length_key)
        slope_key = get_actual_key(h5_file, f"{base_path}Slope")
        print(slope_key)
        aspect_key = get_actual_key(h5_file, f"{base_path}Aspect")
        print(aspect_key)

        path_length_data = h5_file[path_length_key][...] if path_length_key else np.array([], dtype=np.float32)
        slope_data = h5_file[slope_key][...] if slope_key else np.array([], dtype=np.float32)
        aspect_data = h5_file[aspect_key][...] if aspect_key else np.array([], dtype=np.float32)

        # Initialize the HyTools object and read the NEON file for metadata
        hy_obj = ht.HyTools()
        hy_obj.read_file(h5_file_path, 'neon')

        # Define ancillary attributes based on the specific datasets
        ancillary_attributes = {
            'samples': max([d.shape[1] for d in [path_length_data, slope_data, aspect_data] if d.size > 0], default=0),
            'lines': max([d.shape[0] for d in [path_length_data, slope_data, aspect_data] if d.size > 0], default=0),
            'data type': 4,  # ENVI data type code for float32
            'band names': ['Path Length', 'Slope', 'Aspect'],  # Add more names if needed
            # Include any additional ancillary-specific attributes here
        }

        # Create the header for ancillary data
        header = envi_header_for_ancillary(hy_obj, ancillary_attributes, interleave='bil')

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        output_name = os.path.join(output_dir, os.path.splitext(os.path.basename(h5_file_path))[0] + "_ancillary")
        
        # Initialize the ENVI writer with the generated header
        print(header)
        writer = WriteENVI(output_name + ".hdr", header)
        
        # Write the data as separate bands, ensuring data is the correct type
        if path_length_data.size > 0: writer.write_band(path_length_data, 0)
        if slope_data.size > 0: writer.write_band(slope_data, 1)
        if aspect_data.size > 0: writer.write_band(aspect_data, 2)
        
        writer.close()
        print(f"Ancillary data exported to {output_name}")


def main():
    """
    Main function to convert NEON AOP H5 files to ENVI format and optionally export ancillary data.
    """
    print("Here we GO!")
    parser = argparse.ArgumentParser(description="Convert NEON AOP H5 to ENVI format and optionally export ancillary data.")
    parser.add_argument('--images', nargs='+', required=True, help="Input image pathnames")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('-anc', action='store_true', help="Flag to output ancillary data", required=False)

    args = parser.parse_args()

    for image_path in args.images:
        # Extract site code from filename
        site_code = extract_site_code_from_filename(image_path)
        
        # Construct specific output directory for each image
        specific_output_dir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(image_path))[0])
        
        # Ensure the specific output directory exists
        if not os.path.exists(specific_output_dir):
            os.makedirs(specific_output_dir, exist_ok=True)
        
        # Process image to ENVI format
        print(f"Converting {image_path} for site {site_code} to ENVI: {specific_output_dir}")
        neon_to_envi(image_path, specific_output_dir)
        print(f"Finished processing {image_path} conversion to {specific_output_dir}")

        # If ancillary data flag is set, export ancillary data
        if args.anc:
            print(f"Processing ancillary data for {image_path}")
            export_anc(image_path, site_code, specific_output_dir)
            print(f"Finished processing ancillary data for {image_path}")

    print("Processing complete.")


if __name__ == "__main__":
    main()