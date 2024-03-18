import argparse
import os
import h5py
import ray
import numpy as np
import hytools as ht
from hytools.io.envi import WriteENVI
import re

def get_actual_key(h5_file, expected_key):
    """Attempt to find a key in the HDF5 file that matches the expected key, regardless of case sensitivity."""
    actual_keys = {key.lower(): key for key in h5_file.keys()}
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

def get_actual_key(h5_file, expected_key):
    """Attempt to find a key in the HDF5 file that matches the expected key, regardless of case sensitivity."""
    actual_keys = {key.lower(): key for key in h5_file.keys()}
    return actual_keys.get(expected_key.lower())

def extract_site_code_from_filename(filename):
    """Extracts the NEON site code from the given filename."""
    match = re.search(r"NEON_D\d+_([A-Z]+)_", filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Site code not found in filename: {filename}")

def export_anc(h5_file_path, site_code, output_dir):
    """Exports ancillary data from an HDF5 file to ENVI format."""
    with h5py.File(h5_file_path, 'r') as h5_file:
        base_path = f"/{site_code}/Reflectance/Metadata/Ancillary_Imagery/"
        
        # Use get_actual_key to handle potential case sensitivity issues
        path_length_key = get_actual_key(h5_file, f"{base_path}Path_Length")
        slope_key = get_actual_key(h5_file, f"{base_path}Slope")
        aspect_key = get_actual_key(h5_file, f"{base_path}Aspect")

        path_length_data = h5_file[path_length_key][...] if path_length_key else np.array([])
        slope_data = h5_file[slope_key][...] if slope_key else np.array([])
        aspect_data = h5_file[aspect_key][...] if aspect_key else np.array([])

        # Prepare ancillary header
        bands = 3  # Number of datasets to write as bands
        header = {
            'description': f'Ancillary data for {site_code}',
            'bands': bands,
            'band names': ['Path Length', 'Slope', 'Aspect'],
            'data type': 4,  # Assuming float32 data type
            'interleave': 'bil',
        }

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        output_name = os.path.join(output_dir, os.path.splitext(os.path.basename(h5_file_path))[0] + "_ancillary")
        writer = WriteENVI(output_name + ".hdr", header)
        
        # Write the data as separate bands
        if path_length_data.size > 0: writer.write_band(path_length_data, 0)
        if slope_data.size > 0: writer.write_band(slope_data, 1)
        if aspect_data.size > 0: writer.write_band(aspect_data, 2)
        
        writer.close()
        print(f"Ancillary data exported to {output_name}")

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
    
def main():
    """
    Main function to convert NEON AOP H5 files to ENVI format and optionally export ancillary data.
    """
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
        neon_to_envi(image_path, specific_output_dir)
        print(f"Processed {image_path} for site {site_code} with output in {specific_output_dir}")

        # If ancillary data flag is set, export ancillary data
        if args.anc:
            # Here you would call your function to export ancillary data
            # Make sure to define or import `export_anc` function appropriately
            export_anc({'h5_file_path': image_path, 'site_code': site_code, 'output_dir': specific_output_dir})
            print(f"Exported ancillary data for {image_path}")

    print("Processing complete.")

if __name__ == "__main__":
    main()