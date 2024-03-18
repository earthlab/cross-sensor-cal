import argparse
import os
import h5py
import numpy as np
from hytools.io.envi import WriteENVI

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

def export_anc(args):
    h5_file_path = args['h5_file_path']
    site_code = args['site_code']
    output_dir = args['output_dir']

    ensure_directory_exists(output_dir)
    
    output_name = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(h5_file_path))[0]}_ancillary")

    with h5py.File(h5_file_path, 'r') as h5_file:
        base_path = f"{site_code}/Reflectance/Metadata/Ancillary_Imagery/"
        path_length_data = h5_file[f"{base_path}Path_Length"][...]
        slope_data = h5_file[f"{base_path}Slope"][...]
        aspect_data = h5_file[f"{base_path}Aspect"][...]

        # Assuming the datasets have similar shapes and types
        bands = 3  # Number of datasets to write as bands
        dtype = path_length_data.dtype  # Assuming all datasets have the same dtype
        shape = path_length_data.shape

        header = {
            'description': f'Ancillary data for {site_code}',
            'samples': shape[1],
            'lines': shape[0],
            'bands': bands,
            'data type': 4,  # Change as per dtype if necessary
            'interleave': 'bil',
            'band names': ['Path Length', 'Slope', 'Aspect']
        }
        
        writer = WriteENVI(f"{output_name}.hdr", header)
        
        # Write the data as separate bands
        writer.write_band(path_length_data, 0)
        writer.write_band(slope_data, 1)
        writer.write_band(aspect_data, 2)
        
        writer.close()
        
        print(f"Ancillary data written to {output_name}.hdr and associated .dat file.")

def main():
    parser = argparse.ArgumentParser(description="Convert NEON AOP H5 files to ENVI format and export ancillary data if specified.")
    parser.add_argument('--images', nargs='+', help="Input image pathnames", required=True)
    parser.add_argument('--output_dir', help="Output directory", required=True)
    parser.add_argument('--site_code', help="NEON site code (e.g., 'NIWO')", required=True)
    parser.add_argument('-anc', action='store_true', help="Flag to output ancillary data", required=False)

    args = parser.parse_args()


    if args.anc:
        for image_path in args.images:
            anc_args = {
                'h5_file_path': image_path,
                'site_code': args.site_code,
                'output_dir': args.output_dir
            }
            export_anc(anc_args)

    print("Processing complete.")

if __name__ == "__main__":
    main()
