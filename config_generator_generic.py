import json
import glob
import numpy as np
import os

def generate_configurations(input_dir):
    """
    Generates configuration files for main and ancillary image data in the same directory as the input files.
    
    Args:
        input_dir (str): Directory containing the main and ancillary files.
    """
    # List all files in the input directory
    all_files = glob.glob(os.path.join(input_dir, "*"))
    all_files.sort()

    for file_path in all_files:
        config_dict = {}
        base_name = os.path.basename(file_path)

        # Check if the file is an ancillary file
        if '_ancillary' in file_path:
            # Prepare the configuration for an ancillary file
            config_file = os.path.join(input_dir, f"{base_name.split('.')[0]}_ancillary.json")
            # Example configuration settings specific to ancillary data
            config_dict['type'] = 'ancillary'
        else:
            # Prepare the configuration for the main image file
            config_file = os.path.join(input_dir, f"{base_name.split('.')[0]}.json")
            # Example configuration settings specific to main image data
            config_dict['type'] = 'main'

        # Common configuration settings
        config_dict['file_type'] = 'envi'
        config_dict['input_file'] = file_path
        config_dict['bad_bands'] = [[300, 400], [1337, 1430], [1800, 1960], [2450, 2600]]
        # Add other necessary configurations here...

        # Avoid overwriting original files
        if not config_file == file_path:
            # Save the configuration to a JSON file
            with open(config_file, 'w') as outfile:
                json.dump(config_dict, outfile, indent=3)
            print(f"Configuration saved to {config_file}")
        else:
            print(f"Skipped writing configuration for {file_path} to avoid overwriting.")

# Example usage
input_dir = 'path/to/your/input/directory'  # Update this to your actual input directory path
generate_configurations(input_dir)
