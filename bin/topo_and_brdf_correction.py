from argparse import ArgumentParser
import json
from src.topo_and_brdf_correction import topo_and_brdf_correction

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, help='''
        Path to a JSON configuration file for TOPO and BRDF corrections.
        The configuration file should include:
        - "input_files": List of input file paths.
        - "num_cpus": Number of CPUs for parallel processing.
        - "file_type": "envi" or "neon".
        - Correction parameters and settings.
        - Export settings for corrected images and coefficients.
    ''')
    args = parser.parse_args()

    # Validate config before running
    with open(args.config_file, 'r') as f:
        config_data = json.load(f)
    if "input_files" not in config_data:
        raise ValueError(f"❌ Config file {args.config_file} is missing 'input_files'. Please regenerate the config.")
    if not config_data["input_files"]:
        raise ValueError(f"❌ Config file {args.config_file} has an empty 'input_files' list.")

    topo_and_brdf_correction(args.config_file)
