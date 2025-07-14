from argparse import ArgumentParser

from src.topo_and_brdf_correction import topo_and_brdf_correction

if __name__== "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, help='''
        A single command line argument specifying the path to the JSON configuration file.
        The configuration file should contain:
        - Input file paths.
        - Number of CPUs to use for processing.
        - File type (e.g., 'envi', 'neon').
        - Correction parameters and settings.
        - Export settings for the corrected images and coefficients.
    ''')
    args = parser.parse_args()

    topo_and_brdf_correction(args.config_file)
