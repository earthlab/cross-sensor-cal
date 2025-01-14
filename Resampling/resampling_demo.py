conda_env_path = '/opt/conda/envs/macrosystems/bin/python'

import argparse
from hytools.transform.resampling import calc_resample_coeffs
import numpy as np
import spectral
import json 
import matplotlib.pyplot as plt






def apply_resampler(hy_obj,data):
    ''' Apply SCSS correction to a slice of the data

    Args:
        hy_obj (TYPE): DESCRIPTION.
        band (TYPE): DESCRIPTION.
        index (TYPE): DESCRIPTION.

    Returns:
        band (TYPE): DESCRIPTION.

    '''

    interp_types = ['linear', 'nearest', 'nearest-up',
                   'zero', 'slinear', 'quadratic',
                   'cubic']

    #Convert to float
    data = data.astype(np.float32)

    if hy_obj.resampler['type'] == 'gaussian':

        # Load resampling coeffs to memory if needed
        if 'resample_coeffs' not in hy_obj.ancillary.keys():
            in_wave = hy_obj.wavelengths[~hy_obj.bad_bands]
            in_fwhm =hy_obj.fwhm[~hy_obj.bad_bands]
            resample_coeffs = calc_resample_coeffs(in_wave,in_fwhm,
                                             hy_obj.resampler['out_waves'],
                                             hy_obj.resampler['out_fwhm'])
            hy_obj.ancillary['resample_coeffs'] = resample_coeffs

        data = np.dot(data, hy_obj.ancillary['resample_coeffs'] )


    elif hy_obj.resampler['type'] in interp_types:
        interp_func =  interp1d(hy_obj.wavelengths[~hy_obj.bad_bands], data,
                                kind=hy_obj.resampler['type'],
                                axis=2, fill_value="extrapolate")
        data = interp_func(hy_obj.resampler['out_waves'])

    return data

pass

class resampler_hy_obj:

    def __init__(self, sensor_type, json_file, hdr_path):
        """
            Initializes the resampler object with the specified sensor type, JSON file, and HDR file path.
            - sensor_type: The type of the sensor (e.g., 'Landsat 8 OLI') for which resampling is to be performed.
            - json_file: Path to the JSON file containing Landsat band parameters (wavelengths and FWHMs).
            - hdr_path: Path to the HDR file which contains wavelength information for the sensor data.

            The method performs several key steps:
            1. Reads and converts wavelengths from the HDR file into a NumPy array.
            2. Sets up a fixed number of bands and a default FWHM value, creating an array for these FWHMs.
            3. Initializes a dictionary to store ancillary data and an array to flag bad bands.
            4. Reads the JSON file to get the band parameters for the given sensor type.
            5. Checks if the sensor type is valid and present in the JSON data.
            6. Extracts output wavelengths and FWHM values for the specified sensor type and stores them.
            7. Configures the resampler settings (type, output wavelengths, and FWHMs).

            The configuration for resampling is set up based on the Landsat sensor 
            type specified. This includes defining the output wavelengths and FWHMs which are crucial
            for the resampling process.
        """
        self.wavelengths = np.array(self.get_in_wavelength(hdr_path), dtype=np.float32)

        num_of_bands = 426
        fwhm_value = 5
        self.fwhm = np.array(self.get_in_fwhm(num_of_bands, fwhm_value), dtype=np.float32)
        self.ancillary = {}
        self.bad_bands = np.full(num_of_bands, False)

        with open(json_file, 'r') as file:
            band_parameters = json.load(file)

        if sensor_type not in band_parameters:
            raise ValueError(f"Sensor type {sensor_type} not found in JSON data.")

        self.out_wave = np.array(band_parameters[sensor_type]['wavelengths'], dtype=np.float32)
        self.out_fwhm = np.array(band_parameters[sensor_type]['fwhms'], dtype=np.float32)

        self.resampler = {
            'type': 'gaussian',
            'out_waves': self.out_wave,
            'out_fwhm': self.out_fwhm
        }

    @staticmethod
    def get_in_wavelength(hdr_path):
        """
            Reads and extracts wavelength information from an ENVI header file.

            - hdr_path: The file path to the ENVI header (.hdr) file.

            The method performs the following steps:
            1. Reads the header file using the Spectral Python (SPy) library's `read_envi_header` function.
            2. Extracts the wavelength information, which is typically a list of strings representing wavelength values.
            3. Converts these string values to floats, making them usable for numerical operations.

            Returns:
            - A list of wavelength values as floats if successfully extracted.

            If the wavelength information is not found in the header file, the method raises a ValueError, indicating the absence of this crucial data. This method is essential for processing spectral data as it provides the fundamental wavelength values needed for various analytical processes, such as resampling.
        """
        
        header = spectral.envi.read_envi_header(hdr_path)

        # Extract wavelength information
        wavelengths = header.get('wavelength')

        if wavelengths:
            # Convert string values to floats
            wavelengths = [float(w) for w in wavelengths]
            return wavelengths
        else:
            raise ValueError("Wavelength information not found in the header file.")

    @staticmethod
    def get_in_fwhm(number_of_bands, fwhm_value):
        """
            Generate a list of FWHM values for each band in hyperspectral data.

            :param number_of_bands: int
                The total number of bands in the hyperspectral data.
            :param fwhm_value: float, optional (default is 5)
                The FWHM value for each band (in nm).

            :return: list
                A list of FWHM values for each band.
        """
        fwhm_list = [fwhm_value] * number_of_bands
        return fwhm_list
    
    def create_header_info(self, hdr_path):
        """
        Create header information for the resampled data.
        :param hdr_path: string
            Path and file name of the hdr file
        :return: dict
            Dictionary containing header information for the ENVI file, including CRS.
        """
        header = spectral.envi.read_envi_header(hdr_path)
    
        # Extract the number of lines and samples
        lines = header.get('lines')
        samples = header.get('samples')
    
        # Extract CRS information if available
        map_info = header.get('map info', None)  # 'map info' contains georeferencing details
        coordinate_system_string = header.get('coordinate system string', None)  # Optional CRS details
    
        header_info = {
            'description': 'Resampled hyperspectral data',
            'file type': 'ENVI Standard',
            'sensor type': 'landsat',
            'bands': len(self.out_wave),
            'lines': lines,  # Replace with actual number of lines in your data
            'samples': samples,  # Replace with actual number of samples in your data
            'header offset': 0,
            'data type': 4,  # Data type (e.g., 4 for float32)
            'interleave': 'bil',
            'byte order': 0,
            'wavelength units': 'Nanometers',
            'wavelength': self.out_wave,
        }
    
        # Add georeferencing and CRS details if available
        if map_info:
            header_info['map info'] = map_info
        if coordinate_system_string:
            header_info['coordinate system string'] = coordinate_system_string
    
        return header_info

    
    @staticmethod    
    def save_envi_data(data, header_info, output_filename):
        """
            Save data in ENVI format.

            :param data: The hyperspectral data to save.
            :param header_info: Header information for the data.
            :param output_filename: The filename for the saved data.
        """
        spectral.envi.save_image(output_filename, data, metadata=header_info, force=True)

def load_envi_data(filename):
        """
            Load hyperspectral data from an ENVI binary file.

            :param filename: The filename of the ENVI binary file (without the .hdr extension).
            :return: The loaded hyperspectral data as a spectral image object.
        """
        # Add .hdr extension to get the header file
        hdr_filename = filename + '.hdr'
        # Read the ENVI data
        img = spectral.open_image(hdr_filename)
        # Load the data into memory (optional, depends on the data size and memory capacity)
        data = img.load()
        return data

# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Landsat data resampling script')
parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON file with Landsat band parameters')
parser.add_argument('--hdr_path', type=str, required=True, help='Path to the HDR file')
parser.add_argument('--sensor_type', type=str, required=True, help='Give the type of sensor')
parser.add_argument('--resampling_file_path', type=str, required=True, help='Path to the resampling file')
parser.add_argument('--output_path', type=str, required=True, help='Path for the output file')
args = parser.parse_args()

# Using arguments in the script
json_file = args.json_file
hdr_path = args.hdr_path
sensor_type = args.sensor_type
resampling_file_path = args.resampling_file_path
output_path = args.output_path

# create the obj
resampler_config_obj = resampler_hy_obj(sensor_type, json_file, hdr_path)

# Generate header information
header_info = resampler_config_obj.create_header_info(hdr_path)
print(header_info)
# call the resampling code # Usage
filename = resampling_file_path


data = load_envi_data(filename)
final_resampled_data = apply_resampler(resampler_config_obj, data)

#plt.plot(final_resampled_data)
#plt.ylim(0,.6)

output_file_path = output_path
# save the data after resampling
resampler_config_obj.save_envi_data(data=final_resampled_data, header_info=header_info, output_filename = output_path)
