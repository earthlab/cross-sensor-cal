
### README for Landsat Resampling Tools

#### Overview
This package contains two main files: `landsat_band_parameters.json` and `resampling_deom.py`. These are designed to facilitate the resampling of Landsat satellite data using the HyTools library.

#### File Descriptions
1. **landsat_band_parameters.json**: This JSON file contains parameters related to various Landsat missions (Landsat 5 TM, Landsat 7 ETM+, Landsat 8 OLI, and Landsat 9 OLI-2). For each mission, it lists the wavelengths (in nanometers) and full-width at half maximum (FWHM) values for different bands.

2. **resampling_deom.py**: A Python script that implements a resampling process for Landsat imagery. It defines a class `resampler_hy_obj` that initializes with sensor type and the JSON file path. The script uses functions from the HyTools library to apply resampling to the satellite data.

#### Prerequisites
- Python 3.x
- NumPy
- Spectral Python (SPy)
- HyTools library

#### Installation
1. Ensure Python 3.x is installed on your system.
2. Install the required Python libraries:
   ```
   pip install numpy spectral hytools
   ```

#### Usage
1. Place both the `landsat_band_parameters.json` and `resampling_deom.py` in your working directory.
2. Import the `resampler_hy_obj` class from the `resampling_deom.py` script in your Python environment.
3. Initialize an instance of `resampler_hy_obj` with the desired sensor type and the path to the JSON file.
4. Apply the resampling process to your Landsat data using methods from this class.

#### Example
```python
from resampling_deom import resampler_hy_obj

# Initialize resampler object for Landsat 8 OLI
resampler = resampler_hy_obj(sensor_type='Landsat 8 OLI', json_file='landsat_band_parameters.json')

# Apply resampling (add details based on your data and requirements)
```