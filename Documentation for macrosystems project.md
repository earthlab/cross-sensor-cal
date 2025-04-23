# Documenation of Work Done for Macrosystems project

## Table of Contents
1. [Environment Setup for Macrosystems Project](#environment-setup-for-macrosystems-project)
2. [Topographic Correction using methods for NIWO (multiple flightlines)](#topographic-correction-using-methods-for-niwo-multiple-flightlines)
3. [Calculating Sun Angles](#calculating-sun-angles)
4. [Extracting Slope and Aspect for Drone Data using DEM](#extracting-slope-and-aspect-for-drone-data-using-dem)
5. [Topographic Correction using Methods for Drone Data](#topographic-correction-using-methods-for-drone-data)
6. [Topo and BRDF Correction using Hytools (Steps to Use It)](#topo-and-brdf-correction-using-hytools-steps-to-use-it)
7. [Resampling](#resampling)
8. [NEON Data Access using API](#neon-data-access-using-api)
9. [About Cyverse](#about-cyverse)

## Environment Setup for Macrosystems Project
This section details the steps for setting up the environment required for the Macrosystems project. Follow these steps to ensure a smooth and consistent development environment.
### Dependencies
Before starting, make sure you have the [environment.yml](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/environment.yaml) file. This file contains all the necessary dependencies.

### Step-by-Step Guide
1.  Prepare the Environment File
Locate your environment.yml file. This should include all dependencies from name: cires-demo to the prefix line.

2.  Open Command Line Interface
    Depending on your operating system, use the following:
    Windows Users: Anaconda Prompt or Command Prompt.
    macOS/Linux Users: Terminal application.

3.  Navigate to File Location: 
    In your CLI, navigate to the directory containing environment.yml using the cd command.

4.  Create the Conda Environment: Run the following command to create the Conda environment:

    ``` conda env create -f environment.yaml ```

    This will set up an environment named cires-demo and install all necessary packages.

5.  Activate the Environment
    Switch to your new environment with:
    ```conda activate cires-demo ``` 
    It's important to activate cires-demo to access the installed packages.

```conda activate macrosystems```
6.  Verify Installation
    To check if all packages are installed correctly:
    ``` conda list```
    This command lists all packages in the cires-demo environment.

7.  Environment Ready
    After these steps, your environment is set up and ready for project work.


## BASH
```
conda activate macrosystems
pip install spectral
```

```
conda activate macrosystems
bash correct.sh NEON_D13_NIWO_DP1_20200801_161441_reflectance.h5 "NIWO"
```

```bash resample.sh export/ENVI__corrected_0```


## Topographic Correction using methods for NIWO (multiple flightlines)
**Data Product Name:** NEON_D13_NIWO_DP3_449000_4435000_reflectance.h5

This section outlines the methods used for topographic correction on the specified NEON data product. Two primary methods are implemented: SCS (Sun-Canopy-Sensor) and SCS+C (Sun-Canopy-Sensor + Cosine).

### SCS (Sun-Canopy-Sensor) Topographic Correction
**Objective:** Correct variations in reflectance caused by topographic effects like slope and aspect. 

**Implementation:**
Parameters such as solar zenith angle are averaged across multiple flightlines, extracted from the NEON file metadata.

The notebook includes:
*   Extract Parameters from Metadata: Outline the process for extracting necessary parameters from the NEON data metadata.

*   Topographic Correction using SCS Method: Detailed steps and code implementation for applying the Sun-Canopy-Sensor (SCS) method for topographic correction.

*   Function for Plotting Aspect and Illumination: A function to visualize aspect, illumination alongside original and corrected reflectance values.

*   Statistical Analysis of Pixel Values: Analyzing pixel value distributions before and after the topographic correction.

*   Correlation Analysis: Evaluating the correlation between different variables in the dataset.

*   NDVI Analysis: Application of the Normalized Difference Vegetation Index (NDVI) on the dataset.

For a detailed walkthrough, see the notebook: [Topo_Corr_SCS.ipynb](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/Topo_Corr_using_Methods/Topo_Corr_SCS_final.ipynb)


### SCS+C (Sun-Canopy-Sensor + Cosine) Topographic Correction
**Objective:** Extend the SCS method by including a cosine correction factor, enhancing effectiveness in rugged terrains.

**Implementation:**

The notebook includes:

*   Parameters are derived from metadata within the NEON file, with solar zenith angles averaged over multiple flightlines.

*   Extract Parameters from Metadata: Similar to Notebook 1, detailing the extraction process of relevant parameters from the NEON data metadata.

*   Topographic Correction using SCS+C Method: Implementing the SCS+C method for topographic correction which includes an additional cosine correction factor.

*   Various Graphical Plots for Topographic Correction: Visualizing data and correction effects, especially focusing on NIR band 93.

*   Function for Plotting Aspect and Illumination: Similar to Notebook 1, but tailored for the SCS+C method.

*   Statistical Analysis of Pixel Values: Examining the impact of SCS+C correction on pixel values.

*   Correlation Analysis: Assessing correlations post-topographic correction using the SCS+C method.

*   NDVI Analysis and Correction: Detailed steps for applying NDVI on corrected data, including methods for choosing the nearest red and NIR bands.

*   Comparative NDVI Graphs: Visual comparison between reflectance NDVI and post-correction NDVI.

For an in-depth explanation and code, view: [Topo_Corr_SCS_C.ipynb](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/Topo_Corr_using_Methods/Topo_Corr_SCS_C_Final.ipynb)

## Calculating Sun angles

This section of the documentation outlines the methodologies used in the Jupyter notebook for calculating the sun angles, which is a crucial step in topographic and radiometric corrections for remote sensing data.

### Overview
The notebook provides two distinct methods for calculating sun angles, leveraging geographical data from drone imagery. It includes functions to extract latitude and longitude coordinates from the drone images and to compute sun angles for each pixel.

### Detailed Steps
1.  Extracting Latitude and Longitude from Drone Images:
Functionality is developed to extract geographic coordinates (latitude and longitude) directly from the metadata of drone imagery.

2.  Method 1: Basic Sun Angle Calculation
This method calculates the sun angles using a simplified approach, ideal for scenarios where detailed topographic information is not critical.

3.  Method 2: Advanced Sun Angle Calculation
An advanced technique for calculating sun angles, providing more accuracy and detail. This method is particularly useful for rigorous topographic and radiometric analyses.

4.  Function to Calculate Sun Angles for Each Image Pixel:
A comprehensive function that calculates the sun angles for every pixel in the drone image, using the extracted latitude and longitude coordinates. This function is essential for detailed pixel-by-pixel analysis in remote sensing applications.

### Application
These methods are crucial for understanding the solar illumination conditions for each pixel, which significantly impacts the reflectance values in remote sensing data. Accurate sun angle calculation allows for more precise corrections and analyses in subsequent steps of the project.

For an in-depth explanation and code, view: [calculating_sun_angles.ipynb](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/Topo_Corr_using_Methods/Calulating_Sun_angles.ipynb)



## Extracting Slope and Aspect for drone data using DEM

This section of the documentation details the process outlined in the Jupyter notebook for deriving slope and aspect data from Digital Elevation Models (DEM) applied to drone imagery. This procedure is crucial for understanding the topography of the area captured by drone data and getting slope and aspect for further corrections as they are the required parameters.

### Overview
The notebook demonstrates the procedure to calculate slope and aspect using DEM data, which are critical parameters in topographical analysis and can significantly impact the accuracy of remote sensing data interpretation.

### Detailed Steps
*   Loading the Digital Elevation Model (DEM) File: The first step involves loading the DEM file specified in the DEM_path variable. This file contains the elevation data necessary for calculating slope and aspect.
*   Calculating Slope and Aspect:The notebook provides code for calculating two key topographic parameters:
    *   Slope: This measures the steepness or degree of incline of the terrain. The slope is essential for understanding the terrain's gradient and is calculated in degrees.
    *   Aspect: This refers to the compass direction that the slope faces. Aspect is crucial for determining the direction of the sun's illumination on the terrain.
    
*   Output Path for Saving Results:
The notebook includes functionality to save the calculated slope and aspect data to specified output paths. This ensures that the results are stored for further analysis and use in the project.

For an in-depth explanation and code, view: [slope_aspect_drone_dtm.ipynb](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/Topo_Corr_using_Methods/Slope_Aspect_Drone_Data.ipynb)

### Application
Slope and aspect data extracted from DEM are fundamental in environmental and geographical studies, especially in projects involving remote sensing and aerial imagery. They provide insights into the terrain characteristics, which are vital for various analyses, including ecological studies, land-use planning, and agricultural assessments.

## Topographic Correction using methods for Drone Data

### Overview
The notebook focuses on applying the SCS+C method for correcting topographic effects in drone imagery. This method is particularly effective for landscapes with varied terrain, as it accounts for differences in sunlight angles and terrain features.

### Detailed Steps
1.  Assigning Paths for Drone Data, Slope, and Aspect:
Define file paths for the drone data, slope, and aspect datasets necessary for topographic correction.

2.  Setting Parameters for SCS+C Topographic Correction:
Outlines the parameters needed for the SCS+C method, including the reading of specific bands from drone data.

3.  Function for Slope and Aspect Extraction:
Describes the process of using rasterio to extract slope and aspect data, essential components for the SCS+C correction.

4.  Calculating Sun Angles:
Details the calculation of sun angles for each pixel in the drone imagery, a key factor in the topographic correction process.

5.  Gathering Parameters for Topographic Correction:
Consolidates all necessary parameters (including illumination, slope, aspect, and sun angles) for executing the SCS+C correction.

6.  Illumination Plotting:
Visualizing the illumination component, crucial for understanding the light dynamics over the terrain.

7.  Topographic Correction using SCS+C Method:
Implementation of the SCS+C topographic correction, including a detailed function to perform this correction on the drone data.

8.  Visual Analysis for NIR Band:
Includes plots and analysis focusing on the NIR band to assess the effectiveness of the topographic correction.
9.  NDVI Calculation:
Demonstrates the calculation of the Normalized Difference Vegetation Index (NDVI), both with and without a threshold, to analyze vegetation health post-correction.

For an in-depth explanation and code, view: [Topo_Corr_drone_data.ipynb](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/Topo_Corr_using_Methods/Topo_corr_Drone_data.ipynb)

#### Application
The application of topographic correction using the SCS+C method is vital in ensuring the accuracy of remote sensing data, especially in areas with complex terrain. This notebook provides a comprehensive guide for applying this correction to drone imagery, which can be crucial for environmental monitoring, agricultural assessments, and land-use studies.
## Topo and BRDF correction using Hytools (steps to use it)

### Overview
This documentation covers a comprehensive workflow for processing NEON data using Python scripts. The process involves converting NEON data to ENVI format, generating configuration files for topographic and BRDF corrections, and applying these corrections to the imagery.

### Python Scripts Description
0. Copy over files
```cp ~/data-store/data/iplant/home/shared/earthlab/macrosystems/Topo_Corr/NIWO_RMNP-Full-Data ~/data-store/cross-sensor-cal -r```

1. neon2envi2.py: NEON to ENVI Conversion, code: [neon2envi.py](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/BRDF-Topo-HyTools/Topo%20and%20Brdf%20Corr/neon2envi2.py)
Purpose: Converts NEON AOP H5 data files to ENVI format.
Usage:
Run the script from the command line with the dataset path and output folder.
Optional flag -anc to export ancillary data.
Example:
```python neon2envi2.py <path-to-dataset_name> <path-to-output_folder> -anc```
```python neon2envi2.py NEON_D13_NIWO_DP1_20170904_181819_reflectance.h5 output -anc```

2. config_generator.py: Configuration File Generation, code: [config_generator.py](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/BRDF-Topo-HyTools/Topo%20and%20Brdf%20Corr/config_generator.py)
Functionality: Generates JSON configuration files for applying topographic (TOPO) and Bidirectional Reflectance Distribution Function (BRDF) corrections.
Configuration Options: Includes settings for various correction types, wavelengths, and other parameters.
Running the Script: Edit the script according to the desired corrections and run it to create config_<iteration>.json files.
Example Command:
```python config_generator.py```

3. image_correct.py: Applying Corrections, code: [image_correct.py](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/BRDF-Topo-HyTools/Topo%20and%20Brdf%20Corr/image_correct.py)
Purpose: Reads the generated JSON configuration file and applies the specified TOPO and BRDF corrections to the imagery.
Execution: Run the script with the configuration file as a command-line argument.
```python image_correct.py <path-to-config-file>```
```python image_correct.py output/config_0.json``` 

```gdal_translate -of GTiff export/NEON_D13_NIWO_DP1_20170904_181819_reflectance__corrected_0 output_file.tif```
```gdal_translate -of GTiff output/NEON_D13_NIWO_DP1_20170904_181819_reflectance output_file.tif```

4. Copy export files to data-store
```cp ~/data-store/cross-sensor-cal/exports ~/data-store/data/iplant/home/shared/earthlab/macrosystems/Topo_Corr -r```

### Overview for config_generator.py
The config_generator.py script is designed to automate the creation of configuration files for topographic (TOPO) and Bidirectional Reflectance Distribution Function (BRDF) corrections of geospatial imagery. It allows customization to accommodate different correction methods and input formats.

#### TOPO Correction Methods
The script supports various TOPO correction methods, including SCS (Sun-Canopy-Sensor), SCS+C (Sun-Canopy-Sensor + Cosine), and C correction. For this project, the SCS+C method has been chosen due to its effectiveness in handling varied terrain by incorporating an additional cosine correction factor.

##### Key Features of SCS+C Method:
*   Accounts for solar zenith angle, slope, and aspect.
*   Adjusts reflectance values based on pixel-specific illumination conditions.
*   Particularly effective in landscapes with significant elevation changes.

#### BRDF Correction Methods
Two primary methods for BRDF correction are supported: the Universal method and the Flex method. In this project, the Flex method is used due to its adaptability and suitability for the specific requirements of NEON data.

##### Key Features of Flex Method:
*   Tailors BRDF corrections based on scene-specific characteristics.
*   Handles a wide range of surface and atmospheric conditions.
*   Note: Diagnostic plots are more challenging with the Flex method compared to the Universal method, as Flex returns different values that require extensive modifications to the HyTools library.

##### Customization and Preferences
Users can modify the config_generator.py script to choose their preferred methods for both TOPO and BRDF corrections. The script is structured to allow easy switching between different correction algorithms and settings.

#### Configuration File Generation:
*   The script generates JSON files for each ancillary file related to the main image.
*   Users can specify bad bands, file types, input files, and other settings.
*   The output includes detailed settings for export options, masks, coefficients, and correction-specific parameters.
#### Usage:
*   Ideal for workflows requiring specific topographic and BRDF corrections.
*   Users can edit the script to select desired correction methods and parameters.
*   The output JSON files serve as input for subsequent correction processes using tools like image_correct.py.

#### Flexibility and Extensibility:
The configuration generator script offers flexibility and extensibility, allowing users to adapt the correction process to their specific needs. By modifying the script, users can experiment with different correction methods and parameters, optimizing their workflow for the best possible results in geospatial imagery analysis.
### Steps to Run the Workflow ([readme](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/BRDF-Topo-HyTools/Topo%20and%20Brdf%20Corr/README.md))
*   Step 1: Convert NEON Data to ENVI
Ensure the output folder exists before running the conversion script.
Example:
```python neon2envi2.py neon.h5 output/ -anc```
*   Step 2: Generate Configuration JSON
Modify config_generator.py as needed for the specific corrections.
Run the script to generate the configuration file.
Example:
```python config_generator.py```
*   Step 3: Perform Correction
Use image_correct.py with the generated config file to apply corrections.
Example:
```python image_correct.py output/config_01.json```

### Applications of the Workflow
This workflow is ideal for remote sensing professionals and researchers working with NEON data who require precise spectral matching and corrections for their analysis. The streamlined process from data conversion to correction application ensures accuracy and efficiency in multispectral and ecological studies.


## Resampling
### Overview 
This tool facilitates the resampling of NEON and drone hyperspectral data to align with Landsat sensor specifications. It utilizes a JSON file for defining sensor parameters and a Python script incorporating the HyTools library for the resampling process.
### [Readme](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/Resampling/README_RESAMPLING.md)
### Components
*   [landsat_band_parameters.json](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/Resampling/landsat_band_parameters.json): This JSON file includes band parameters for various Landsat missions. It can be adapted to include specifications for NEON and drone sensors, allowing users to resample their data to match the Landsat spectral response.

*   [resampling_demo.py](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/Resampling/resampling_demo.py): A Python script that executes the resampling of NEON and drone data. Key components include:

    *   resampler_hy_obj Class: Initializes with the sensor type (Landsat, NEON, or drone), JSON file path, and HDR file path. Manages the reading and application of band parameters for resampling.

    *  create_header_info Method: Generates necessary header information from the HDR file for the resampling process.

    *   save_envi_data Method: Saves resampled data in the ENVI format.

    *   load_envi_data Function: Loads hyperspectral data from an ENVI binary file, preparing it for resampling.

### Prerequisites
*   Python 3.x
*   NumPy
*   Spectral Python (SPy)
*   HyTools library

### Installation

Install necessary Python libraries:
```pip install numpy spectral hytools```

### Usage
*   Update the landsat_band_parameters.json with NEON or drone sensor parameters if required.
*   Ensure both the JSON file and resampling_demo.py are in your working directory.
*   Execute the script with appropriate arguments for sensor type, file paths, and output specifications.

#### Example
```python
from resampling_deom import resampler_hy_obj

# Initialize resampler object for Landsat 8 OLI
resampler = resampler_hy_obj(sensor_type='Landsat 8 OLI', json_file='landsat_band_parameters.json')

# Apply resampling (add details based on your data and requirements)
```

### Application
This tool is crucial for researchers and professionals working with NEON and drone hyperspectral data who need to align their datasets with Landsat spectral characteristics. Such resampling is essential for comparative analysis across different sensors, enhancing the validity of environmental and geographical studies.


## NEON data access using API
### Overview
This guide provides a generalized method to access data from the National Ecological Observatory Network (NEON) by altering the site name and product ID. It is based on a Python script that utilizes the NEON API for data retrieval.
### [Code](https://github.com/earthlab/cross-sensor-cal/blob/janushi-main/neon-api.ipynb)
### Step-by-Step Guide
1. Defining NEON API Endpoint
Define the base URL for the NEON API. For example:
```NEON_API_ENDPOINT = "https://data.neonscience.org/api/v0/"```
2. Specifying Site and Product ID
Assign variables for the site name and product ID. These can be changed to access different datasets.
```site_name = "NEON_SITE_NAME"  # Replace with desired site name```
```product_id = "NEON_PRODUCT_ID"  # Replace with specific product ID```
3. Constructing the API Request
Build the request URL using the specified site and product ID, and make the API request:
```request_url = f"{NEON_API_ENDPOINT}data/{product_id}/{site_name}"```
```response = requests.get(request_url)```
4. Parsing the Response
Convert the response to a JSON format and extract relevant data:
```data = response.json()```
4. Handling Data
Depending on your requirements, process or analyze the retrieved data. This might involve data cleaning, analysis, visualization, etc.


### Application
This method is ideal for ecologists, environmental scientists, and data analysts who require access to NEON's vast ecological datasets. By simply changing the site name and product ID, a wide range of ecological data can be accessed and utilized for research and analysis.


## About Cyverse

*   I have downloaded NIWO and RMNP data from year 2020 and stored them inside commuitydata -> earthlab -> macrosystems -> NIWO_and_RMNP
*   I have also cloned gitHub repo into macrosystems environment
*   While following above steps we can run the files required.


