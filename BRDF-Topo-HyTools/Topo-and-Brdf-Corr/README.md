## Python Scripts Description

- #### neon2envi2.py
To convert neon data to envi format.

- #### config_generator.py
To generate config files for topo and brdf corr.

- #### image_correct.py
Which will load the config file and perform corrections

- #### correction comparsion notebook
This file load config file and perfrom corrections and visualizes plots for before and after correction.



#### Steps to run the code:

1. Convert neon data to envi.
    - CMD: ``` python neon2envi2.py <path-to-dataset_name> <path-to-output_folder> -anc ```
    - Example: python neon2envi.py neon.h5 output/ -anc
    - Make sure output folder exists from the level at which the conversion code is being called.
    - You can change the name of the folder according to your preference.

2. Generate config json
    - Edit the config_generator.py according what correction is to be performed.
    - This script has all the options required.
    - Run the script as ``` python config_generator.py ```
    - This will create the config_<iteration>.json in the folder specified in the config file.

3. Perform Correction
    - Run the image_correct.py file with the config file as the cmd args
    - CMD: ``` python image_correct.py <path-to-config-file> ```
    - Example: ``` python image_correct.py output/config_01.json ```