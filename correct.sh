#!/bin/bash

echo "$1"

python neon2envi2.py $1 output -anc

#python config_generator.py

#python image_correct.py output/config_0.json

#gdal_translate -of GTiff output/NEON_D13_NIWO_DP1_20170904_181819_reflectance before_correction.tif

#gdal_translate -of GTiff export/NEON_D13_NIWO_DP1_20170904_181819_reflectance__corrected_0 after_correction.tif