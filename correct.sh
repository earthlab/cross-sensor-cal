#!/bin/bash


echo "$1"

filename="${1%.h5}"

python neon2envi2.py $1 output -anc

python config_generator.py

python image_correct.py output/config_0.json

output_after="${filename}__after_correction.tif"

gdal_translate -of GTiff export/ENVI__corrected_0 $output_after

output_before="${filename}__before_correction.tif"

gdal_translate -of GTiff output/ENVI $output_before