#!/bin/bash

echo "$1"

python neon2envi2.py $1 output -anc

python config_generator.py

python image_correct.py output/config_0.json

gdal_translate -of GTiff export/ENVI__corrected_0 after_correction.tif

gdal_translate -of GTiff output/ENVI before_correction.tif