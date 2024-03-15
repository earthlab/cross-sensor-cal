#!/bin/bash



echo "$1"

filename="${1%.h5}"

#python neon2envi2.py $1 output -anc

#python config_generator.py

#python image_correct.py output/config_0.json

#output_after="export/${filename}__after_correction.tif"

#gdal_translate -of GTiff export/ENVI__corrected_0 $output_after

#output_before="export/${filename}__before_correction.tif"

#gdal_translate -of GTiff output/ENVI $output_before


#python Resampling/resampling_demo.py --json_file, --hdr_path, --sensor_type, --resampling_file_path, --output_path

python Resampling/resampling_demo.py --resampling_file_path export/ENVI__corrected_0 --json_file Resampling/landsat_band_parameters.json --hdr_path export/ENVI__corrected_0.hdr  --sensor_type='Landsat 5 TM' --output_path export/resample_landsat5.hdr

python Resampling/resampling_demo.py --resampling_file_path export/ENVI__corrected_0 --json_file Resampling/landsat_band_parameters.json --hdr_path export/ENVI__corrected_0.hdr  --sensor_type='Landsat 7 ETM+' --output_path export/resample_landsat7.hdr

python Resampling/resampling_demo.py --resampling_file_path export/ENVI__corrected_0 --json_file Resampling/landsat_band_parameters.json --hdr_path export/ENVI__corrected_0.hdr  --sensor_type='Landsat 8 OLI' --output_path export/resample_landsat8.hdr

python Resampling/resampling_demo.py --resampling_file_path export/ENVI__corrected_0 --json_file Resampling/landsat_band_parameters.json --hdr_path export/ENVI__corrected_0.hdr  --sensor_type='Landsat 9 OLI-2' --output_path export/resample_landsat9.hdr


#output_before="export/resample__after_correction.tif"

#gdal_translate -of GTiff output/ENVI $output_before