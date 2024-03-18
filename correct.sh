#!/bin/bash

echo "Passing $1 to subprocess."
echo "This takes a few minutes...."

filename="${1%.h5}"

# Initialize a variable for additional flags
#additional_flags=""

# Loop through all arguments starting from the second one
#for arg in "${@:2}"; do
#    case "$arg" in
#        -anc ) additional_flags="$additional_flags -anc" ;;
#        # Add more cases here if you have more flags to handle
#    esac
#done

# Call the Python script with conditional flags
python neon2envi2.py  "$1"  output -anc

#python neon2envi2_generic.py  --images "$1"  --output_dir output -anc

