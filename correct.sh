#!/bin/bash

echo "Passing $1 to subprocess."
echo "This takes a few minutes...."

python neon2envi2.py  "$1"  output -anc

#python neon2envi2_generic.py  --images "$1"  --output_dir output -anc


