from spectral import *
import numpy as np

def find_spatial_extent(map_info):
    ulx = float(map_info[3])  # Upper-left x
    uly = float(map_info[4])  # Upper-left y
    xres = float(map_info[5])  # Pixel width
    yres = float(map_info[6])  # Pixel height
    # Assume you already have img.shape
    rows, cols = 1000, 1000  # Replace with actual shape (Y, X)
    # Compute lower-right coordinates
    lrx = ulx + (cols * xres)
    lry = uly - (rows * yres)
    extent = {
        "xmin": ulx,
        "xmax": lrx,
        "ymin": lry,
        "ymax": uly
    }

    return extent

def download_validation_data(resampled_header_file_path):
    img = open_image(resampled_header_file_path)
    extent = find_spatial_extent(img.metadata['map info'])

    # Landsat 5 TM

