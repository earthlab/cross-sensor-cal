import os.path

import requests
import os
import subprocess

import requests


def list_neon_products():
    resp = requests.get('https://data.neonscience.org/api/v0/products')
    products = resp.json()['data']
    for product in products:  # just first 10 for demo
        print(product['productCode'], '-', product['productName'])


def download_neon_file(site_code, product_code, year_month, flight_line, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    server = 'http://data.neonscience.org/api/v0/'
    data_url = f'{server}data/{product_code}/{site_code}/{year_month}'

    # Make the API request
    response = requests.get(data_url)
    if response.status_code == 200:
        data_json = response.json()

        # Initialize a flag to check if the file was found
        file_found = False

        # Iterate through files in the JSON response to find the specific flight line
        for file_info in data_json['data']['files']:
            file_name = file_info['name']
            if flight_line in file_name:
                out_path = os.path.join(out_dir, file_name)
                if os.path.exists(out_path):
                    print(f'Skipping {out_path}, already exists')
                    file_found = True
                    continue
                print(f"Downloading {file_name} from {file_info['url']} to {out_path}")

                # Use subprocess.run to handle output
                try:
                    result = subprocess.run(
                        ['wget', '--no-check-certificate', file_info["url"], '-O', out_path],
                        stdout=subprocess.PIPE,  # Capture standard output
                        stderr=subprocess.PIPE,  # Capture standard error
                        text=True  # Decode to text
                    )

                    # Check for errors
                    if result.returncode != 0:
                        print(f"Error downloading file: {result.stderr}")
                    else:
                        print(f"Download completed for {out_path}")
                except Exception as e:
                    print(f"An error occurred: {e}")

                file_found = True
                break

        if not file_found:
            print(f"Flight line {flight_line} not found in the data for {year_month}.")
    else:
        print(
            f"Failed to retrieve data for {year_month}. Status code: {response.status_code}, Response: {response.text}")

def download_neon_flight_lines(site_code: str, year_month: str, flight_lines: str, out_dir: str, product_code: str = 'DP1.30006.001'):
    """
    Downloads NEON flight line files given a site code, product code, year, month, and flight line(s).

    Args:
    - site_code (str): The site code.
    - product_code (str): The product code.
    - year_month (str): The year and month of interest in 'YYYY-MM' format.
    - flight_lines (str or list): A single flight line identifier or a list of flight line identifiers.
    """

    # Check if flight_lines is a single string (flight line), if so, convert it to a list
    if isinstance(flight_lines, str):
        flight_lines = [flight_lines]

    # Iterate through each flight line and download the corresponding file
    for flight_line in flight_lines:
        print(f"Downloading NEON flight line: {flight_line}")
        download_neon_file(site_code, product_code, year_month, flight_line, out_dir)
