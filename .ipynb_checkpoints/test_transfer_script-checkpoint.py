import os
import subprocess
import shutil
import numpy as np
import pandas as pd
import rasterio
from spectral import open_image
from tqdm import tqdm

GOCMD_PATH = "./gocmd"

def run_gocmd(command_args, desc="Running gocmd"):
    print(f"[{desc}] {' '.join(command_args)}")
    process = subprocess.Popen(
        command_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    for line in process.stdout:
        print(line.strip())
    process.wait()
    if process.returncode != 0:
        print(f"[ERROR] gocmd failed with code {process.returncode}")
        return False
    return True

def process_envi_array_simple(array, map_info=None, folder_name="Unknown", nodata_val=-9999):
    with tqdm(total=6, desc="Processing ENVI array", leave=False) as pbar:
        bands, rows, cols = array.shape
        reshaped_array = array.reshape(bands, -1).T
        pbar.update(1)

        valid_mask = ~(np.all(reshaped_array == nodata_val, axis=1))
        valid_data = reshaped_array[valid_mask]
        pbar.update(1)

        df = pd.DataFrame(valid_data, columns=[f'corrected_Band_{i+1}' for i in range(bands)])

        pixel_indices = np.indices((rows, cols)).reshape(2, -1).T
        pixel_indices = pixel_indices[valid_mask]
        pixel_ids = [f"{folder_name}_{i}" for i in range(len(df))]
        df.insert(0, 'Pixel_Col', pixel_indices[:, 1])
        df.insert(0, 'Pixel_Row', pixel_indices[:, 0])
        df.insert(0, 'Pixel_id', pixel_ids)
        pbar.update(1)

        if map_info and len(map_info) >= 7:
            x_pixel_start = float(map_info[1])
            y_pixel_start = float(map_info[2])
            map_x = float(map_info[3])
            map_y = float(map_info[4])
            x_res = float(map_info[5])
            y_res = float(map_info[6])

            pixel_row = pixel_indices[:, 0]
            pixel_col = pixel_indices[:, 1]
            Easting = map_x + (pixel_col - (x_pixel_start - 1)) * x_res
            Northing = map_y - (pixel_row - (y_pixel_start - 1)) * y_res

            df.insert(3, 'Easting', Easting)
            df.insert(4, 'Northing', Northing)
        else:
            df.insert(3, 'Easting', np.nan)
            df.insert(4, 'Northing', np.nan)
        pbar.update(2)

        df.insert(0, 'Subdirectory', folder_name)
        df.insert(1, 'Data_Source', 'NEON_AOP_ENVI_extraction')
        df.insert(2, 'Sensor_Type', 'Hyperspectral')
        pbar.update(1)

    return df

def write_csv_in_chunks(df, output_csv, chunk_size=500_000):
    total_rows = len(df)
    with tqdm(total=total_rows, desc="Writing CSV", unit="rows", leave=False) as pbar:
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            mode = 'w' if start == 0 else 'a'
            header = start == 0
            df.iloc[start:end].to_csv(output_csv, mode=mode, header=header, index=False)
            pbar.update(end - start)

def remote_file_exists(remote_path):
    check = subprocess.run([GOCMD_PATH, 'ls', remote_path],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           universal_newlines=True)
    return check.returncode == 0

def extract_and_transfer(remote_path, remote_output_dir):
    base_name = os.path.basename(remote_path)
    folder_name = os.path.basename(os.path.dirname(remote_path))
    remote_csv_path = os.path.join(remote_output_dir, base_name + ".csv")

    if remote_file_exists(remote_csv_path):
        print(f"[SKIP] {remote_csv_path} already exists in iRODS.")
        return

    temp_dir = os.path.abspath("local_test_transfer")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    local_envi_path = os.path.join(temp_dir, base_name)
    local_hdr_path = local_envi_path + ".hdr"
    local_csv_path = os.path.join(temp_dir, base_name + ".csv")

    try:
        with tqdm(total=6, desc=f"Processing {base_name}", unit="step") as stepbar:
            stepbar.set_description("Step 1: Downloading data")
            if not run_gocmd([GOCMD_PATH, 'get', remote_path, local_envi_path], desc="Downloading .dat"):
                return
            run_gocmd([GOCMD_PATH, 'get', remote_path + '.hdr', local_hdr_path], desc="Downloading .hdr")
            stepbar.update(1)

            stepbar.set_description("Step 2: Reading raster")
            with rasterio.open(local_envi_path) as src:
                array = src.read()
            stepbar.update(1)

            stepbar.set_description("Step 3: Reading HDR metadata")
            map_info = None
            if os.path.exists(local_hdr_path):
                try:
                    img = open_image(local_hdr_path)
                    map_info = img.metadata.get("map info", None)
                except Exception as e:
                    print(f"[WARNING] Could not read HDR metadata: {e}")
            stepbar.update(1)

            stepbar.set_description("Step 4: Extracting pixels")
            df = process_envi_array_simple(array, map_info=map_info, folder_name=folder_name)
            stepbar.update(1)

            stepbar.set_description("Step 5: Writing CSV")
            write_csv_in_chunks(df, local_csv_path)
            stepbar.update(1)

            stepbar.set_description("Step 6: Uploading to iRODS")
            if run_gocmd([GOCMD_PATH, 'put', '--diff', local_csv_path, remote_csv_path], desc="Uploading CSV"):
                print("[UPLOAD] File uploaded successfully.")
            else:
                print("[UPLOAD FAIL] File upload failed.")
            stepbar.update(1)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    try:
        sample_df = pd.read_csv(local_csv_path, usecols=lambda c: (
            c in ['Pixel_id', 'Easting', 'Northing'] or c.startswith('corrected_Band_')
        ), nrows=100000)

        summary_lines = []
        summary_lines.append(f"SUMMARY FOR {base_name}")
        summary_lines.append(f"Total Pixels (approx.): {sum(1 for _ in open(local_csv_path)) - 1:,}")
        summary_lines.append(f"Easting range: {sample_df['Easting'].min():.2f} to {sample_df['Easting'].max():.2f}")
        summary_lines.append(f"Northing range: {sample_df['Northing'].min():.2f} to {sample_df['Northing'].max():.2f}")
        summary_lines.append("NaN counts:")
        for col in ['Easting', 'Northing']:
            summary_lines.append(f"  - {col}: {sample_df[col].isna().sum()}")

        band_cols = [col for col in sample_df.columns if col.startswith("corrected_Band_")][:3]
        for col in band_cols:
            summary_lines.append(
                f"{col}: mean={sample_df[col].mean():.4f}, min={sample_df[col].min():.4f}, max={sample_df[col].max():.4f}"
            )

        with open("combined_summary_log_2.txt", "a") as f:
            f.write("\n".join(summary_lines) + "\n" + "="*60 + "\n")
        print(f"[LOGGED] Appended summary to combined_summary_log.txt")

    except Exception as e:
        print(f"[WARNING] Could not generate summary: {e}")

if __name__ == "__main__":
    folder_remote = "/iplant/home/shared/earthlab/macrosystems/cross-sensor-cal/sorted_files/envi/Reflectance__ENVI"
    folder_output = "/iplant/home/shared/earthlab/macrosystems/cross-sensor-cal/sorted_files/csv/masked/Reflectance__ENVI"

    print(f"[INFO] Getting list of files from: {folder_remote}")
    try:
        result = subprocess.run(
            [GOCMD_PATH, 'ls', folder_remote],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        all_files = result.stdout.strip().splitlines()

        file_list = [os.path.splitext(f.strip())[0] for f in all_files
                     if f.strip().endswith(".hdr") and "_masked" not in f]

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to list directory with gocmd:\n{e.stderr}")
        file_list = []

    if not file_list:
        print(f"[INFO] No non-masked .hdr files found in: {folder_remote}")
    else:
        print(f"[INFO] Found {len(file_list)} non-masked files to process.")

        for fname in file_list:
            print(f"\n=== Processing: {fname} ===")
            extract_and_transfer(
                remote_path=os.path.join(folder_remote, fname),
                remote_output_dir=folder_output
            )
