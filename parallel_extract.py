#!/usr/bin/env python3
import os
import subprocess
import shutil
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from spectral import open_image
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
FOLDER_REMOTE = "/iplant/home/shared/earthlab/macrosystems/cross-sensor-cal/sorted_files/envi"
FOLDER_OUTPUT = "/iplant/home/shared/earthlab/macrosystems/cross-sensor-cal/sorted_files/csv/masked"
GOCMD         = "./gocmd"

# ─── GOCMD UTILS ────────────────────────────────────────────────────────────────
def run_gocmd(cmd, desc=None):
    if desc: print(f"[{desc}] {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in p.stdout:
        print(line.strip())
    p.wait()
    return p.returncode == 0

def remote_exists(path):
    return subprocess.run([GOCMD, "ls", path],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0

# ─── PIXEL EXTRACTION ────────────────────────────────────────────────────────────
def process_envi_array_simple(array, map_info, folder_name, nodata=-9999):
    bands, rows, cols = array.shape
    flat = array.reshape(bands, -1).T
    mask = ~(np.all(flat == nodata, axis=1))
    data = flat[mask]
    df = pd.DataFrame(data, columns=[f"corrected_Band_{i+1}" for i in range(bands)])
    idx = np.indices((rows, cols)).reshape(2, -1).T[mask]
    df.insert(0, "Pixel_id",    [f"{folder_name}_{i}" for i in range(len(df))])
    df.insert(1, "Pixel_Row",   idx[:,0])
    df.insert(2, "Pixel_Col",   idx[:,1])
    if map_info and len(map_info) >= 7:
        x0, y0, mx, my, xr, yr = map(float, map_info[1:7])
        col_idx, row_idx = idx[:,1], idx[:,0]
        df.insert(3, "Easting",  mx + (col_idx - (x0-1))*xr)
        df.insert(4, "Northing", my - (row_idx - (y0-1))*yr)
    else:
        df.insert(3, "Easting",  np.nan)
        df.insert(4, "Northing", np.nan)
    df.insert(0, "Subdirectory", folder_name)
    df.insert(1, "Data_Source",  "NEON_AOP_ENVI_extraction")
    df.insert(2, "Sensor_Type",  "Hyperspectral")
    return df

def write_csv_in_chunks(df, out_csv, chunk_size=500_000):
    total = len(df)
    with tqdm(total=total, desc="Writing CSV", leave=False) as pbar:
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            df.iloc[start:end].to_csv(
                out_csv,
                mode   = "w" if start == 0 else "a",
                header = (start == 0),
                index  = False
            )
            pbar.update(end - start)

# ─── EXTRACT + UPLOAD ───────────────────────────────────────────────────────────
def extract_and_transfer(envi_path, hdr_path, remote_out, temp_root, log_path):
    base       = os.path.splitext(os.path.basename(envi_path))[0]
    subdir     = os.path.basename(os.path.dirname(envi_path))
    remote_csv = f"{remote_out}/{base}.csv"

    if remote_exists(remote_csv):
        print(f"[SKIP] {remote_csv} already exists.")
        return

    # prepare temp
    shutil.rmtree(temp_root, ignore_errors=True)
    os.makedirs(temp_root, exist_ok=True)

    local_dat = os.path.join(temp_root, base)
    local_hdr = local_dat + ".hdr"
    local_csv = os.path.join(temp_root, base + ".csv")

    # 1) download
    if not run_gocmd([GOCMD, "get", envi_path, local_dat], desc="Downloading .dat"): return
    run_gocmd([GOCMD, "get", hdr_path,  local_hdr], desc="Downloading .hdr")

    # 2) read
    with rasterio.open(local_dat) as src:
        arr = src.read()

    # 3) metadata
    try:
        img      = open_image(local_hdr)
        map_info = img.metadata.get("map info", None)
    except:
        map_info = None

    # 4) extract
    df = process_envi_array_simple(arr, map_info, subdir)

    # 5) write
    write_csv_in_chunks(df, local_csv)

    # 6) upload
    run_gocmd([GOCMD, "put", "--diff", local_csv, remote_csv], desc="Uploading CSV")

    # 7) summary
    try:
        samp = pd.read_csv(local_csv, nrows=100_000)
        lines = [
            f"SUMMARY for {base}",
            f" Rows (approx): {sum(1 for _ in open(local_csv)) - 1:,}",
            f" Easting: {samp.Easting.min():.2f} to {samp.Easting.max():.2f}",
            f" Northing: {samp.Northing.min():.2f} to {samp.Northing.max():.2f}",
        ]
        with open(log_path, "a") as L:
            L.write("\n".join(lines) + "\n" + ("="*40) + "\n")
    except Exception as e:
        print(f"[WARN] summary failed: {e}")

    shutil.rmtree(temp_root, ignore_errors=True)

# ─── PER-DIR WORKER ─────────────────────────────────────────────────────────────
def process_directory(input_dir, output_dir, test_mode=False):
    name      = os.path.basename(input_dir.rstrip("/"))
    temp_root = os.path.abspath(f"local_transfer_{name}")
    log_path  = os.path.abspath(f"summary_{name}.txt")

    # list .hdr remotely
    try:
        ls = subprocess.run([GOCMD, "ls", input_dir], check=True,
                            stdout=subprocess.PIPE, universal_newlines=True)
        hdr_files = [f.strip() for f in ls.stdout.splitlines() if f.strip().endswith(".hdr")]
    except subprocess.CalledProcessError:
        print(f"[ERROR] could not list {input_dir}")
        return

    # TEST MODE: only first file
    if test_mode and hdr_files:
        hdr_files = hdr_files[:1]

    for hdr in hdr_files:
        dat = hdr[:-4]
        extract_and_transfer(
            envi_path   = f"{input_dir}/{dat}",
            hdr_path    = f"{input_dir}/{hdr}",
            remote_out  = output_dir,
            temp_root   = temp_root,
            log_path    = log_path
        )

# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only one file per subdirectory")
    args = parser.parse_args()

    # list subdirs under the base “envi” folder
    try:
        ls   = subprocess.run([GOCMD, "ls", FOLDER_REMOTE], check=True,
                              stdout=subprocess.PIPE, universal_newlines=True)
        subs = [d.strip() for d in ls.stdout.splitlines() if d.strip()]
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] listing base folder: {e.stderr}")
        sys.exit(1)

    with ProcessPoolExecutor() as executor:
        futures = {}
        for sub in subs:
            in_dir  = f"{FOLDER_REMOTE}/{sub}"
            out_dir = f"{FOLDER_OUTPUT}/{sub}"
            futures[executor.submit(process_directory, in_dir, out_dir, args.test)] = sub

        for fut in as_completed(futures):
            sub = futures[fut]
            try:
                fut.result()
                print(f"[DONE] {sub}")
            except Exception as e:
                print(f"[ERROR] {sub} → {e}")
