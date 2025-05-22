#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

GOCMD = "./gocmd"  # adjust path if needed
PARENT = "/iplant/home/shared/earthlab/macrosystems/cross-sensor-cal/sorted_files/csv/masked_convolution"
MERGED_FOLDER = "merged_per_site"
CHUNK_SIZE = 500_000  # rows per write-chunk

def list_remote(path):
    proc = subprocess.run(
        [GOCMD, "ls", f"i:{path}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        print(f"[ERROR] ls failed for i:{path}:\n{proc.stderr}", file=sys.stderr)
        return []
    entries = []
    for line in proc.stdout.splitlines():
        s = line.strip()
        if not s or s.endswith(":"):
            continue
        for pfx in ("C- ", "d- ", "- "):
            if s.startswith(pfx):
                s = s[len(pfx):]
                break
        entries.append(os.path.basename(s))
    return entries

def discover_complete():
    versions = [v for v in list_remote(PARENT) if v != MERGED_FOLDER]
    pat = re.compile(r"^(.+?)_resample_", re.IGNORECASE)
    seen = defaultdict(dict)
    for v in versions:
        for fn in list_remote(f"{PARENT}/{v}"):
            if not fn.lower().endswith(".csv"):
                continue
            m = pat.match(fn)
            if m:
                seen[m.group(1)][v] = fn
    all_vs = set(versions)
    complete = [o for o, fmap in seen.items() if set(fmap) == all_vs]
    return versions, seen, complete

def ensure_merged_folder():
    subprocess.run(
        [GOCMD, "mkdir", "-p", f"i:{PARENT}/{MERGED_FOLDER}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

def merge_one(origin, versions, seen):
    tmpdir = tempfile.mkdtemp(prefix="merge_")
    try:
        dfs = []
        for v in versions:
            fn     = seen[origin][v]
            remote = f"i:{PARENT}/{v}/{fn}"
            local  = os.path.join(tmpdir, f"{origin}__{v}.csv")
            subprocess.run([GOCMD, "get", remote, local], check=True)

            print(f"[DEBUG] loading {v} for {origin}")
            # split any concatenated records into proper rows
            fixed = local + ".fixed"
            with open(local, 'r', errors="replace") as src, open(fixed, 'w') as dst:
                header = src.readline().rstrip("\n")
                dst.write(header + "\n")
                ncols = header.count(",") + 1
                for line in src:
                    parts = line.rstrip("\n").split(",")
                    idx = 0
                    while idx < len(parts):
                        chunk = parts[idx:idx + ncols]
                        if len(chunk) < ncols:
                            chunk += [""] * (ncols - len(chunk))
                        dst.write(",".join(chunk) + "\n")
                        idx += ncols
            os.replace(fixed, local)

            df = pd.read_csv(local, low_memory=False)

            # keep only the true keys + band columns; drop others
            df = df.drop(columns=["Subdirectory", "Data_Source", "Sensor_Type", "Easting", "Northing"], errors="ignore")
            band_cols = [c for c in df.columns if c.startswith("Band_")]
            df = df[["Pixel_id", "Pixel_Row", "Pixel_Col"] + band_cols]
            df = df.rename(columns={c: f"{v}_{c}" for c in band_cols})

            # cast keys to string for consistency
            for key in ("Pixel_id", "Pixel_Row", "Pixel_Col"):
                df[key] = df[key].astype(str)

            dfs.append(df)

        # merge on the three keys
        keys = ["Pixel_id", "Pixel_Row", "Pixel_Col"]
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on=keys, how="inner")

        # drop rows fully masked across all bands
        all_bands = [c for c in merged.columns if c not in keys]
        if all_bands:
            mask = (merged[all_bands] == -9999).all(axis=1)
            merged = merged.loc[~mask]

        # prepare output path
        out_local = os.path.join(tmpdir, f"{origin}_merged.csv")

        # if empty, write header only
        if merged.empty:
            print(f"[WARN] merged DataFrame for {origin} is empty – writing header only")
            pd.DataFrame(columns=keys + all_bands).to_csv(out_local, index=False)
        else:
            # write in chunks
            total = len(merged)
            with tqdm(total=total, desc=f"Writing {origin}", unit="rows", ncols=80) as pbar:
                for start in range(0, total, CHUNK_SIZE):
                    end   = min(start + CHUNK_SIZE, total)
                    mode  = "w" if start == 0 else "a"
                    hdr   = (start == 0)
                    merged.iloc[start:end].to_csv(
                        out_local, mode=mode, header=hdr, index=False
                    )
                    pbar.update(end - start)

        # upload final merged CSV
        remote_out = f"i:{PARENT}/{MERGED_FOLDER}/{origin}_merged.csv"
        subprocess.run([GOCMD, "put", "--diff", out_local, remote_out], check=True)
        print(f"[OK] Merged {origin} → {remote_out}")

    finally:
        shutil.rmtree(tmpdir)

if __name__ == "__main__":
    print("Discovering complete sites…")
    versions, seen, complete = discover_complete()
    print(f"Found {len(complete)} complete origin(s) across {len(versions)} versions.")
    if not complete:
        sys.exit(1)

    print(f"Ensuring merged folder '{MERGED_FOLDER}' exists…")
    ensure_merged_folder()

    for origin in complete:
        merge_one(origin, versions, seen)

    print("All done.")
