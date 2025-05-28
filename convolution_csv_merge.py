#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from itertools import combinations

import pandas as pd
from pandas.errors import ParserError
from tqdm import tqdm

# Configuration
GOCMD         = "./gocmd"
PARENT        = "/iplant/home/shared/earthlab/macrosystems/cross-sensor-cal/sorted_files/csv/masked_convolution"
MERGED_FOLDER = "merged_per_site"
INSPECT_DIR   = "./inspect"
KEYS_OUT_DIR  = "./shared_keys"
CHUNK_SIZE    = 500_000
KEYS          = ["Pixel_id", "Pixel_Row", "Pixel_Col"]

def list_remote(path):
    proc = subprocess.run([GOCMD, "ls", f"i:{path}"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode:
        print(f"[ERROR] ls failed for i:{path}:\n{proc.stderr}", file=sys.stderr)
        return []
    entries = []
    for line in proc.stdout.splitlines():
        s = line.strip()
        if not s or s.endswith(":"): continue
        for pfx in ("C- ", "d- ", "- "):
            if s.startswith(pfx):
                s = s[len(pfx):]
                break
        entries.append(os.path.basename(s))
    return entries

def discover_complete():
    versions = [v for v in list_remote(PARENT) if v != MERGED_FOLDER]
    pat      = re.compile(r"^(.+?)_resample_", re.IGNORECASE)
    seen     = defaultdict(dict)
    for v in versions:
        for fn in list_remote(f"{PARENT}/{v}"):
            if not fn.lower().endswith(".csv"): continue
            m = pat.match(fn)
            if m:
                seen[m.group(1)][v] = fn
    all_vs   = set(versions)
    complete = [o for o, fmap in seen.items() if set(fmap) == all_vs]
    return versions, seen, complete

def ensure_merged_folder():
    subprocess.run([GOCMD, "mkdir", "-p", f"i:{PARENT}/{MERGED_FOLDER}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def standardize_keys(df):
    for key in KEYS:
        df[key] = pd.to_numeric(df[key], errors='coerce')
    df = df.dropna(subset=KEYS)
    for key in KEYS:
        df[key] = df[key].astype(int).astype(str)
    return df

def merge_one(origin, versions, seen):
    tmpdir = tempfile.mkdtemp(prefix="merge_")
    os.makedirs(INSPECT_DIR, exist_ok=True)
    os.makedirs(KEYS_OUT_DIR, exist_ok=True)
    try:
        dfs = []
        for v in versions:
            fn     = seen[origin][v]
            remote = f"i:{PARENT}/{v}/{fn}"
            local  = os.path.join(tmpdir, f"{origin}__{v}.csv")
            subprocess.run([GOCMD, "get", remote, local], check=True)

            print(f"[DEBUG] reading {v} for {origin}")
            with open(local, 'r', errors='replace') as fh:
                first = fh.readline().strip()
            vals = first.split(',')
            has_header = all(k in vals for k in KEYS)

            try:
                if has_header:
                    df = pd.read_csv(local, engine="python", sep=",", on_bad_lines="skip")
                else:
                    ncols = len(vals)
                    nbands = ncols - 8
                    names = [
                        "Subdirectory", "Data_Source", "Sensor_Type",
                        "Pixel_id", "Pixel_Row", "Pixel_Col", "Easting", "Northing"
                    ] + [f"Band_{i}" for i in range(1, nbands+1)]
                    df = pd.read_csv(local, header=None, names=names,
                                     engine="python", sep=",", on_bad_lines="skip")
            except ParserError as e:
                print(f"[ERROR] ParserError in {local}: {e}", file=sys.stderr)
                raise

            df.columns = [c.strip() for c in df.columns]
            missing = [k for k in KEYS if k not in df.columns]
            if missing:
                print(f"[ERROR] {v} for {origin} missing {missing}", file=sys.stderr)
                print(f"→ columns: {df.columns.tolist()}", file=sys.stderr)
                sys.exit(1)

            df = df.drop(columns=["Subdirectory","Data_Source","Sensor_Type","Easting","Northing"],
                         errors="ignore")

            band_cols = [c for c in df.columns if c.startswith("Band_")]
            df = df[KEYS + band_cols]
            df = df.rename(columns={c: f"{v}_{c}" for c in band_cols})
            df = standardize_keys(df)
            dfs.append(df)

        print(f"[DEBUG] Row counts before merging:")
        for i, df in enumerate(dfs):
            print(f"  {versions[i]}: {len(df)} rows")

        # Shared keys
        key_sets = [set(tuple(row) for row in df[KEYS].values) for df in dfs]
        shared_keys = set.intersection(*key_sets)
        print(f"[DEBUG] Shared key count: {len(shared_keys)}")

        if not shared_keys:
            print(f"[WARNING] No shared keys across all versions for {origin}. Skipping.")
            return

        shared_df = pd.DataFrame(list(shared_keys), columns=KEYS)
        shared_df.to_csv(os.path.join(KEYS_OUT_DIR, f"{origin}_keys.csv"), index=False)

        for i in range(len(dfs)):
            dfs[i] = dfs[i][dfs[i][KEYS].apply(tuple, axis=1).isin(shared_keys)]

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on=KEYS, how="inner")

        print(f"[DEBUG] Merged result has {len(merged)} rows")
        if merged.empty:
            print(f"[WARNING] Merged DataFrame for {origin} is empty after final merge. Skipping.")
            return

        # Save inspection copy
        inspect_path = os.path.join(INSPECT_DIR, f"{origin}_merged.csv")
        merged.to_csv(inspect_path, index=False)
        print(f"[DEBUG] Wrote merged CSV to {inspect_path}")

        # Save final chunked
        out_local = os.path.join(tmpdir, f"{origin}_merged.csv")
        with tqdm(total=len(merged), desc=f"Writing {origin}", unit="rows", ncols=80) as pbar:
            for start in range(0, len(merged), CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, len(merged))
                mode = "w" if start == 0 else "a"
                header = (start == 0)
                merged.iloc[start:end].to_csv(out_local, mode=mode, header=header, index=False)
                pbar.update(end - start)

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
