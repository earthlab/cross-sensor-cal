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
PARENT = "/iplant/home/shared/earthlab/macrosystems/cross-sensor-cal/sorted_files/csv/unmasked"
MERGED_FOLDER = "merged_per_site"
CHUNK_SIZE = 500_000  # rows per write-chunk

def list_remote(path):
    """List entries in an iRODS collection, stripping prefixes and headers."""
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
    """Return (versions, seen, complete_origins), excluding merged_per_site."""
    versions = [v for v in list_remote(PARENT) if v != MERGED_FOLDER]
    pat = re.compile(r"^(.+?)_resample_", re.IGNORECASE)
    seen = defaultdict(dict)  # origin_id -> {version: filename}

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
    """Create merged_per_site collection if it doesn't exist."""
    subprocess.run(
        [GOCMD, "mkdir", "-p", f"i:{PARENT}/{MERGED_FOLDER}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

def merge_one(origin, versions, seen):
    tmpdir = tempfile.mkdtemp(prefix="merge_")
    try:
        dfs = []
        # 1) download and prepare each version's CSV
        for v in versions:
            fn = seen[origin][v]
            remote = f"i:{PARENT}/{v}/{fn}"
            local = os.path.join(tmpdir, f"{v}.csv")
            subprocess.run([GOCMD, "get", remote, local], check=True)

            df = pd.read_csv(local)
            # drop metadata and Pixel_id
            df = df.drop(columns=["Subdirectory","Data_Source","Sensor_Type","Pixel_id"],
                         errors="ignore")
            # prefix band columns
            band_cols = [c for c in df.columns if c.startswith("corrected_Band_")]
            df = df.rename(columns={c: f"{v}_{c}" for c in band_cols})
            dfs.append(df)

        # 2) merge all on the true keys
        keys = ["Pixel_Row","Pixel_Col","Easting","Northing"]
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on=keys, how="inner")

        # 3) write merged CSV in chunks with a progress bar
        out_local = os.path.join(tmpdir, f"{origin}_merged.csv")
        total = len(merged)
        with tqdm(total=total, desc=f"Writing {origin}", unit="rows", ncols=80) as pbar:
            for start in range(0, total, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, total)
                mode = "w" if start == 0 else "a"
                header = (start == 0)
                merged.iloc[start:end].to_csv(
                    out_local, mode=mode, header=header, index=False
                )
                pbar.update(end - start)

        # 4) upload merged CSV back to iRODS
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
