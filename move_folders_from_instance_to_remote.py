#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# -------------------------------
# USER CONFIGURATION
# -------------------------------
"""
Creates NIWO_b85 folder at remote destination and puts all the files from instance. 
"""
SOURCE_FOLDERS = [

    "/home/jovyan/data-store/cross-sensor-cal/NIWO_b85",
    
]

DESTINATION = "i:/iplant/home/shared/earthlab/macrosystems/november_25_processed_flight_lines"


# -------------------------------
# HELPERS
# -------------------------------

def gocmd_exists():
    if not Path("./gocmd").exists():
        raise RuntimeError("❌ ERROR: ./gocmd binary not found here.")


def ensure_remote_folder(path: str):
    subprocess.run(["./gocmd", "mkdir", path], check=False)


def collect_files(local_root: Path):
    """
    Recursively collect ONLY files.
    Exclude everything inside any `.duckdb_tmp` folder.
    """
    file_list = []

    for root, dirs, files in os.walk(local_root):
        # Skip duckdb temp folders entirely
        dirs[:] = [d for d in dirs if d != ".duckdb_tmp"]

        root_path = Path(root)

        for f in files:
            file_list.append(root_path / f)

    return file_list


def upload_file(local_file: Path, remote_file: str):
    """Upload a single file."""
    print(f"📤 Uploading file: {local_file.name}")

    # Ensure parent directory exists remotely
    remote_parent = remote_file.rsplit("/", 1)[0]
    ensure_remote_folder(remote_parent)

    cmd = [
        "./gocmd", "put",
        "--progress", "-f", "-K", "--icat",
        str(local_file),
        remote_parent + "/"
    ]

    subprocess.run(cmd, check=True)


def upload_folder_contents(local_folder: str, remote_root: str):
    local_folder = Path(local_folder).resolve()

    print("\n📂 Uploading folder:")
    print(f"   Local:  {local_folder}")
    print(f"   Remote: {remote_root}\n")

    all_files = collect_files(local_folder)

    print(f"📊 Found {len(all_files)} files to upload (excluding .duckdb_tmp)\n")

    for f in all_files:
        rel_path = f.relative_to(local_folder)  # preserve folder structure
        remote_path = f"{remote_root}/{rel_path.as_posix()}"
        upload_file(f, remote_path)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    gocmd_exists()

    print("\n==============================")
    print("🚚 BEGIN TRANSFER")
    print("==============================\n")

    ensure_remote_folder(DESTINATION)

    for src in SOURCE_FOLDERS:
        folder_name = os.path.basename(src.rstrip("/"))
        remote_target = f"{DESTINATION}/{folder_name}"
        ensure_remote_folder(remote_target)

        upload_folder_contents(src, remote_target)

    print("\n✅ ALL TRANSFERS COMPLETED SUCCESSFULLY!\n")
