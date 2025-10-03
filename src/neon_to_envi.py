import os
import argparse
import h5py
import shutil
from h5py import Dataset

# Suppress Ray's /dev/shm fallback warning to keep conversion logs clean.
os.environ.setdefault("RAY_DISABLE_OBJECT_STORE_WARNING", "1")

import ray
import numpy as np
from pathlib import Path
import hytools as ht
from hytools.io.envi import WriteENVI
import re
from functools import partial
from src.file_types import NEONReflectanceFile, NEONReflectanceENVIFile, NEONReflectanceAncillaryENVIFile

# --- Utility functions ---
def get_all_keys(group):
    if isinstance(group, Dataset):
        return [group.name]
    all_keys = []
    for key in group.keys():
        all_keys += get_all_keys(group[key])
    return all_keys

def get_actual_key(h5_file, expected_key):
    actual_keys = {key.lower(): key for key in get_all_keys(h5_file)}
    return actual_keys.get(expected_key.lower())

def get_all_solar_angles(logs_group):
    return np.array([
        (logs_group[log]["Solar_Azimuth_Angle"][()], logs_group[log]["Solar_Zenith_Angle"][()])
        for log in logs_group.keys()
    ])

def ensure_directory_exists(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def envi_header_for_ancillary(hy_obj, attributes):
    """
    Generates a minimal ENVI header dictionary for ancillary data export.
    """
    return {
        "samples": attributes.get("samples", 0),
        "lines": attributes.get("lines", 0),
        "bands": len(attributes.get("band names", [])),
        "interleave": "bsq",
        "data type": attributes.get("data type", 4),
        "file type": "ENVI Standard",
        "byte order": 0,
        "band names": attributes.get("band names", []),
        "map info": hy_obj.get_header().get("map info", []),
        "coordinate system string": hy_obj.get_header().get("coordinate system string", "")
    }

# --- Main export task ---
def neon_to_envi_task(hy_obj, output_dir, metadata=None):
    original_path = Path(hy_obj.file_name)

    # Use metadata if provided, else fallback to parsing from filename
    if metadata:
        print(f"📎 Using injected metadata for: {original_path.name}")
        neon_file = NEONReflectanceFile(
            path=original_path,
            domain=metadata.get("domain", "D00"),
            site=metadata.get("site", "UNK"),
            date=metadata.get("date", "00000000"),
            time=metadata.get("time"),
            tile=metadata.get("tile"),
            suffix=metadata.get("suffix", None)
        )
    else:
        try:
            neon_file = NEONReflectanceFile.from_filename(original_path)
        except ValueError:
            print(f"⚠️ Could not parse {original_path}, using fallback naming.")
            neon_file = NEONReflectanceFile(
                path=original_path, domain="D00", site="UNK",
                date="00000000", time=None, tile=None, suffix="fallback"
            )

    # Create output directory
    specific_output_dir = Path(output_dir) / original_path.stem
    specific_output_dir.mkdir(parents=True, exist_ok=True)

    # Build ENVI output file (suffix removed!)
    envi_file = NEONReflectanceENVIFile.from_components(
        domain=neon_file.domain,
        site=neon_file.site,
        date=neon_file.date,
        time=neon_file.time,
        folder=specific_output_dir,
        tile=neon_file.tile
    )

    if envi_file.path.exists():
        print(f"⚠️ Skipping existing file: {envi_file.file_path}")
        return

    # Process and write
    hy_obj.load_data()
    writer = WriteENVI(envi_file.file_path, hy_obj.get_header())
    iterator = hy_obj.iterate(by='chunk')
    while not iterator.complete:
        chunk = iterator.read_next()
        writer.write_chunk(chunk, iterator.current_line, iterator.current_column)
        if iterator.complete:
            writer.close()

    print(f"✅ Saved: {envi_file.file_path}")


# --- Ancillary export ---
def find_reflectance_metadata_group(h5_file):
    for group in h5_file.keys():
        candidate = f"{group}/Reflectance/Metadata"
        if candidate in h5_file:
            return candidate
    raise ValueError("Could not find Reflectance/Metadata group.")

def export_anc(hy_obj, output_dir):
    neon_file = NEONReflectanceFile.from_filename(Path(hy_obj.file_name))
    with h5py.File(hy_obj.file_name, 'r') as h5_file:
        try:
            base_path = find_reflectance_metadata_group(h5_file) + "/"
        except ValueError as e:
            print(f"❌ {e} in file: {hy_obj.file_name}")
            return

        ancillary_keys = [
            "Ancillary_Imagery/Path_Length",
            "to-sensor_Azimuth_Angle",
            "to-sensor_Zenith_Angle",
            "Logs/Solar_Azimuth_Angle",
            "Logs/Solar_Zenith_Angle",
            "Ancillary_Imagery/Slope",
            "Ancillary_Imagery/Aspect"
        ]

        data = [
            h5_file.get(base_path + key)[()]
            if h5_file.get(base_path + key) is not None else np.array([], dtype=np.float32)
            for key in ancillary_keys
        ]

        attributes = {
            'samples': max((d.shape[1] for d in data if d.ndim == 2), default=0),
            'lines': max((d.shape[0] for d in data if d.ndim == 2), default=0),
            'data type': 4,
            'band names': ['Path Length', 'Sensor Azimuth', 'Sensor Zenith',
                           'Solar Azimuth', 'Solar Zenith', 'Slope', 'Aspect']
        }

        header = envi_header_for_ancillary(hy_obj, attributes)
        specific_output_dir = Path(output_dir) / neon_file.path.stem
        specific_output_dir.mkdir(parents=True, exist_ok=True)

        ancillary_file = NEONReflectanceAncillaryENVIFile.from_components(
            domain=neon_file.domain,
            site=neon_file.site,
            date=neon_file.date,
            time=neon_file.time,
            folder=specific_output_dir,
            tile=neon_file.tile
        )

        if ancillary_file.path.exists():
            print(f"⚠️ Skipping existing ancillary file: {ancillary_file.file_path}")
            return

        writer = WriteENVI(ancillary_file.file_path, header)
        for i, array in enumerate(data):
            if array.size > 0:
                writer.write_band(array, i)
        writer.close()
        print(f"📦 Ancillary data saved: {ancillary_file.file_path}")

# --- Main driver ---
def neon_to_envi(images: list[str], output_dir: str, anc: bool = False, metadata_override: dict = None):
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=len(images))

    hytool = ray.remote(ht.HyTools)
    actors = [hytool.remote() for _ in images]

    _ = ray.get([
        actor.read_file.remote(image, 'neon')
        for actor, image in zip(actors, images)
    ])

    _ = ray.get([
        actor.do.remote(
            partial(
                neon_to_envi_task,
                output_dir=str(output_dir),
                metadata=metadata_override.get(Path(image).name) if metadata_override else None
            )
        )
        for actor, image in zip(actors, images)
    ])

    if anc:
        print("\n📦 Exporting ancillary ENVI data...")
        _ = ray.get([
            actor.do.remote(
                partial(export_anc, output_dir=str(output_dir))
            )
            for actor in actors
        ])

    print("✅ All processing complete.")
    ray.shutdown()

# --- CLI runner (optional use) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NEON AOP H5 to ENVI format and optionally export ancillary data.")
    parser.add_argument('--images', nargs='+', required=True, help="Input image path names")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('-anc', action='store_true', help="Flag to output ancillary data")
    args = parser.parse_args()

    neon_to_envi(args.images, args.output_dir, args.anc)

