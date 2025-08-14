import os
import argparse
import h5py
import shutil
from h5py import Dataset
import ray
import numpy as np
from pathlib import Path
import hytools as ht
from hytools.io.envi import WriteENVI
import re
from src.file_types import NEONReflectanceFile, NEONReflectanceENVIFile, NEONReflectanceAncillaryENVIFile


def get_all_keys(group):
    if isinstance(group, Dataset):
        return [group.name]
    all_keys = []
    for key in group.keys():
        all_keys += get_all_keys(group[key])
    return all_keys


def get_actual_key(h5_file, expected_key):
    """Find a key in the HDF5 file matching expected_key, ignoring case."""
    actual_keys = {key.lower(): key for key in get_all_keys(h5_file)}
    return actual_keys.get(expected_key.lower())


def get_all_solar_angles(logs_group):
    return np.array([
        (logs_group[log]["Solar_Azimuth_Angle"][()], logs_group[log]["Solar_Zenith_Angle"][()])
        for log in logs_group.keys()
    ])


def ensure_directory_exists(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def neon_to_envi_task(hy_obj, output_dir, metadata=None):
    original_path = Path(hy_obj.file_name)

    # === Always force metadata if supplied ===
    if metadata:
        print(f"📎 Forcing injected metadata for: {original_path.name}")
        neon_file = NEONReflectanceFile(
            path=original_path,
            domain=metadata.get("domain", "D00"),
            site=metadata.get("site", "UNK"),
            date=metadata.get("date", "00000000"),
            time=metadata.get("time", "000000"),
            tile=metadata.get("tile", "L000-0"),
            suffix=metadata.get("suffix", None)
        )
    else:
        try:
            neon_file = NEONReflectanceFile.from_filename(original_path)
        except ValueError:
            print(f"⚠️ Could not parse {original_path}, using minimal fallback.")
            neon_file = NEONReflectanceFile(
                path=original_path,
                domain="D00",
                site="UNK",
                date="00000000",
                time="000000",
                tile="L000-0",
                suffix="fallback"
            )

    # Output folder setup
    specific_output_dir = Path(output_dir) / original_path.stem
    specific_output_dir.mkdir(parents=True, exist_ok=True)

    # Build ENVI output path with correct time included
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



def envi_header_for_ancillary(hy_obj, attributes, interleave='bil'):
    return {
        "ENVI description": "Ancillary data",
        "samples": attributes['samples'],
        "lines": attributes['lines'],
        "bands": len(attributes['band names']),
        "header offset": 0,
        "file type": "ENVI Standard",
        "data type": attributes['data type'],
        "interleave": interleave,
        "byte order": 0,
        "map info": attributes.get('map info', ""),
        "coordinate system string": attributes.get('coordinate system string', ""),
        "wavelength units": "Unknown",
        "band names": attributes['band names'],
    }


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

def neon_to_envi(images: list[str], output_dir: str, anc: bool = False):
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=len(images))
    hytool = ray.remote(ht.HyTools)
    actors = [hytool.remote() for _ in images]

    _ = ray.get([actor.read_file.remote(image, 'neon') for actor, image in zip(actors, images)])
    _ = ray.get([actor.do.remote(neon_to_envi_task, output_dir) for actor in actors])

    if anc:
        print("\n📦 Exporting ancillary ENVI data...")
        _ = ray.get([actor.do.remote(export_anc, output_dir) for actor in actors])

    print("✅ All processing complete.")
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NEON AOP H5 to ENVI format and optionally export ancillary data.")
    parser.add_argument('--images', nargs='+', required=True, help="Input image path names")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('-anc', action='store_true', help="Flag to output ancillary data")
    args = parser.parse_args()

    neon_to_envi(args.images, args.output_dir, args.anc)

