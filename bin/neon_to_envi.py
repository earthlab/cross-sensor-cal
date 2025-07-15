import sys
from pathlib import Path

# Add the repo root and src to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from neon_to_envi import neon_to_envi_task, export_anc

import argparse
import ray
import hytools as ht



def neon_to_envi(images: list[str] = None, output_dir: str = None, anc: bool = False):
    """
    Main function to convert NEON AOP H5 files to ENVI format and optionally export ancillary data.
    """
    if ray.is_initialized():
        ray.shutdown()

    ray.init(num_cpus=len(images))
    hytool = ray.remote(ht.HyTools)
    actors = [hytool.remote() for _ in images]
    _ = ray.get([actor.read_file.remote(image, 'neon') for actor, image in zip(actors, images)])
    _ = ray.get([actor.do.remote(neon_to_envi_task, output_dir) for actor in actors])
    if anc:
        print("\nExporting ancillary data")
        _ = ray.get([actor.do.remote(export_anc, output_dir) for actor in actors])
    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert NEON AOP H5 to ENVI format and optionally export ancillary data.")
    parser.add_argument('--images', nargs='+', required=True, help="Input image path names")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('-anc','--anc', action='store_true', help="Flag to output ancillary data", required=False)
    args = parser.parse_args()

    neon_to_envi(args.images, args.output_dir, args.anc)
