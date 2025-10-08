import os
import sys
import argparse
from pathlib import Path

# Silence Ray's shared-memory warning about falling back to /tmp. This reduces
# noisy log output while keeping the default behavior intact.
os.environ.setdefault("RAY_DISABLE_OBJECT_STORE_WARNING", "1")

try:  # pragma: no cover - optional dependency guard
    import ray
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    ray = None  # type: ignore[assignment]
import hytools as ht

# Add the repo root and src to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from neon_to_envi import neon_to_envi_task, export_anc
from neon_file_types import (
    NEONReflectanceENVIFile,
    NEONReflectanceAncillaryENVIFile,
)

def has_directional(filename: str) -> bool:
    """Check if '_directional' is present in the filename."""
    return "_directional" in filename

def extract_tile(filename: str) -> str | None:
    """Extract tile code if present (e.g., L019-1)."""
    match = Path(filename).stem
    parts = match.split("_")
    for part in parts:
        if part.startswith("L") and "-" in part:
            return part
    return None

def normalize_sensor_name(sensor: str) -> str:
    """Replace spaces with underscores in sensor names."""
    return sensor.replace(" ", "_")

def neon_to_envi(images: list[str], output_dir: str, anc: bool = False):
    """
    Convert NEON AOP H5 files to ENVI format and optionally export ancillary data.
    """
    print("🚀 Running updated neon_to_envi()...")

    if ray is None:
        raise ModuleNotFoundError(
            "ray is required for parallel NEON conversion. Install it with `pip install ray`."
        )

    # Shut down Ray if already running
    if ray.is_initialized():
        ray.shutdown()

    # Initialize Ray
    ray.init(num_cpus=min(len(images), ray.available_resources().get("CPU", 4)))
    hytool = ray.remote(ht.HyTools)

    # Create actors pool
    actors = [hytool.remote() for _ in images]

    # Process each file
    for actor, image in zip(actors, images):
        img_path = Path(image)
        domain, site, date, time = parse_filename_metadata(img_path.name)
        tile = extract_tile(img_path.name)
        directional = has_directional(img_path.name)

        # Load into HyTools actor
        ray.get(actor.read_file.remote(str(img_path), 'neon'))

        # Build ENVI output path using corrected class
        envi_file = NEONReflectanceENVIFile.from_components(
            domain, site, date, time, Path(output_dir), tile=tile, directional=directional
        )
        print(f"📂 Writing ENVI file: {envi_file.path}")

        # Convert and save
        ray.get(actor.do.remote(neon_to_envi_task, (str(img_path), str(output_dir))))

        if anc:
            # Build ancillary output path using corrected class
            anc_file = NEONReflectanceAncillaryENVIFile.from_components(
                domain, site, date, time, Path(output_dir), tile=tile, directional=directional
            )
            print(f"📦 Exporting ancillary ENVI file: {anc_file.path}")

            ray.get(actor.do.remote(export_anc, (str(img_path), str(output_dir))))

    print("✅ All files processed.")

def parse_filename_metadata(filename: str) -> tuple[str, str, str, str]:
    """Extract domain, site, date, time from filename."""
    parts = filename.split("_")
    domain = parts[1]
    site = parts[2]
    date = "00000000"
    time = "000000"
    for part in parts:
        if part.isdigit() and len(part) == 8:
            date = part
        elif part.isdigit() and len(part) == 6:
            time = part
    return domain, site, date, time

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert NEON AOP H5 to ENVI format and optionally export ancillary data."
    )
    parser.add_argument('--images', nargs='+', required=True, help="Input image path names")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('-anc', '--anc', action='store_true', help="Flag to output ancillary data", required=False)
    args = parser.parse_args()

    neon_to_envi(args.images, args.output_dir, args.anc)
