import json
import warnings
from pathlib import Path
from typing import Iterable, Optional
import glob
import os

import ray
import numpy as np
from hytools import HyTools
from hytools.io.envi import WriteENVI
from hytools.topo import calc_topo_coeffs
from hytools.brdf import calc_brdf_coeffs
from hytools.glint import set_glint_parameters
from hytools.masks import mask_create

from src.file_types import (
    NEONReflectanceENVIFile,
    NEONReflectanceConfigFile,
    NEONReflectanceCoefficientsFile,
    NEONReflectanceAncillaryENVIFile,
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceBRDFMaskENVIFile,
    NEONReflectanceBRDFCorrectedENVIHDRFile,  # imported in case you later need HDR
)

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')


# ──────────────────────────────────────────────────────────────────────────────
# Main correction driver
# ──────────────────────────────────────────────────────────────────────────────

def topo_and_brdf_correction(config_file: str):
    """Apply TOPO and BRDF corrections using settings in the (HyTools-ready) config JSON file."""
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    images = config_dict["input_files"]
    if len(images) != 1:
        raise ValueError("Config file must specify exactly 1 input image.")

    image = images[0]
    reflectance_file = NEONReflectanceENVIFile.from_filename(Path(image))

    if ray.is_initialized():
        ray.shutdown()
    print(f"🚀 Using {config_dict['num_cpus']} CPUs for correction.")
    ray.init(num_cpus=config_dict['num_cpus'])

    HyToolsActor = ray.remote(HyTools)
    actors = [HyToolsActor.remote() for _ in images]

    # read image (and ancillary mapping for ENVI)
    if config_dict['file_type'] == 'envi':
        anc_files = config_dict["anc_files"]
        print(f"📦 Ancillary mapping for HyTools:\n{json.dumps(anc_files, indent=2)}\n")
        ray.get([
            actor.read_file.remote(image, config_dict['file_type'], anc_files[str(image)])
            for actor, image in zip(actors, images)
        ])
    else:
        ray.get([
            actor.read_file.remote(image, config_dict['file_type'])
            for actor, image in zip(actors, images)
        ])

    # bad bands
    ray.get([actor.create_bad_bands.remote(config_dict['bad_bands']) for actor in actors])

    # Prepare output BRDF-corrected target (for existence checks / logs)
    brdf_corrected_file = NEONReflectanceBRDFCorrectedENVIFile.from_components(
        domain=reflectance_file.domain,
        site=reflectance_file.site,
        date=reflectance_file.date,
        time=reflectance_file.time,
        suffix=config_dict['export']["suffix"],  # e.g., "_corrected_envi" -> becomes "envi"
        folder=Path(config_dict['export']['output_dir']),
        tile=reflectance_file.tile,
        directional=reflectance_file.directional
    )

    # compute coefficients
    for correction in config_dict["corrections"]:
        coefficients_file = NEONReflectanceCoefficientsFile.from_components(
            domain=reflectance_file.domain,
            site=reflectance_file.site,
            date=reflectance_file.date,
            time=reflectance_file.time,
            correction=correction,
            suffix=config_dict['export']["suffix"],
            folder=Path(config_dict['export']['output_dir']),
            tile=reflectance_file.tile,
            directional=reflectance_file.directional
        )
        print(f"📝 BRDF File: {brdf_corrected_file.file_path}")
        print(f"📝 Coefficients File: {coefficients_file.file_path}")

        # If both outputs already exist, skip recomputing
        if brdf_corrected_file.path.exists() and coefficients_file.path.exists():
            print(f"✅ Skipping existing corrections for {reflectance_file.file_path}")
            continue

        if correction == 'topo':
            calc_topo_coeffs(actors, config_dict['topo'])
        elif correction == 'brdf':
            calc_brdf_coeffs(actors, config_dict)
        elif correction == 'glint':
            set_glint_parameters(actors, config_dict)

    # export coeffs if requested
    if config_dict['export']['coeffs'] and config_dict["corrections"]:
        print("📦 Exporting correction coefficients...")
        ray.get([actor.do.remote(export_coeffs, config_dict['export']) for actor in actors])

    # export corrected images
    if config_dict['export']['image']:
        print("📦 Exporting corrected images...")
        ray.get([actor.do.remote(apply_corrections, config_dict) for actor in actors])

    ray.shutdown()


# ──────────────────────────────────────────────────────────────────────────────
# Additive brightness/reflectance offset on ENVI rasters
# ──────────────────────────────────────────────────────────────────────────────


def _iter_envi_pairs(input_dir: Path) -> Iterable[tuple[Path, Path]]:
    """
    Yield (img_path, hdr_path) pairs for ENVI rasters under ``input_dir``.
    Looks for ``*.hdr`` with a sibling ``*.img``.
    """

    for hdr in Path(input_dir).rglob("*.hdr"):
        img = hdr.with_suffix(".img")
        if img.exists():
            yield img, hdr


def apply_offset_to_envi(
    input_dir: str | Path,
    offset: float,
    clip_to_01: bool = True,
    only_if_name_contains: Optional[list[str]] = None,
) -> int:
    """Add a constant offset to all pixel values in ENVI rasters under ``input_dir``.

    This applies an additive brightness/reflectance shift. Results can optionally be
    clipped to the ``[0, 1]`` range.

    Parameters
    ----------
    input_dir : str | Path
        Root folder to search for ENVI files (``*.hdr`` with sibling ``*.img``).
    offset : float
        Value to **add** to every pixel (e.g., ``-0.005`` to shift down).
    clip_to_01 : bool, default ``True``
        If ``True``, clip results to ``[0, 1]`` after adding the offset.
    only_if_name_contains : list[str] | None
        If provided, only modify files whose basename contains *any* of these
        substrings (case-insensitive).

    Returns
    -------
    int
        Number of ENVI images modified.
    """

    import rasterio

    input_dir = Path(input_dir)
    changed = 0
    subs = [s.lower() for s in (only_if_name_contains or [])]

    for img_path, _ in _iter_envi_pairs(input_dir):
        name_l = img_path.name.lower()
        if subs and not any(s in name_l for s in subs):
            continue

        with rasterio.open(img_path, "r+") as ds:
            for band in range(1, ds.count + 1):
                for _, window in ds.block_windows(band):
                    arr = ds.read(band, window=window)
                    arr = arr.astype(np.float32, copy=False)
                    arr += offset
                    if clip_to_01:
                        np.clip(arr, 0.0, 1.0, out=arr)
                    ds.write(arr, band, window=window)

        changed += 1

    return changed


# Backwards/forwards-friendly alias
apply_brightness_offset_to_envi = apply_offset_to_envi


# ──────────────────────────────────────────────────────────────────────────────
# Helpers invoked on HyTools actors
# ──────────────────────────────────────────────────────────────────────────────

def export_coeffs(hy_obj, export_dict):
    """Export correction coefficients to JSON."""
    reflectance_file = NEONReflectanceENVIFile.from_filename(Path(hy_obj.file_name))
    for correction in hy_obj.corrections:
        coefficients_file = NEONReflectanceCoefficientsFile.from_components(
            domain=reflectance_file.domain,
            site=reflectance_file.site,
            date=reflectance_file.date,
            time=reflectance_file.time,
            correction=correction,
            suffix=export_dict["suffix"],
            folder=Path(export_dict["output_dir"]),
            tile=reflectance_file.tile,
            directional=reflectance_file.directional
        )

        if coefficients_file.path.exists():
            print(f"⚠️ Skipping existing coefficients: {coefficients_file.file_path}")
            continue

        with open(coefficients_file.file_path, 'w') as f:
            corr_dict = getattr(hy_obj, correction, {})
            json.dump(corr_dict, f)
        print(f"✅ Exported coefficients: {coefficients_file.file_path}")


def apply_corrections(hy_obj, config_dict):
    """Apply corrections and export corrected ENVI imagery and masks."""
    header_dict = hy_obj.get_header()
    header_dict['data ignore value'] = hy_obj.no_data
    header_dict['data type'] = 4  # float32

    reflectance_file = NEONReflectanceENVIFile.from_filename(Path(hy_obj.file_name))
    brdf_corrected_file = NEONReflectanceBRDFCorrectedENVIFile.from_components(
        domain=reflectance_file.domain,
        site=reflectance_file.site,
        date=reflectance_file.date,
        time=reflectance_file.time,
        suffix=config_dict['export']["suffix"],
        folder=Path(config_dict['export']['output_dir']),
        tile=reflectance_file.tile,
        directional=reflectance_file.directional
    )

    # write corrected image
    if not brdf_corrected_file.path.exists():
        print(f"📦 Writing corrected image: {brdf_corrected_file.file_path}")
        writer = WriteENVI(brdf_corrected_file.file_path, header_dict)
        iterator = hy_obj.iterate(by='line', corrections=hy_obj.corrections,
                                  resample=config_dict['resample'])
        while not iterator.complete:
            line = iterator.read_next()
            writer.write_line(line, iterator.current_line)
        writer.close()

    # write masks, if requested
    if config_dict['export']['masks']:
        brdf_corrected_masked_file = NEONReflectanceBRDFMaskENVIFile.from_components(
            domain=reflectance_file.domain,
            site=reflectance_file.site,
            date=reflectance_file.date,
            time=reflectance_file.time,
            suffix=config_dict['export']["suffix"],
            folder=Path(config_dict['export']['output_dir']),
            tile=reflectance_file.tile,
            directional=reflectance_file.directional
        )
        if not brdf_corrected_masked_file.path.exists():
            print(f"📦 Writing correction masks: {brdf_corrected_masked_file.file_path}")
            masks = []
            mask_names = []
            for correction in config_dict["corrections"]:
                for mask_type in config_dict[correction]['apply_mask']:
                    masks.append(mask_create(hy_obj, [mask_type]))
                    mask_names.append(f"{correction}_{mask_type[0]}")

            hdr = dict(header_dict)  # shallow copy
            hdr.update({
                'data type': 1,            # uint8
                'bands': len(masks),
                'band names': mask_names,
                'samples': hy_obj.columns,
                'lines': hy_obj.lines,
                'wavelength': [],
                'fwhm': [],
                'wavelength units': '',
                'data ignore value': 255
            })

            writer = WriteENVI(brdf_corrected_masked_file.file_path, hdr)
            for i, mask in enumerate(masks):
                writer.write_band(mask.astype(np.uint8), i)
            writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Config generation (HyTools-ready schema)
# ──────────────────────────────────────────────────────────────────────────────

def generate_correction_configs_for_directory(reflectance_file: NEONReflectanceENVIFile):
    """
    Build a single HyTools-ready config JSON (next to the reflectance image)
    with complete ancillary mapping and parameters.
    """
    # find ancillary stack next to the reflectance file
    anc_files = NEONReflectanceAncillaryENVIFile.find_in_directory(reflectance_file.directory)
    print(f"🔗 Found {len(anc_files)} ancillary files for {reflectance_file.file_path}")
    if not anc_files:
        print(f"⚠️ No ancillary files found for {reflectance_file.file_path}. Skipping.")
        return
    anc_file = anc_files[0]

    # expected ancillary band names (indices are zero-based)
    ancillary_bands = [
        'path_length', 'sensor_az', 'sensor_zn',
        'solar_az', 'solar_zn', 'slope', 'aspect'
    ]
    ancillary_mapping = {band: [str(anc_file.file_path), i] for i, band in enumerate(ancillary_bands)}

    # compose config dict
    config_dict = {
        "bad_bands": [
            [300, 400],
            [1337, 1430],
            [1800, 1960],
            [2450, 2600]
        ],
        "file_type": "envi",
        "input_files": [str(reflectance_file.file_path)],
        "anc_files": {
            str(reflectance_file.file_path): ancillary_mapping
        },
        "export": {
            "coeffs": True,
            "image": True,
            "masks": True,
            "subset_waves": [],
            "output_dir": str(reflectance_file.directory),
            # NOTE: We pass "_corrected_envi", which your BRDF classes normalize to "envi"
            "suffix": "_corrected_envi"
        },
        "corrections": ["topo", "brdf"],
        "topo": {
            "type": "scs+c",
            "calc_mask": [
                ["ndi", {"band_1": 850, "band_2": 660, "min": 0.1, "max": 1.0}],
                ["ancillary", {"name": "slope",    "min": 0.0873, "max": "+inf"}],
                ["ancillary", {"name": "cosine_i", "min": 0.12,   "max": "+inf"}],
                ["cloud", {"method": "zhai_2018", "cloud": True, "shadow": True,
                           "T1": 0.01, "t2": 0.1, "t3": 0.25, "t4": 0.5, "T7": 9, "T8": 9}]
            ],
            "apply_mask": [
                ["ndi",       {"band_1": 850, "band_2": 660, "min": 0.1, "max": 1.0}],
                ["ancillary", {"name": "slope",    "min": 0.0873, "max": "+inf"}],
                ["ancillary", {"name": "cosine_i", "min": 0.12,   "max": "+inf"}]
            ],
            "c_fit_type": "nnls"
        },
        "brdf": {
            "solar_zn_type": "scene",
            "type": "flex",
            "grouped": True,
            "sample_perc": 0.1,
            "geometric": "li_dense_r",
            "volume": "ross_thick",
            "b/r": 10,
            "h/b": 2,
            "interp_kind": "linear",
            "calc_mask": [
                ["ndi", {"band_1": 850, "band_2": 660, "min": 0.1, "max": 1.0}]
            ],
            "apply_mask": [
                ["ndi", {"band_1": 850, "band_2": 660, "min": 0.1, "max": 1.0}]
            ],
            "diagnostic_plots": True,
            "diagnostic_waves": [448, 849, 1660, 2201],
            "bin_type": "dynamic",
            "num_bins": 25,
            "ndvi_bin_min": 0.05,
            "ndvi_bin_max": 1.0,
            "ndvi_perc_min": 10,
            "ndvi_perc_max": 95
        },
        "num_cpus": 4,
        "resample": False,
        "resampler": {
            "type": "cubic",
            "out_waves": [450, 550, 650],
            "out_fwhm": []
        }
    }

    # write config next to the reflectance
    cfg = NEONReflectanceConfigFile.from_components(
        domain=reflectance_file.domain,
        site=reflectance_file.site,
        product=reflectance_file.product,  # e.g., "DP1.30006.001" or "DP1"
        tile=reflectance_file.tile,
        date=reflectance_file.date,
        time=reflectance_file.time,
        directional=reflectance_file.directional,
        folder=Path(reflectance_file.directory),
    )
    with open(cfg.file_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✅ Config saved: {cfg.file_path}")


def generate_config_json(directory: str, num_cpus: int = 8):
    """
    Generate HyTools-ready config JSONs for all ENVI reflectance files under `directory`.
    """
    print("📝 Generating configuration JSON (HyTools-ready)...")

    # find base reflectance ENVI images
    refl_paths = glob.glob(os.path.join(directory, "**", "*_reflectance_envi.img"), recursive=True)
    # exclude corrected and ancillary
    refl_paths = [p for p in refl_paths if "ancillary" not in p.lower() and "corrected" not in p.lower()]

    if not refl_paths:
        print("📂 Found 0 reflectance files.")
        return

    print(f"📂 Found {len(refl_paths)} reflectance files.")

    for img_path in refl_paths:
        img = Path(img_path)
        try:
            reflectance_file = NEONReflectanceENVIFile.from_filename(img)
        except ValueError as e:
            print(f"⚠️ Skipping unparseable reflectance: {img.name} ({e})")
            continue

        # find ancillary stack in same dir
        ancandidates = NEONReflectanceAncillaryENVIFile.find_in_directory(reflectance_file.directory)
        if not ancandidates:
            print(f"⚠️ No ancillary files found for {reflectance_file.file_path}; skipping.")
            continue
        ancillary = ancandidates[0]

        # ancillary mapping (zero-based indices)
        ancillary_band_names = [
            'path_length', 'sensor_az', 'sensor_zn',
            'solar_az', 'solar_zn', 'slope', 'aspect'
        ]
        ancillary_mapping = {name: [str(ancillary.file_path), i] for i, name in enumerate(ancillary_band_names)}

        # full config
        export_suffix = "_corrected_envi"
        config_dict = {
            "bad_bands": [
                [300, 400],
                [1337, 1430],
                [1800, 1960],
                [2450, 2600]
            ],
            "file_type": "envi",
            "input_files": [str(reflectance_file.file_path)],
            "anc_files": {
                str(reflectance_file.file_path): ancillary_mapping
            },
            "export": {
                "coeffs": True,
                "image": True,
                "masks": True,
                "subset_waves": [],
                "output_dir": str(reflectance_file.directory),
                "suffix": export_suffix
            },
            "corrections": ["topo", "brdf"],
            "topo": {
                "type": "scs+c",
                "calc_mask": [
                    ["ndi", {"band_1": 850, "band_2": 660, "min": 0.1, "max": 1.0}],
                    ["ancillary", {"name": "slope",    "min": 0.0873, "max": "+inf"}],
                    ["ancillary", {"name": "cosine_i", "min": 0.12,   "max": "+inf"}],
                    ["cloud", {"method": "zhai_2018", "cloud": True, "shadow": True,
                               "T1": 0.01, "t2": 0.1, "t3": 0.25, "t4": 0.5, "T7": 9, "T8": 9}]
                ],
                "apply_mask": [
                    ["ndi",       {"band_1": 850, "band_2": 660, "min": 0.1, "max": 1.0}],
                    ["ancillary", {"name": "slope",    "min": 0.0873, "max": "+inf"}],
                    ["ancillary", {"name": "cosine_i", "min": 0.12,   "max": "+inf"}]
                ],
                "c_fit_type": "nnls"
            },
            "brdf": {
                "solar_zn_type": "scene",
                "type": "flex",
                "grouped": True,
                "sample_perc": 0.1,
                "geometric": "li_dense_r",
                "volume": "ross_thick",
                "b/r": 10,
                "h/b": 2,
                "interp_kind": "linear",
                "calc_mask": [
                    ["ndi", {"band_1": 850, "band_2": 660, "min": 0.1, "max": 1.0}]
                ],
                "apply_mask": [
                    ["ndi", {"band_1": 850, "band_2": 660, "min": 0.1, "max": 1.0}]
                ],
                "diagnostic_plots": True,
                "diagnostic_waves": [448, 849, 1660, 2201],
                "bin_type": "dynamic",
                "num_bins": 25,
                "ndvi_bin_min": 0.05,
                "ndvi_bin_max": 1.0,
                "ndvi_perc_min": 10,
                "ndvi_perc_max": 95
            },
            "num_cpus": int(num_cpus),
            "resample": False,
            "resampler": {
                "type": "cubic",
                "out_waves": [450, 550, 650],
                "out_fwhm": []
            }
        }

        # write config using the proper file_types constructor (note: no "suffix" arg here)
        cfg = NEONReflectanceConfigFile.from_components(
            domain=reflectance_file.domain,
            site=reflectance_file.site,
            product=reflectance_file.product,  # "DP1.30006.001" or "DP1"
            tile=reflectance_file.tile,
            date=reflectance_file.date,
            time=reflectance_file.time,
            directional=reflectance_file.directional,
            folder=Path(reflectance_file.directory),
        )
        with open(cfg.file_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ Config saved: {cfg.file_path}")
