import json
import warnings
from pathlib import Path

import ray
import numpy as np
from hytools import HyTools
from hytools.io.envi import WriteENVI
from hytools.topo import calc_topo_coeffs
from hytools.brdf import calc_brdf_coeffs
from hytools.glint import set_glint_parameters
from hytools.masks import mask_create
from spectral import open_image
from spectral.io import envi

from src.file_types import (
    NEONReflectanceENVIFile,
    NEONReflectanceConfigFile,
    NEONReflectanceCoefficientsFile,
    NEONReflectanceAncillaryENVIFile,
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceBRDFMaskENVIFile,
    NEONReflectanceBRDFCorrectedENVIHDRFile
)

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')


def topo_and_brdf_correction(config_file: str):
    """Apply TOPO and BRDF corrections using settings in the config JSON file."""
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

    ray.get([actor.create_bad_bands.remote(config_dict['bad_bands']) for actor in actors])

    # Prepare BRDF corrected file
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

        if brdf_corrected_file.path.exists() and coefficients_file.path.exists():
            print(f"✅ Skipping existing corrections for {reflectance_file.file_path}")
            continue

        if correction == 'topo':
            calc_topo_coeffs(actors, config_dict['topo'])
        elif correction == 'brdf':
            calc_brdf_coeffs(actors, config_dict)
        elif correction == 'glint':
            set_glint_parameters(actors, config_dict)

    if config_dict['export']['coeffs'] and config_dict["corrections"]:
        print("📦 Exporting correction coefficients...")
        ray.get([actor.do.remote(export_coeffs, config_dict['export']) for actor in actors])

    if config_dict['export']['image']:
        print("📦 Exporting corrected images...")
        ray.get([actor.do.remote(apply_corrections, config_dict) for actor in actors])

    ray.shutdown()


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
    header_dict['data type'] = 4

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

    if not brdf_corrected_file.path.exists():
        print(f"📦 Writing corrected image: {brdf_corrected_file.file_path}")
        writer = WriteENVI(brdf_corrected_file.file_path, header_dict)
        iterator = hy_obj.iterate(by='line', corrections=hy_obj.corrections,
                                  resample=config_dict['resample'])
        while not iterator.complete:
            line = iterator.read_next()
            writer.write_line(line, iterator.current_line)
        writer.close()

    # Export masks
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

            header_dict.update({
                'data type': 1,
                'bands': len(masks),
                'band names': mask_names,
                'samples': hy_obj.columns,
                'lines': hy_obj.lines,
                'wavelength': [],
                'fwhm': [],
                'wavelength units': '',
                'data ignore value': 255
            })

            writer = WriteENVI(brdf_corrected_masked_file.file_path, header_dict)
            for i, mask in enumerate(masks):
                writer.write_band(mask.astype(np.uint8), i)
            writer.close()



def generate_correction_configs_for_directory(reflectance_file: NEONReflectanceENVIFile):
    """
    Generate correction configs with complete ancillary mapping and parameters for HyTools.
    """
    anc_files = NEONReflectanceAncillaryENVIFile.find_in_directory(reflectance_file.directory)
    print(f"🔗 Found {len(anc_files)} ancillary files for {reflectance_file.file_path}")

    if not anc_files:
        print(f"⚠️ No ancillary files found for {reflectance_file.file_path}. Skipping.")
        return

    # Build ancillary mapping
    ancillary_bands = [
        'path_length', 'sensor_az', 'sensor_zn',
        'solar_az', 'solar_zn', 'slope', 'aspect', 'phase', 'cosine_i'
    ]
    ancillary_mapping = {
        band_name: [str(anc_files[0].file_path), band_index]
        for band_index, band_name in enumerate(ancillary_bands)
    }

    # Build config file path
    suffix = "envi"
    config_file = NEONReflectanceConfigFile.from_components(
        domain=reflectance_file.domain,
        site=reflectance_file.site,
        date=reflectance_file.date,
        suffix=suffix,
        folder=Path(reflectance_file.directory),
        time=reflectance_file.time,
        tile=reflectance_file.tile,
        directional=reflectance_file.directional
    )
    print(f"📄 Writing config JSON: {config_file.file_path}")

    # Build config dictionary matching example
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
            "suffix": "_corrected_envi"
        },
        "corrections": ["topo", "brdf"],
        "topo": {
            "type": "scs+c",
            "calc_mask": [
                ["ndi", {"band_1": 850, "band_2": 660, "min": 0.1, "max": 1.0}],
                ["ancillary", {"name": "slope", "min": 0.0873, "max": "+inf"}],
                ["ancillary", {"name": "cosine_i", "min": 0.12, "max": "+inf"}],
                ["cloud", {
                    "method": "zhai_2018",
                    "cloud": True, "shadow": True,
                    "T1": 0.01, "t2": 0.1, "t3": 0.25, "t4": 0.5, "T7": 9, "T8": 9
                }]
            ],
            "apply_mask": [
                ["ndi", {"band_1": 850, "band_2": 660, "min": 0.1, "max": 1.0}],
                ["ancillary", {"name": "slope", "min": 0.0873, "max": "+inf"}],
                ["ancillary", {"name": "cosine_i", "min": 0.12, "max": "+inf"}]
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

    # Write JSON to file
    with open(config_file.file_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✅ Config saved: {config_file.file_path}")


def generate_config_json(parent_directory: str):
    """
    Generate correction JSONs for all reflectance files in the parent directory.
    """
    reflectance_files = NEONReflectanceENVIFile.find_in_directory(Path(parent_directory))
    print(f"📂 Found {len(reflectance_files)} reflectance files.")

    for reflectance_file in reflectance_files:
        print(f"📝 Processing: {reflectance_file.file_path}")
        generate_correction_configs_for_directory(reflectance_file)
    print("🎉 All configuration JSONs generated.")



