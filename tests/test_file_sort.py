from __future__ import annotations

from pathlib import Path

from cross_sensor_cal.file_sort import categorize_file
from cross_sensor_cal.file_types import (
    NEONReflectanceAncillaryENVIFile,
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceBRDFMaskENVIFile,
    NEONReflectanceENVIFile,
    NEONReflectanceENVIHDRFile,
    NEONReflectanceResampledENVIFile,
    NEONReflectanceResampledMaskENVIFile,
)


def _write_nonempty(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"xx")
    return path


def test_categorize_brdf_corrected_returns_corrected(tmp_path: Path) -> None:
    corrected_path = _write_nonempty(
        tmp_path
        / "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_brdfandtopo_corrected_envi.img"
    )
    file_obj = NEONReflectanceBRDFCorrectedENVIFile(
        path=corrected_path,
        domain="D13",
        site="NIWO",
        date="20230815",
    )

    assert categorize_file(file_obj) == "Corrected"


def test_categorize_resampled_sensor_returns_sensor_label(tmp_path: Path) -> None:
    resampled_path = _write_nonempty(
        tmp_path
        / "NEON_D13_NIWO_DP1_L019-1_20230815_directional_resampled_landsat_oli_envi.img"
    )
    file_obj = NEONReflectanceResampledENVIFile(
        path=resampled_path,
        domain="D13",
        site="NIWO",
        date="20230815",
        sensor="landsat_oli",
    )

    assert categorize_file(file_obj) == "landsat oli"


def test_categorize_resampled_mask_flags_masked(tmp_path: Path) -> None:
    masked_path = _write_nonempty(
        tmp_path
        / "NEON_D13_NIWO_DP1_L019-1_20230815_directional_resampled_mask_landsat_oli_envi.img"
    )
    file_obj = NEONReflectanceResampledMaskENVIFile(
        path=masked_path,
        domain="D13",
        site="NIWO",
        date="20230815",
        sensor="landsat_oli",
    )

    assert categorize_file(file_obj) == "landsat oli_Masked"


def test_categorize_reflectance_and_masked(tmp_path: Path) -> None:
    raw_img = _write_nonempty(
        tmp_path
        / "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.img"
    )
    raw_hdr = _write_nonempty(
        tmp_path
        / "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_envi.hdr"
    )

    raw_obj = NEONReflectanceENVIFile(
        path=raw_img,
        domain="D13",
        site="NIWO",
        date="20230815",
    )
    hdr_obj = NEONReflectanceENVIHDRFile(
        path=raw_hdr,
        domain="D13",
        site="NIWO",
        date="20230815",
    )

    assert categorize_file(raw_obj) == "Reflectance"
    assert categorize_file(hdr_obj) == "Reflectance"

    mask_path = _write_nonempty(
        tmp_path
        / "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_brdfandtopo_corrected_mask_envi.img"
    )
    mask_obj = NEONReflectanceBRDFMaskENVIFile(
        path=mask_path,
        domain="D13",
        site="NIWO",
        date="20230815",
    )

    assert categorize_file(mask_obj) == "Reflectance_Masked"


def test_categorize_generic_for_ancillary(tmp_path: Path) -> None:
    ancillary_path = _write_nonempty(
        tmp_path
        / "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_ancillary_envi.img"
    )
    file_obj = NEONReflectanceAncillaryENVIFile(
        path=ancillary_path,
        domain="D13",
        site="NIWO",
        date="20230815",
    )

    assert categorize_file(file_obj) == "Generic"
