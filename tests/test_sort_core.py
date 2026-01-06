from __future__ import annotations
import pytest
from spectralbridge.sort_core import classify_name

@pytest.mark.parametrize("fname,expected", [
    ("", "Generic"),
    ("random.txt", "Generic"),
    ("NEON_D13_NIWO_DP1_L020-1_20230815_reflectance_envi.img", "ENVI"),
    ("NEON_D13_NIWO_DP1_L020-1_20230815_reflectance_envi.hdr", "HDR"),
    ("NEON_D13_NIWO_DP1_L020-1_20230815_resampled_mask.img", "Mask"),
    ("NEON_D13_NIWO_DP1_L020-1_20230815_resampled.img", "Resampled"),
    ("NEON_D13_NIWO_DP1_L020-1_20230815_ancillary_envi.img", "Ancillary"),
    ("NEON_D13_NIWO_DP1_L020-1_20230815_brdf_corrected.img", "Corrected"),
])
def test_classify_name(fname, expected):
    assert classify_name(fname) == expected
