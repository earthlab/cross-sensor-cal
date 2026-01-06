import pytest

from spectralbridge.file_types import (
    NEONReflectanceENVIFile,
    NEONReflectanceENVIHDRFile,
    NEONReflectanceFile,
)


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance.h5",
            {
                "domain": "D13",
                "site": "NIWO",
                "product": "DP1",
                "tile": "L019-1",
                "date": "20230815",
                "time": None,
                "descriptor": "directional_reflectance",
                "directional": True,
            },
        ),
        (
            "NEON_D13_NIWO_DP1.30006.001_20200720_163210_reflectance.h5",
            {
                "domain": "D13",
                "site": "NIWO",
                "product": "DP1.30006.001",
                "tile": "20200720T163210",
                "date": "20200720",
                "time": "163210",
                "descriptor": "reflectance",
                "directional": False,
            },
        ),
        (
            "NEON_D01_HARV_DP3_725000_4700000_reflectance.h5",
            {
                "domain": "D01",
                "site": "HARV",
                "product": "DP3",
                "tile": "725000_4700000",
                "date": None,
                "time": None,
                "descriptor": "reflectance",
                "directional": False,
            },
        ),
        (
            "NEON_D05_LIRO_DP3.30006.002_290000_5097000_bidirectional_reflectance.h5",
            {
                "domain": "D05",
                "site": "LIRO",
                "product": "DP3.30006.002",
                "tile": "290000_5097000",
                "date": None,
                "time": None,
                "descriptor": "bidirectional_reflectance",
                "directional": False,
            },
        ),
    ],
)
def test_reflectance_filename_parsing(filename, expected):
    file_info = NEONReflectanceFile.from_filename(filename)
    assert file_info.domain == expected["domain"]
    assert file_info.site == expected["site"]
    assert file_info.product == expected["product"]
    assert file_info.tile == expected["tile"]
    assert file_info.date == expected["date"]
    assert file_info.time == expected["time"]
    assert file_info.descriptor == expected["descriptor"]
    assert file_info.directional is expected["directional"]


def test_envi_filename_with_synthetic_tile(tmp_path):
    envi_file = NEONReflectanceENVIFile.from_components(
        domain="D13",
        site="NIWO",
        product="DP1.30006.001",
        tile="20200720T163210",
        date="20200720",
        directional=False,
        folder=tmp_path,
    )
    assert envi_file.tile == "20200720T163210"
    assert envi_file.date == "20200720"
    assert not envi_file.directional
    assert envi_file.path.name == (
        "NEON_D13_NIWO_DP1.30006.001_20200720T163210_20200720_reflectance_envi.img"
    )

    parsed = NEONReflectanceENVIFile.from_filename(envi_file.path)
    assert parsed.tile == "20200720T163210"
    assert parsed.date == "20200720"
    assert not parsed.directional


def test_envi_bidirectional_filename_round_trip(tmp_path):
    envi_file = NEONReflectanceENVIFile.from_components(
        domain="D05",
        site="LIRO",
        product="DP3.30006.002",
        tile="290000_5097000",
        descriptor="bidirectional_reflectance",
        directional=False,
        folder=tmp_path,
    )
    assert envi_file.tile == "290000_5097000"
    assert envi_file.descriptor == "bidirectional_reflectance"
    assert not envi_file.directional
    assert (
        envi_file.path.name
        == "NEON_D05_LIRO_DP3.30006.002_290000_5097000_bidirectional_reflectance_envi.img"
    )

    parsed = NEONReflectanceENVIFile.from_filename(envi_file.path)
    assert parsed.tile == "290000_5097000"
    assert parsed.descriptor == "bidirectional_reflectance"
    assert not parsed.directional


def test_envi_hdr_with_synthetic_tile(tmp_path):
    hdr_path = tmp_path / (
        "NEON_D13_NIWO_DP1.30006.001_20200720T163210_20200720_reflectance_envi.hdr"
    )
    hdr_file = NEONReflectanceENVIHDRFile.from_filename(hdr_path)
    assert hdr_file.tile == "20200720T163210"
    assert hdr_file.date == "20200720"
    assert not hdr_file.directional

    bidi_hdr_path = tmp_path / (
        "NEON_D05_LIRO_DP3.30006.002_290000_5097000_20230815_bidirectional_reflectance_envi.hdr"
    )
    bidi_hdr = NEONReflectanceENVIHDRFile.from_filename(bidi_hdr_path)
    assert bidi_hdr.tile == "290000_5097000"
    assert bidi_hdr.date == "20230815"
    assert not bidi_hdr.directional
