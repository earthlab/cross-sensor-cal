import numpy as np
import pytest
from pathlib import Path

h5py = pytest.importorskip("h5py")

from cross_sensor_cal.neon_cube import NeonCube


def _create_fake_neon_file(path: Path) -> None:
    wavelengths = np.array([400, 410, 420, 430, 440], dtype=np.float32)
    fwhm = np.array([10, 10, 10, 10, 10], dtype=np.float32)
    map_info = [
        "UTM",
        "1.0",
        "1.0",
        "500000.0",
        "4420000.0",
        "1.0",
        "-1.0",
        "13",
        "North",
        "WGS-84",
    ]

    data = np.zeros((20, 20, 5), dtype=np.float32)
    for y in range(20):
        for x in range(20):
            for b in range(5):
                data[y, x, b] = y * 1000 + x * 10 + b

    with h5py.File(path, "w") as h5_file:
        base_group = h5_file.create_group("TEST_KEY")
        reflectance_group = base_group.create_group("Reflectance")
        reflectance_dataset = reflectance_group.create_dataset(
            "Reflectance_Data", data=data, dtype=np.float32
        )
        reflectance_dataset.attrs["Data_Ignore_Value"] = np.float32(-9999.0)

        metadata_group = reflectance_group.create_group("Metadata")
        spectral_group = metadata_group.create_group("Spectral_Data")
        wavelength_ds = spectral_group.create_dataset("Wavelength", data=wavelengths)
        wavelength_ds.attrs["Units"] = "Nanometers"
        spectral_group.create_dataset("FWHM", data=fwhm)

        coordinate_group = metadata_group.create_group("Coordinate_System")
        coordinate_group.create_dataset(
            "Map_Info", data=np.array(map_info, dtype="S")
        )
        coordinate_group.create_dataset(
            "Coordinate_System_String",
            data=np.array("FAKE PROJECTION WKT", dtype="S"),
        )


def test_neon_cube_iter_chunks(tmp_path):
    fake_h5_path = tmp_path / "fake_neon.h5"
    _create_fake_neon_file(fake_h5_path)

    cube = NeonCube(h5_path=fake_h5_path)

    assert cube.lines == 20
    assert cube.columns == 20
    assert cube.bands == 5

    assert cube.data.shape == (20, 20, 5)
    assert cube.data.dtype == np.float32

    assert isinstance(cube.mask_no_data, np.ndarray)
    assert cube.mask_no_data.shape == (20, 20)

    assert cube.wavelengths.shape == (5,)
    assert cube.fwhm.shape == (5,)

    chunks = list(cube.iter_chunks(chunk_y=10, chunk_x=10))
    assert len(chunks) == 4
    for ys, ye, xs, xe, arr in chunks:
        assert arr.shape == (ye - ys, xe - xs, cube.bands)

    coverage = set()
    for ys, ye, xs, xe, arr in chunks:
        for yy in range(ys, ye):
            for xx in range(xs, xe):
                coverage.add((yy, xx))
    assert len(coverage) == 20 * 20

    header = cube.build_envi_header()

    assert header["samples"] == 20
    assert header["lines"] == 20
    assert header["bands"] == 5

    assert header["interleave"].lower() == "bsq"
    assert header["data type"] == 4
    assert header["byte order"] == 0

    assert "map info" in header
    assert "projection" in header
    assert "wavelength" in header
    assert "fwhm" in header
    assert "wavelength units" in header

    assert isinstance(header["map info"], (list, tuple))
    assert len(header["map info"]) >= 6
    assert isinstance(header["wavelength"], list)
    assert isinstance(header["fwhm"], list)
    assert len(header["wavelength"]) == cube.bands
    assert len(header["fwhm"]) == cube.bands
    assert all(isinstance(v, float) for v in header["wavelength"])
    assert all(isinstance(v, float) for v in header["fwhm"])
    assert header["wavelength units"].lower() == "nanometers"

