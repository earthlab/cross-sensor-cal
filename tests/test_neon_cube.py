from pathlib import Path

import numpy as np
import pytest

from cross_sensor_cal.io.neon import read_neon_cube
from cross_sensor_cal.neon_cube import NeonCube

h5py = pytest.importorskip("h5py")


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


def _create_fake_legacy_neon_file(path: Path) -> None:
    wavelengths = np.array([500, 600, 700], dtype=np.float32)
    fwhm = np.array([5, 5, 5], dtype=np.float32)
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

    data = np.zeros((10, 10, 3), dtype=np.float32)
    for y in range(10):
        for x in range(10):
            for b in range(3):
                data[y, x, b] = y * 100 + x * 10 + b

    with h5py.File(path, "w") as h5_file:
        reflectance_group = h5_file.create_group("Reflectance")
        reflectance_dataset = reflectance_group.create_dataset(
            "Reflectance",
            data=data,
            dtype=np.float32,
        )
        reflectance_dataset.attrs["NoData"] = np.float32(-9999.0)

        metadata_group = reflectance_group.create_group("Metadata")
        spectral_group = metadata_group.create_group("Spectral")
        wavelength_ds = spectral_group.create_dataset("Wavelengths", data=wavelengths)
        wavelength_ds.attrs["Unit"] = "Nanometers"
        spectral_group.create_dataset("FWHM", data=fwhm)

        coordinate_group = metadata_group.create_group("Coordinates")
        coordinate_group.create_dataset(
            "Map_Info", data=np.array(map_info, dtype="S")
        )
        coordinate_group.create_dataset(
            "Projection", data=np.array("LEGACY PROJECTION", dtype="S")
        )


def _create_fake_site_group_legacy_file(path: Path) -> None:
    wavelengths = np.linspace(400, 420, 5, dtype=np.float32)
    fwhm = np.full(5, 10, dtype=np.float32)
    data = np.arange(20 * 10 * 5, dtype=np.float32).reshape(20, 10, 5)

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

    with h5py.File(path, "w") as h5_file:
        site_group = h5_file.create_group("NIWO")
        reflectance_group = site_group.create_group("Reflectance")
        reflectance_ds = reflectance_group.create_dataset(
            "Reflectance_Data", data=data, dtype=np.float32
        )
        reflectance_ds.attrs["Data_Ignore_Value"] = np.float32(-9999.0)

        metadata_group = reflectance_group.create_group("Metadata")
        spectral_group = metadata_group.create_group("Spectral_Data")
        wavelength_ds = spectral_group.create_dataset("Wavelength", data=wavelengths)
        wavelength_ds.attrs["Units"] = "Nanometers"
        spectral_group.create_dataset("FWHM", data=fwhm)

        coordinate_group = metadata_group.create_group("Coordinate_System")
        coordinate_group.create_dataset("Map_Info", data=np.array(map_info, dtype="S"))
        coordinate_group.create_dataset(
            "Coordinate_System_String",
            data=np.array("LEGACY SITE PROJECTION", dtype="S"),
        )


def test_neon_cube_iter_chunks(tmp_path):
    fake_h5_path = tmp_path / "fake_neon.h5"
    _create_fake_neon_file(fake_h5_path)

    cube = NeonCube(h5_path=fake_h5_path)

    assert cube.lines == 20
    assert cube.columns == 20
    assert cube.bands == 5

    full_cube = cube.load_full_cube()
    assert full_cube.shape == (20, 20, 5)
    assert full_cube.dtype == np.float32

    assert cube.mask_no_data is None

    assert cube.wavelengths.shape == (5,)
    assert cube.fwhm.shape == (5,)

    chunks = list(cube.iter_chunks(chunk_y=10, chunk_x=10))
    assert len(chunks) == 4
    for ys, ye, xs, xe, arr in chunks:
        assert arr.shape == (ye - ys, xe - xs, cube.bands)
        np.testing.assert_allclose(arr, full_cube[ys:ye, xs:xe, :])

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


def test_read_neon_cube_new_layout(tmp_path):
    fake_h5_path = tmp_path / "fake_neon.h5"
    _create_fake_neon_file(fake_h5_path)

    cube, wavelengths, meta = read_neon_cube(fake_h5_path)

    assert cube.shape == (20, 20, 5)
    assert wavelengths.shape == (5,)
    assert meta["bands"] == 5
    assert meta["lines"] == 20
    assert meta["wavelength_units"].lower() == "nanometers"
    assert meta["metadata_group_paths"]
    assert meta["layout"] == "reflectance_group"


def test_read_neon_cube_old_layout(tmp_path):
    fake_h5_path = tmp_path / "legacy_neon.h5"
    _create_fake_legacy_neon_file(fake_h5_path)

    cube, wavelengths, meta = read_neon_cube(fake_h5_path)

    assert cube.shape == (10, 10, 3)
    assert wavelengths.shape == (3,)
    assert meta["bands"] == 3
    assert meta["map_info"]
    assert meta["wavelength_units"].lower() == "nanometers"
    assert meta["metadata_group_paths"]
    assert meta["layout"] == "legacy_hdf5"


def test_read_neon_cube_pre_2021_new_layout(tmp_path):
    fake_h5_path = tmp_path / "NEON_D13_SITE_DP1_L001-1_20200720_directional_reflectance.h5"
    _create_fake_neon_file(fake_h5_path)

    cube, wavelengths, meta = read_neon_cube(fake_h5_path)

    assert cube.shape == (20, 20, 5)
    assert wavelengths.shape == (5,)
    assert meta["bands"] == 5
    assert meta["metadata_group_paths"]


def test_read_neon_cube_site_group_legacy_layout(tmp_path):
    fake_h5_path = tmp_path / "NEON_D13_NIWO_DP1_20200720_reflectance.h5"
    _create_fake_site_group_legacy_file(fake_h5_path)

    cube, wavelengths, meta = read_neon_cube(fake_h5_path)

    assert cube.shape == (20, 10, 5)
    assert wavelengths.shape == (5,)
    assert meta["bands"] == 5
    assert meta["layout"] == "legacy_site_group"
    assert meta["site"] == "NIWO"
    assert meta["metadata_group_paths"]
    assert meta["wavelength_units"].lower() == "nanometers"

