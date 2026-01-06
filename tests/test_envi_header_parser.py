import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("h5py")

from spectralbridge.pipelines.pipeline import _parse_envi_header, convolve_resample_product


def _write_envi_header(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ENVI",
                "samples = 4",
                "lines = 3",
                "bands = 2",
                "interleave = bsq",
                "data type = 4",
                "byte order = 0",
                "wavelength units = Nanometers",
                "wavelength = {",
                "400.0,",
                "410.0",
                "}",
                "fwhm = {",
                "10.0,",
                "10.0",
                "}",
                "map info = {UTM, 1, 1, 500000, 4420000, 1, -1}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_parse_envi_header_multiline(tmp_path):
    hdr_path = tmp_path / "example.hdr"
    _write_envi_header(hdr_path)

    header = _parse_envi_header(hdr_path)

    assert header["samples"] == 4
    assert header["lines"] == 3
    assert header["bands"] == 2
    assert header["data type"] == 4
    assert header["byte order"] == 0
    assert header["interleave"] == "bsq"

    assert header["wavelength"] == [400.0, 410.0]
    assert header["fwhm"] == [10.0, 10.0]
    assert isinstance(header["map info"], list)
    assert header["map info"][0].upper() == "UTM"


def test_convolve_resample_requires_wavelengths(tmp_path):
    hdr_path = tmp_path / "example.hdr"
    _write_envi_header(hdr_path)

    img_path = hdr_path.with_suffix(".img")
    data = np.zeros((2, 3, 4), dtype=np.float32)
    data.tofile(img_path)

    target_dir = tmp_path / "resampled"
    target_dir.mkdir()

    # Build a simple delta function SRF for two bands.
    srfs = {
        "band_01": np.array([1.0, 0.0], dtype=np.float32),
        "band_02": np.array([0.0, 1.0], dtype=np.float32),
    }

    out_stem = target_dir / "resampled"

    convolve_resample_product(
        corrected_hdr_path=hdr_path,
        sensor_srf=srfs,
        out_stem_resampled=out_stem,
        tile_y=2,
        tile_x=2,
    )

    out_hdr = out_stem.with_suffix(".hdr")
    out_img = out_stem.with_suffix(".img")

    assert out_hdr.exists()
    assert out_img.exists()
