from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from cross_sensor_cal.io.neon_schema import (
    canonical_vectors,
    compact_ancillary,
    iter_reflectance_rows,
    resolve,
)
from cross_sensor_cal.pipelines import pipeline


def _build_legacy_h5(tmp_path: Path) -> Path:
    h5_path = tmp_path / "legacy_neon.h5"
    with h5py.File(h5_path, "w") as h5:
        refl_group = h5.require_group("Reflectance")
        meta_group = refl_group.require_group("Metadata")
        spectral_group = meta_group.require_group("Spectral_Data")

        lines, samples, bands = 1024, 1024, 50
        reflectance = refl_group.create_dataset(
            "Reflectance_Data",
            shape=(lines, samples, bands),
            dtype="float32",
            chunks=(1, samples, bands),
            fillvalue=np.float32(0.0),
        )
        reflectance.attrs["Data_Ignore_Value"] = np.float32(-9999.0)

        wavelengths = np.linspace(400.0, 900.0, bands, dtype=np.float32)
        spectral_group.create_dataset("Wavelength", data=wavelengths)
        spectral_group.create_dataset("FWHM", data=np.full(bands, 5.0, dtype=np.float32))

        meta_group.create_dataset(
            "to-sun_Zenith_Angle",
            data=np.full((lines, samples), 45.0, dtype=np.float32),
        )
        meta_group.create_dataset(
            "to-sensor_Zenith_Angle",
            data=np.full((lines, samples), 10.0, dtype=np.float32),
        )

        coord_group = meta_group.require_group("Coordinate_System")
        coord_group.create_dataset(
            "Map_Info",
            data=np.array(
                [
                    b"UTM",
                    b"1",
                    b"1",
                    b"500000",
                    b"4100000",
                    b"1",
                    b"-1",
                ]
            ),
        )
        coord_group.create_dataset("Coordinate_System_String", data=b"")

    return h5_path


def test_iter_reflectance_rows_chunks(tmp_path: Path) -> None:
    h5_path = _build_legacy_h5(tmp_path)
    with h5py.File(h5_path, "r") as h5:
        nr = resolve(h5)
        assert nr.is_legacy is True

        chunk_sizes = []
        for r0, r1, slab in iter_reflectance_rows(nr.ds_reflectance, row_chunk=128):
            chunk_sizes.append(r1 - r0)
            assert (r1 - r0) <= 128
            assert slab.dtype == np.float32
            assert slab.shape[0] == (r1 - r0)
        assert sum(chunk_sizes) == nr.metadata["lines"]


def test_legacy_streaming_calls_writer_multiple_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h5_path = _build_legacy_h5(tmp_path)
    calls: list[int] = []

    def _fake_writer(out_path: Path, table_iter, *, row_group_size: int):
        for table in table_iter:
            calls.append(table.num_rows)
        return Path(out_path)

    monkeypatch.setattr(pipeline, "_write_parquet_stream", _fake_writer)

    with h5py.File(h5_path, "r") as h5:
        nr = resolve(h5)
        wavelengths_nm, _, sun_raw, sen_raw = canonical_vectors(nr)
        to_sun_zenith, to_sensor_zenith = compact_ancillary(sun_raw, sen_raw)

        pipeline._sensor_to_parquet_legacy_safe(  # type: ignore[attr-defined]
            nr,
            tmp_path / "legacy.parquet",
            effective_parquet_chunk_size=15_000,
            row_chunk=128,
            source_image="legacy_reflectance.img",
            wavelengths_nm=wavelengths_nm,
            to_sun_zenith=to_sun_zenith,
            to_sensor_zenith=to_sensor_zenith,
        )

    assert len(calls) > 1
