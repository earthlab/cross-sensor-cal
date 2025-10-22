
import os
import sys
from pathlib import Path

import pytest
from tests.conftest import MODE, require_mode

pytestmark = require_mode("full")

if MODE != "full":
    pytest.skip("CSCAL_TEST_MODE!='full'", allow_module_level=True)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.file_types import NEONReflectanceENVIFile, SpectralDataParquetFile
from src.polygon_extraction import _process_single_raster, process_raster_in_chunks

class DummyDataFile:
    def __init__(self, path: Path):
        self.path = path

def test_process_single_raster_skips_existing_output(tmp_path, monkeypatch):
    raster_folder = tmp_path / "raster" / "nested"
    raster_folder.mkdir(parents=True)
    raster_file = NEONReflectanceENVIFile.from_components(
        domain="D01",
        site="ABCD",
        tile="L001-1",
        date="20220101",
        time="120000",
        folder=raster_folder,
    )
    spectral_file = SpectralDataParquetFile.from_raster_file(raster_file)
    spectral_file.path.write_text("existing")

    def fail(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("Extraction should have been skipped")

    monkeypatch.setattr("src.polygon_extraction.process_raster_in_chunks", fail)

    _process_single_raster(raster_file, polygon_path=None)

def test_process_raster_in_chunks_skips_when_output_exists(tmp_path, monkeypatch):
    raster_path = tmp_path / "dummy.img"
    output_path = tmp_path / "out.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("existing")

    def fail_open(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("Raster should not be opened when skipping")

    monkeypatch.setattr("src.polygon_extraction.rasterio.open", fail_open)

    process_raster_in_chunks(
        DummyDataFile(raster_path),
        polygon_path=None,
        output_parquet_file=DummyDataFile(output_path),
    )

def test_process_raster_in_chunks_overwrite_removes_existing_output(tmp_path, monkeypatch):
    raster_path = tmp_path / "dummy.img"
    output_path = tmp_path / "out.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("existing")

    class ExplodingDataset:
        def __enter__(self):
            raise RuntimeError("stop")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "src.polygon_extraction.rasterio.open",
        lambda *args, **kwargs: ExplodingDataset(),
    )

    with pytest.raises(RuntimeError):
        process_raster_in_chunks(
            DummyDataFile(raster_path),
            polygon_path=None,
            output_parquet_file=DummyDataFile(output_path),
            overwrite=True,
        )

    assert not output_path.exists()
