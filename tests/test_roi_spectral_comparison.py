
from pathlib import Path

import pytest
from tests.conftest import require_mode

pytestmark = require_mode("full")


pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")
Affine = pytest.importorskip("affine", minversion="2.4.0").Affine
box = pytest.importorskip("shapely.geometry", minversion="2.0.0").box

from src.roi_spectral_comparison import extract_roi_spectra

def _write_test_raster(path: Path, data: np.ndarray, *, crs: str = "EPSG:32613") -> None:
    bands, height, width = data.shape
    transform = Affine.from_origin(0, height, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dataset:
        dataset.write(data)

def test_extract_roi_spectra_mean(tmp_path):
    image_dir = Path(tmp_path)

    raster1 = image_dir / "image1.tif"
    raster2 = image_dir / "image2.tif"

    band1 = np.array(
        [
            [1.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    band2 = np.array(
        [
            [10.0, 10.0, 20.0],
            [10.0, 14.0, 20.0],
            [20.0, 20.0, 20.0],
        ],
        dtype=np.float32,
    )
    data1 = np.stack([band1, band2])
    _write_test_raster(raster1, data1)

    data2 = np.stack([band1 + 1.0, band2 + 2.0])
    _write_test_raster(raster2, data2)

    polygon = box(0, 1, 2, 3)
    roi_gdf = gpd.GeoDataFrame(
        {"roi_name": ["test_roi"]},
        geometry=[polygon],
        crs="EPSG:32613",
    )
    roi_path = image_dir / "rois.geojson"
    roi_gdf.to_file(roi_path, driver="GeoJSON")

    result = extract_roi_spectra(
        [raster1, raster2],
        roi_path,
        label_column="roi_name",
        statistics=("mean",),
        invalid_values=(),
    )

    df = result.dataframe
    assert set(df["image"]) == {"image1.tif", "image2.tif"}
    assert set(df["statistic"]) == {"mean"}
    assert (df["roi_label"].unique() == ["test_roi"]).all()

    pivot = df.pivot_table(
        index="image",
        columns="band",
        values="value",
        aggfunc="first",
    )
    expected = pd.DataFrame(
        {
            1: {"image1.tif": 1.0, "image2.tif": 2.0},
            2: {"image1.tif": 11.0, "image2.tif": 13.0},
        }
    )
    pd.testing.assert_frame_equal(pivot, expected)

    # Each band should include four contributing pixels from the ROI
    counts = df.set_index(["image", "band"])["pixel_count"]
    assert (counts == 4).all()
