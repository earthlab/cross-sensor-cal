from pathlib import Path

from cross_sensor_cal.paths import (
    FlightlinePaths,
    scene_prefix_from_dir,
    normalize_brdf_model_path,
)


def test_normalize_brdf_model_path(tmp_path: Path):
    fl = tmp_path / "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance"
    fl.mkdir()
    # legacy file
    (fl / "NIWO_brdf_model.json").write_text("{}", encoding="utf-8")
    out = normalize_brdf_model_path(fl)
    assert out is not None
    assert out.name == "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance_brdf_model.json"
    assert out.exists()
    assert scene_prefix_from_dir(fl) == "NEON_D13_NIWO_DP1_L019-1_20230815_directional_reflectance"


def test_flightline_paths_properties(tmp_path: Path):
    base = tmp_path
    flight_id = "NEON_D13_TEST_DP1_L001-1_20230101_directional_reflectance"
    paths = FlightlinePaths(base, flight_id)

    assert paths.flight_dir == base / flight_id
    assert paths.h5 == base / f"{flight_id}.h5"
    assert paths.envi_img == base / flight_id / f"{flight_id}_envi.img"
    assert paths.corrected_json.name.endswith("_brdfandtopo_corrected_envi.json")

    sensor = paths.sensor_product("landsat_oli")
    assert sensor.img.name.endswith("landsat_oli_envi.img")
    assert sensor.parquet.name.endswith("landsat_oli_envi.parquet")
    assert sensor.qa_png.name.endswith("landsat_oli_envi_qa.png")
