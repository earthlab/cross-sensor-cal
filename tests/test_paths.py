from pathlib import Path

from cross_sensor_cal.paths import scene_prefix_from_dir, normalize_brdf_model_path


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
