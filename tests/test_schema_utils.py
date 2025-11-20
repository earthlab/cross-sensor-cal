from pathlib import Path

import pandas as pd

from cross_sensor_cal.exports.schema_utils import infer_stage_from_name, sort_and_rename_spectral_columns


def test_infer_stage_from_name_marks_undarkened_variant() -> None:
    base_name = "scene_landsat_tm_envi.parquet"
    undarkened_name = "scene_landsat_tm_undarkened_envi.parquet"

    base_stage = infer_stage_from_name(base_name)
    assert infer_stage_from_name(undarkened_name) == f"{base_stage}_undarkened"


def test_sort_and_rename_labels_undarkened_columns() -> None:
    df = pd.DataFrame({"wl0485": [0.1], "wl0560": [0.2], "pixel_id": [1]})
    stage_key = infer_stage_from_name(Path("scene_landsat_tm_undarkened_envi.parquet").name)

    renamed = sort_and_rename_spectral_columns(df, stage_key=stage_key, wavelengths_nm=[485, 560])

    spectral = [col for col in renamed.columns if "_wl" in col]
    assert spectral == [f"{stage_key}_b001_wl0485nm", f"{stage_key}_b002_wl0560nm"]
