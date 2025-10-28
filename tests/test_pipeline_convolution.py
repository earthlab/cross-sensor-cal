from pathlib import Path
from unittest.mock import patch


def test_pipeline_order_and_inputs(tmp_path):
    fake_raw_img = tmp_path / "raw_envi.img"
    fake_raw_hdr = tmp_path / "raw_envi.hdr"
    fake_raw_img.write_text("raw")
    fake_raw_hdr.write_text("raw")

    corrected_img = tmp_path / "NEON_xxx_brdfandtopo_corrected_envi.img"
    corrected_hdr = tmp_path / "NEON_xxx_brdfandtopo_corrected_envi.hdr"
    corrected_img.write_text("corrected")
    corrected_hdr.write_text("corrected")

    corr_json = tmp_path / "NEON_xxx_brdfandtopo_corrected_envi.json"
    corr_json.write_text('{"params":"ok"}')

    with patch("cross_sensor_cal.pipeline.export_envi_from_h5") as mock_export, \
        patch("cross_sensor_cal.pipeline.build_and_write_correction_json") as mock_json, \
        patch("cross_sensor_cal.pipeline.apply_brdf_topo_correction") as mock_corr, \
        patch("cross_sensor_cal.pipeline.convolve_all_sensors") as mock_conv:

        mock_export.return_value = (fake_raw_img, fake_raw_hdr)
        mock_json.return_value = corr_json
        mock_corr.return_value = (corrected_img, corrected_hdr)

        from cross_sensor_cal.pipeline import process_flightline

        process_flightline(
            h5_path=tmp_path / "NEON_fake_directional_reflectance.h5",
            out_dir=tmp_path,
        )

    assert mock_export.call_count == 1
    assert mock_json.call_count == 1
    assert mock_corr.call_count == 1
    assert mock_conv.call_count == 1

    corr_args, corr_kwargs = mock_corr.call_args
    assert (
        corr_json in corr_args
        or corr_kwargs.get("correction_json_path") == corr_json
    ), "Correction must use the pre-written JSON"

    conv_args, conv_kwargs = mock_conv.call_args

    used_img = conv_kwargs.get("corrected_img_path", conv_args[0] if conv_args else None)
    used_hdr = conv_kwargs.get("corrected_hdr_path", conv_args[1] if len(conv_args) > 1 else None)

    assert used_img is not None and "brdfandtopo_corrected_envi" in Path(used_img).name
    assert used_hdr is not None and "brdfandtopo_corrected_envi" in Path(used_hdr).name
    assert not str(used_img).endswith(".h5")
    assert not str(used_hdr).endswith(".h5")
