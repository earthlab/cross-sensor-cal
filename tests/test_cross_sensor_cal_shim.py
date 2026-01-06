import importlib
import warnings


def test_cross_sensor_cal_imports():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        compat = importlib.import_module("cross_sensor_cal")

    assert compat.__path__, "compatibility shim should expose the implementation path"
    assert importlib.import_module("cross_sensor_cal.pipelines.pipeline")
    assert importlib.import_module("cross_sensor_cal.brdf_topo")
