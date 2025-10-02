import importlib


def test_package_importable():
    assert importlib.import_module("cross_sensor_cal")
