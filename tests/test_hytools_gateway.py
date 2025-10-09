from __future__ import annotations

import importlib
import sys

import pytest

from cross_sensor_cal.third_party.hytools_api import HyToolsNotAvailable, import_hytools


def test_hytools_import_gateway(monkeypatch: pytest.MonkeyPatch):
    dummy_hytools = importlib.import_module("types")
    monkeypatch.setitem(sys.modules, "hytools", dummy_hytools)

    ht, info = import_hytools()
    assert ht is dummy_hytools
    assert isinstance(info.version, str)


def test_hytools_import_gateway_failure(monkeypatch: pytest.MonkeyPatch):
    for name in list(sys.modules):
        if name == "hytools" or name.startswith("hytools."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    with pytest.raises(HyToolsNotAvailable) as excinfo:
        import_hytools()

    assert "HyTools failed to import" in str(excinfo.value)
