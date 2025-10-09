import importlib
import sys
from types import SimpleNamespace

import pytest


def _reload_gateway():
    return importlib.reload(importlib.import_module("src.third_party.hytools_api"))


def _install_stub(monkeypatch, version: str) -> SimpleNamespace:
    module = SimpleNamespace(__version__=version)
    monkeypatch.setitem(sys.modules, "hytools", module)
    return module


def test_hytools_import_gateway(monkeypatch):
    stub = _install_stub(monkeypatch, "1.6.1")
    gateway = _reload_gateway()

    module, info = gateway.import_hytools()

    assert module is stub
    assert isinstance(info.version, str)
    assert info.version == "1.6.1"


def test_hytools_import_gateway_rejects_out_of_range(monkeypatch):
    _install_stub(monkeypatch, "1.5.0")
    gateway = _reload_gateway()

    with pytest.raises(gateway.HyToolsNotAvailable) as excinfo:
        gateway.import_hytools()

    message = str(excinfo.value)
    assert "unsupported" in message
    assert "1.5.0" in message
