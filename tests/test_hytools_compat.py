"""Tests for the HyTools compatibility helpers."""

from __future__ import annotations

import importlib
import os
import sys

import pytest
from tests.conftest import require_mode

pytestmark = require_mode("full")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for path in (PROJECT_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

@pytest.fixture()
def hytools_package(tmp_path, monkeypatch):
    """Create a temporary hytools package with nested modules."""

    package_dir = tmp_path / "hytools"
    (package_dir / "custom").mkdir(parents=True)
    (package_dir / "io").mkdir(parents=True)

    # Package markers
    (package_dir / "__init__.py").write_text("\n")
    (package_dir / "custom" / "__init__.py").write_text("\n")
    (package_dir / "io" / "__init__.py").write_text("\n")

    # Nested modules where our compatibility layer needs to discover classes.
    (package_dir / "custom" / "toolbox.py").write_text(
        "class HyTools:\n"
        "    def __init__(self):\n"
        "        self.file_name = ''\n"
        "        self._header = {}\n"
        "\n"
        "    def read_file(self, file_name, file_type):\n"
        "        self.file_name = file_name\n"
        "        self._header = {'file type': file_type}\n"
        "\n"
        "    def get_header(self):\n"
        "        return self._header\n"
        "\n"
        "    def iterate(self, **kwargs):\n"
        "        raise NotImplementedError\n"
    )

    (package_dir / "io" / "writer.py").write_text(
        "class WriteENVI:\n"
        "    def __init__(self, file_path, header):\n"
        "        self.file_path = file_path\n"
        "        self.header = header\n"
        "\n"
        "    def write_chunk(self, *args, **kwargs):\n"
        "        pass\n"
        "\n"
        "    def write_band(self, *args, **kwargs):\n"
        "        pass\n"
        "\n"
        "    def close(self):\n"
        "        pass\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    # Ensure any previously imported hytools modules are cleared so that the
    # compatibility helper has to discover the freshly created package.
    for name in list(sys.modules):
        if name == "hytools" or name.startswith("hytools."):
            sys.modules.pop(name)

    return package_dir

def test_get_hytools_class_discovers_nested_module(hytools_package):
    hytools_compat = importlib.reload(importlib.import_module("hytools_compat"))

    HyTools = hytools_compat.get_hytools_class()

    instance = HyTools()
    instance.read_file("example.h5", "neon")

    assert instance.file_name == "example.h5"
    assert instance.get_header()["file type"] == "neon"

def test_get_write_envi_discovers_nested_module(hytools_package):
    hytools_compat = importlib.reload(importlib.import_module("hytools_compat"))

    WriteENVI = hytools_compat.get_write_envi()

    writer = WriteENVI("output.bsq", {"data type": 4})
    writer.write_chunk([], 0, 0)
    writer.write_band([], 0)
    writer.close()

    assert writer.file_path == "output.bsq"
    assert writer.header == {"data type": 4}

def test_missing_hytools_error_includes_install_hint(monkeypatch):
    hytools_compat = importlib.reload(importlib.import_module("hytools_compat"))

    original_import_module = hytools_compat.import_module

    def _mock_import(name, package=None):
        if name.startswith("hytools"):
            raise ModuleNotFoundError(name)
        return original_import_module(name, package=package)

    monkeypatch.setattr(hytools_compat, "import_module", _mock_import)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        hytools_compat.get_hytools_class()

    message = str(excinfo.value)
    assert "pip install hytools" in message
    assert "conda install -c conda-forge hytools" in message
