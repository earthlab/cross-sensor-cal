import importlib
import importlib.metadata as md

from tests.conftest import require_mode

pytestmark = require_mode("full")

def test_hytools_present_and_pinned():
    # Import must succeed via module name "hytools"
    _ = importlib.import_module("hytools")
    # Distribution must be the PyPI package "hy-tools" at our pinned version
    ver = md.version("hy-tools")
    assert ver == "1.6.0", f"Expected hy-tools==1.6.0, got {ver}"
