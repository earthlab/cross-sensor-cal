from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
import pytest

# Ensure project root is on sys.path for package imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Stabilize threads/native libs to prevent flaky CI
os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
os.environ.setdefault("GDAL_NUM_THREADS", "1")
os.environ.setdefault("CPL_DEBUG", "OFF")
os.environ.setdefault("PROJ_NETWORK", "OFF")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

for name in ("ray", "osgeo", "fiona", "rasterio"):
    logging.getLogger(name).setLevel(logging.ERROR)

MODE = os.getenv("CSCAL_TEST_MODE", "unit").lower()

def require_mode(expected: str):
    return pytest.mark.skipif(MODE != expected, reason=f"CSCAL_TEST_MODE!='{expected}'")
