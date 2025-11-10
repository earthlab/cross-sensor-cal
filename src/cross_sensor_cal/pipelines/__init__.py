"""Pipeline entry points for Cross-Sensor Calibration."""
from __future__ import annotations

from .download import run_download
from .pipeline import run_pipeline
from . import pipeline as pipeline  # re-export for tests that monkeypatch

__all__ = ["run_pipeline", "run_download", "pipeline"]
