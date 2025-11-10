"""Tests ensuring the pipeline does not require Ray unless explicitly requested."""
from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from cross_sensor_cal.pipelines.pipeline import go_forth_and_multiply


@pytest.fixture
def _block_ray_import(monkeypatch: pytest.MonkeyPatch):
    """Force ``import ray`` to raise ImportError for the duration of a test."""

    real_import = builtins.__import__

    def _fake_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):  # type: ignore[override]
        if name == "ray" or name.startswith("ray."):
            raise ImportError("Ray is not available in this environment")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


@pytest.fixture
def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Stub heavy pipeline stages so tests can run without external data."""

    class _DummyPaths:
        def __init__(self, base_folder: Path, flight_id: str):
            base = Path(base_folder)
            self.flight_dir = base / flight_id
            self.flight_dir.mkdir(parents=True, exist_ok=True)
            self.envi_img = self.flight_dir / f"{flight_id}_envi.img"
            self.envi_hdr = self.flight_dir / f"{flight_id}_envi.hdr"

    monkeypatch.setattr("cross_sensor_cal.pipelines.pipeline.FlightlinePaths", _DummyPaths)
    monkeypatch.setattr("cross_sensor_cal.pipelines.pipeline.stage_download_h5", lambda **_: None)
    monkeypatch.setattr("cross_sensor_cal.pipelines.pipeline.process_one_flightline", lambda **_: None)


def test_thread_engine_operates_without_ray(
    tmp_path: Path,
    _block_ray_import: None,
    _stub_pipeline: None,
) -> None:
    """Thread engine fallback should not require Ray."""

    go_forth_and_multiply(
        base_folder=tmp_path,
        site_code="TEST",
        year_month="2024-01",
        flight_lines=["FLIGHT"],
        engine="thread",
    )


def test_ray_engine_requires_dependency(
    tmp_path: Path,
    _block_ray_import: None,
    _stub_pipeline: None,
) -> None:
    """Ray default should surface a helpful error when Ray is missing."""

    with pytest.raises(RuntimeError, match="Optional dependency 'ray'"):
        go_forth_and_multiply(
            base_folder=tmp_path,
            site_code="TEST",
            year_month="2024-01",
            flight_lines=["FLIGHT"],
        )


def test_ray_engine_uses_ray_map(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ray engine should delegate to the shared ``ray_map`` helper."""

    calls: dict[str, object] = {}

    def _fake_ray_map(func, iterable, *, num_cpus=None):
        calls["num_cpus"] = num_cpus
        calls["count"] = len(list(iterable))
        return []

    monkeypatch.setattr(
        "cross_sensor_cal.pipelines.pipeline.ray_map",
        _fake_ray_map,
    )

    monkeypatch.setattr(
        "cross_sensor_cal.pipelines.pipeline.process_one_flightline",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "cross_sensor_cal.pipelines.pipeline.stage_download_h5",
        lambda **_: None,
    )

    go_forth_and_multiply(
        base_folder=tmp_path,
        site_code="TEST",
        year_month="2024-01",
        flight_lines=["A", "B"],
        engine="ray",
        max_workers=3,
    )

    assert calls["num_cpus"] == 3
    assert calls["count"] == 2
