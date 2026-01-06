"""Compatibility layer that re-exports Ray helpers."""
from __future__ import annotations

from ._ray_utils import init_ray, ray_map

__all__ = ["init_ray", "ray_map"]
