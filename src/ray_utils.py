"""Utility helpers for initializing and managing Ray locally.

These wrappers centralise the logic for sizing the local Ray cluster so that
callers do not have to duplicate defensive checks (e.g. clamping the requested
CPU count to what is actually available on the host).  They also opt-out of
Ray's dashboard and the noisy shared-memory warning that frequently appears in
containerised environments with a small ``/dev/shm``.
"""
from __future__ import annotations

import os
from typing import Any

try:  # pragma: no cover - exercised via integration paths
    import ray
except ModuleNotFoundError:  # pragma: no cover - handled gracefully below
    ray = None  # type: ignore[assignment]


def _resolve_cpu_request(requested: int | None) -> int:
    """Return a safe CPU count for ``ray.init``.

    Parameters
    ----------
    requested:
        The caller's desired CPU count. ``None`` means "use everything Ray can
        detect locally".  Values ``< 1`` are rejected to avoid Ray raising a
        less clear error downstream.
    """

    available = os.cpu_count() or 1

    if requested is None:
        return available
    if requested < 1:
        raise ValueError("num_cpus must be at least 1 when provided")

    return min(requested, available)


def init_ray(
    num_cpus: int | None = None,
    *,
    shutdown_existing: bool = True,
    disable_object_store_warning: bool = True,
    **ray_kwargs: Any,
) -> int:
    """Initialise a local Ray instance with sensible defaults.

    The helper ensures we do not over-subscribe CPUs, optionally shuts down any
    previously running instance, and suppresses the common ``/dev/shm`` warning
    that can slow down jobs with excessive logging.

    Parameters
    ----------
    num_cpus:
        Desired CPU count (clamped to the host availability). ``None`` lets Ray
        decide automatically.
    shutdown_existing:
        If ``True`` (default), an already-initialised Ray instance is shut down
        before starting a new one.  This mirrors the common pattern in the
        repository and avoids confusing re-use of stale clusters between runs.
    disable_object_store_warning:
        Suppress the "object store is using /tmp" warning, which is noisy in
        Docker environments. Set to ``False`` if the warning is desired.
    **ray_kwargs:
        Additional keyword arguments forwarded to ``ray.init``.  ``num_cpus`` is
        managed by this helper and therefore not allowed in ``ray_kwargs``.

    Returns
    -------
    int
        The CPU count actually passed to ``ray.init``.
    """

    if ray is None:
        raise ModuleNotFoundError(
            "ray is required for parallel processing. Install it with ``pip install ray``"
        )

    if "num_cpus" in ray_kwargs:
        raise TypeError("Pass num_cpus via the positional argument, not kwargs.")

    if disable_object_store_warning:
        os.environ.setdefault("RAY_DISABLE_OBJECT_STORE_WARNING", "1")

    resolved_cpus = _resolve_cpu_request(num_cpus)

    if shutdown_existing and ray.is_initialized():
        ray.shutdown()

    init_kwargs: dict[str, Any] = {
        "include_dashboard": False,
        "ignore_reinit_error": True,
        **ray_kwargs,
    }

    ray.init(num_cpus=resolved_cpus, **init_kwargs)
    return resolved_cpus


__all__ = ["init_ray"]
