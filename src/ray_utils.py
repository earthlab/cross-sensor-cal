"""Utility helpers for initializing and managing Ray locally.

These wrappers centralise the logic for sizing the local Ray cluster so that
callers do not have to duplicate defensive checks (e.g. clamping the requested
CPU count to what is actually available on the host).  They also opt-out of
Ray's dashboard and the noisy shared-memory warning that frequently appears in
containerised environments with a small ``/dev/shm``.
"""
from __future__ import annotations

import os
import math
from typing import Any

# Ray inspects this environment variable during initialisation; set it eagerly so
# the warning is suppressed even if ``ray`` is imported before :func:`init_ray`.
os.environ.setdefault("RAY_DISABLE_OBJECT_STORE_WARNING", "1")

try:  # pragma: no cover - exercised via integration paths
    import ray
except ModuleNotFoundError:  # pragma: no cover - handled gracefully below
    ray = None  # type: ignore[assignment]


_ENV_CPU_KEYS = (
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "PBS_NP",
    "NSLOTS",
    "OMP_NUM_THREADS",
    "RAY_NUM_CPUS",
)


def _cpu_quota() -> int | None:
    """Return CPUs permitted by cgroup quotas (if configured)."""

    # cgroups v2: ``cpu.max`` has "max" or ``<quota> <period>``
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as fh:
            quota_str, period_str = fh.read().strip().split()
        if quota_str != "max":
            quota = int(quota_str)
            period = int(period_str)
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
    except FileNotFoundError:
        pass
    except Exception:
        return None

    # cgroups v1: ``cpu.cfs_quota_us`` and ``cpu.cfs_period_us``
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r", encoding="utf-8") as quota_fh, open(
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r", encoding="utf-8"
        ) as period_fh:
            quota = int(quota_fh.read().strip())
            period = int(period_fh.read().strip())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except FileNotFoundError:
        pass
    except Exception:
        return None

    return None


def _env_cpu_hint() -> int | None:
    """Return a CPU count hinted via common scheduler environment variables."""

    for key in _ENV_CPU_KEYS:
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            return max(1, int(value))
    return None


def _available_cpus() -> int:
    """Best-effort detection of CPUs available to this process."""

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            cpu_count = len(affinity(0))
            if cpu_count:
                return cpu_count
        except Exception:
            pass

    hint = _env_cpu_hint()
    if hint:
        return hint

    quota = _cpu_quota()
    if quota:
        return quota

    detected = os.cpu_count()
    return detected if detected and detected > 0 else 1


def _resolve_cpu_request(requested: int | None) -> int:
    """Return a safe CPU count for ``ray.init``.

    Parameters
    ----------
    requested:
        The caller's desired CPU count. ``None`` means "use everything Ray can
        detect locally".  Values ``< 1`` are rejected to avoid Ray raising a
        less clear error downstream.
    """

    available = _available_cpus()

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
        os.environ["RAY_DISABLE_OBJECT_STORE_WARNING"] = "1"

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
