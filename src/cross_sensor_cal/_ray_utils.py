from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from ._optional import require_ray

_DEFAULT_CPUS = 8
_ENV_VAR = "CSC_RAY_NUM_CPUS"

_T = TypeVar("_T")
_U = TypeVar("_U")


def _resolve_cpu_target(num_cpus: int | None) -> int:
    if num_cpus is None:
        raw_env = os.environ.get(_ENV_VAR)
        if raw_env:
            try:
                num_cpus = int(raw_env)
            except ValueError:
                num_cpus = None
    target = num_cpus if num_cpus and num_cpus > 0 else _DEFAULT_CPUS
    available = os.cpu_count() or target
    return max(1, min(target, available))


def init_ray(*, num_cpus: int | None = None, **ray_kwargs: Any):
    """Initialize Ray with project defaults and disable memory monitoring."""

    # --- Disable Ray's memory monitor and safety thresholds ---
    # These settings intentionally disable Ray's internal memory protection so that
    # tasks may continue running until the operating system or container enforces
    # its own limits. Use with caution on limited-memory environments.
    os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")
    os.environ.setdefault("RAY_memory_usage_threshold", "1.0")
    os.environ.setdefault("RAY_enable_object_store_memory_monitor", "0")

    RAY_DEBUG = os.environ.get("CSC_RAY_DEBUG", "").lower() in {"1", "true", "yes"}

    ray = require_ray()
    if ray.is_initialized():
        if RAY_DEBUG:
            print("[init_ray] Memory monitor disabled. Ray already initialised.")
        return ray

    resolved = _resolve_cpu_target(num_cpus)
    if RAY_DEBUG:
        print(f"[init_ray] Memory monitor disabled. num_cpus={resolved}")
    source_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[2]
    existing_path = os.environ.get("PYTHONPATH")
    path_parts = [str(source_root), str(project_root)]
    if existing_path:
        path_parts.append(existing_path)
    path_value = os.pathsep.join(path_parts)

    default_runtime_env = {
        "py_modules": [str(source_root)],
        "env_vars": {"PYTHONPATH": path_value},
    }
    user_runtime_env = ray_kwargs.pop("runtime_env", None)
    if user_runtime_env:
        merged_env = {**default_runtime_env, **user_runtime_env}
    else:
        merged_env = default_runtime_env

    init_kwargs = {
        "num_cpus": resolved,
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "runtime_env": merged_env,
    }
    init_kwargs.update(ray_kwargs)
    ray.init(**init_kwargs)
    return ray


def ray_map(
    func: Callable[[
        _T,
    ], _U],
    iterable: Iterable[_T],
    *,
    num_cpus: int | None = None,
) -> list[_U]:
    """Execute ``func`` across ``iterable`` using Ray remote tasks."""

    ray = init_ray(num_cpus=num_cpus)

    # Import Ray exception types lazily so this module can be imported without Ray
    # installed. ``require_ray`` in :func:`init_ray` ensures ``ray`` is available
    # before these imports execute.
    from ray.exceptions import LocalRayletDiedError, OutOfDiskError, RayError

    @ray.remote(max_retries=0)
    def _task(arg: _T) -> _U:
        return func(arg)

    max_in_flight = max(1, _resolve_cpu_target(num_cpus))
    items_iter = iter(iterable)
    in_flight: list["ray.ObjectRef[_U]"] = []
    results: list[_U] = []

    try:
        for _ in range(max_in_flight):
            try:
                next_item = next(items_iter)
            except StopIteration:
                break
            in_flight.append(_task.remote(next_item))

        while in_flight:
            done, in_flight = ray.wait(in_flight, num_returns=1, timeout=None)
            if not done:
                continue

            completed_ref = done[0]
            results.append(ray.get(completed_ref))

            try:
                next_item = next(items_iter)
            except StopIteration:
                continue
            in_flight.append(_task.remote(next_item))

        return results
    except OutOfDiskError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Ray failed during execution because the local disk used for the object "
            "store is full. Free space under the Ray temporary or spilling "
            "directory, or rerun with a smaller 'max_workers' / 'parquet_chunk_size'."
        ) from exc
    except LocalRayletDiedError as exc:  # pragma: no cover - runtime/environmental
        raise RuntimeError(
            "Ray's local raylet process died during execution, often due to memory or "
            "disk pressure. Check Ray logs and consider reducing parallelism via "
            "'max_workers' or decreasing 'parquet_chunk_size'."
        ) from exc
    except RayError as exc:  # pragma: no cover - depends on Ray behaviour
        raise RuntimeError(
            "Ray reported an internal error during task execution. Review the Ray "
            "logs for details and adjust resource usage if necessary."
        ) from exc
