from __future__ import annotations

import os
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
    """Initialise Ray with a default of eight CPUs unless overridden."""

    ray = require_ray()
    if ray.is_initialized():
        return ray

    resolved = _resolve_cpu_target(num_cpus)
    init_kwargs = {
        "num_cpus": resolved,
        "ignore_reinit_error": True,
        "include_dashboard": False,
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

    @ray.remote
    def _task(arg: _T) -> _U:
        return func(arg)

    items = list(iterable)
    if not items:
        return []

    futures = [_task.remote(item) for item in items]
    return list(ray.get(futures))
