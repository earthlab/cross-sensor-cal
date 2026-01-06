"""Memory management helpers for pipeline stages."""

import gc
import logging

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def clean_memory(label: str = "") -> None:
    """
    Force garbage collection and log memory usage.
    Called between major pipeline stages to keep RAM low.
    """
    try:
        import ray  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - ray optional in tests
        ray = None

    gc.collect()

    # Try freeing Ray object store if active
    try:
        if ray is not None:  # type: ignore[truthy-function]
            ray._private.internal_api.free_objects([], local_only=True)
    except Exception:  # pragma: no cover - best effort cleanup
        pass

    # Log memory status
    try:
        if psutil is not None:
            mem = psutil.virtual_memory()
            logger.info(
                "🧹 Memory cleanup after %s — %.1f%% used (%.1f GB free)",
                label or "step",
                mem.percent,
                mem.available / 1e9,
            )
            return
    except Exception:  # pragma: no cover - psutil may not be available
        logger.info("🧹 Memory cleanup after %s complete", label or "step")
        return

    logger.info("🧹 Memory cleanup after %s complete", label or "step")
