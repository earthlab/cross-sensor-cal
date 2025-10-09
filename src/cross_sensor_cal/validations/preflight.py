from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional


class PreflightError(RuntimeError):
    pass


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def require_paths(paths: Mapping[str, Path], label: str) -> None:
    missing = [key for key, value in paths.items() if not _exists(Path(value))]
    if missing:
        raise PreflightError(f"Missing {label}: {', '.join(missing)}")


def validate_inputs(
    input_path: Path,
    ancillary: Mapping[str, Path],
    required_keys: Optional[Iterable[str]] = None,
) -> None:
    if not _exists(input_path):
        raise PreflightError(f"Input not found: {input_path}")

    required_keys = list(required_keys or [])
    needed = {key: Path(ancillary.get(key, "")) for key in required_keys}
    require_paths(needed, "ancillary inputs")
    # TODO: optionally add CRS/shape/nodata checks with rasterio (kept out to avoid heavy deps here)
