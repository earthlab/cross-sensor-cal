"""ENVI header parsing utilities shared across the pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _coerce_scalar(value: str) -> Any:
    token = value.strip().strip('"').strip("'")
    if not token:
        return ""
    lowered = token.lower()
    if lowered == "nan":
        return float("nan")
    for caster in (int, float):
        try:
            return caster(token)
        except (TypeError, ValueError):
            continue
    return token


def _split_block(value: str) -> list[str]:
    inner = value.strip()
    if inner.startswith("{"):
        inner = inner[1:]
    if inner.endswith("}"):
        inner = inner[:-1]
    inner = inner.replace("\n", " ")
    return [token.strip() for token in inner.split(",") if token.strip()]


def _parse_envi_header(hdr_path: Path) -> dict[str, Any]:
    """Parse an ENVI ``.hdr`` file into a dictionary of metadata."""

    if not hdr_path.exists():
        raise FileNotFoundError(hdr_path)

    raw_entries: dict[str, str] = {}
    collecting_key: str | None = None
    collecting_value: list[str] = []

    with hdr_path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            stripped = raw_line.strip()
            if not stripped or stripped.upper() == "ENVI":
                continue

            if collecting_key is not None:
                collecting_value.append(stripped)
                if "}" in stripped:
                    value = " ".join(collecting_value)
                    raw_entries[collecting_key] = value
                    collecting_key = None
                    collecting_value = []
                continue

            if "=" not in stripped:
                continue

            key_part, value_part = stripped.split("=", 1)
            key = key_part.strip().lower()
            value = value_part.strip()

            if value.startswith("{") and "}" not in value:
                collecting_key = key
                collecting_value = [value]
                continue

            raw_entries[key] = value

    list_float_keys = {"wavelength", "fwhm"}
    list_string_keys = {"map info", "band names"}
    int_scalar_keys = {"samples", "lines", "bands", "data type", "byte order"}

    processed: dict[str, Any] = {}
    for key, raw_value in raw_entries.items():
        if raw_value.startswith("{") and raw_value.endswith("}"):
            tokens = _split_block(raw_value)
            if key in list_float_keys:
                try:
                    processed[key] = [float(token) for token in tokens]
                except ValueError as exc:
                    raise RuntimeError(
                        f"Could not parse numeric list for '{key}' from ENVI header"
                    ) from exc
            elif key in list_string_keys:
                processed[key] = [token.strip('"').strip("'") for token in tokens]
            else:
                processed[key] = [
                    _coerce_scalar(token.strip('"').strip("'")) for token in tokens
                ]
            continue

        cleaned = raw_value.strip().strip('"').strip("'")
        if key in int_scalar_keys:
            try:
                processed[key] = int(cleaned)
            except ValueError as exc:
                raise RuntimeError(f"Header value for '{key}' is not an integer") from exc
            continue

        processed[key] = _coerce_scalar(cleaned)

    return processed


__all__ = ["_parse_envi_header"]
