"""Dataclasses and helpers for QA metrics serialization."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HeaderReport:
    keys_present: List[str]
    keys_missing: List[str]
    wavelength_unit: Optional[str]
    n_bands: int
    n_wavelengths_finite: int
    first_nm: Optional[float]
    last_nm: Optional[float]
    wavelengths_monotonic: Optional[bool]
    wavelength_source: str  # "header" | "sensor_default" | "absent"


@dataclass
class MaskReport:
    n_total: int
    n_valid: int
    valid_pct: float


@dataclass
class CorrectionReport:
    # robust deltas across wavelengths (med, iqr); indices of largest |Δ|
    delta_median: List[float]
    delta_iqr: List[float]
    largest_delta_indices: List[int]


@dataclass
class ConvolutionReport:
    sensor: str
    rmse: List[float]
    sam: Optional[float]


@dataclass
class Provenance:
    flightline_id: str
    created_utc: str
    package_version: str
    git_sha: str
    input_hashes: Dict[str, str]  # e.g., {"corrected_envi_hdr":"<sha1>", ...}


@dataclass
class QAMetrics:
    provenance: Provenance
    header: HeaderReport
    mask: MaskReport
    correction: Optional[CorrectionReport]
    convolution: List[ConvolutionReport]
    negatives_pct: float
    issues: List[str]  # textual flags, e.g., "non_monotonic_wavelengths"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def write_json(metrics: QAMetrics, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(metrics.to_json())
    return out_path


def main(argv: List[str] | None = None) -> None:
    """Tiny helper CLI for inspecting QA metric JSON payloads."""

    parser = argparse.ArgumentParser(description="Pretty-print QA metrics JSON")
    parser.add_argument("json_path", type=Path, help="Path to a *_qa.json file")
    args = parser.parse_args(argv)

    data = json.loads(args.json_path.read_text())
    print(json.dumps(data, indent=2))


__all__ = [
    "HeaderReport",
    "MaskReport",
    "CorrectionReport",
    "ConvolutionReport",
    "Provenance",
    "QAMetrics",
    "write_json",
    "main",
]

