from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Dict
import json


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
    input_hashes: Dict[str, str]


@dataclass
class QAMetrics:
    provenance: Provenance
    header: HeaderReport
    mask: MaskReport
    correction: Optional[CorrectionReport]
    convolution: List[ConvolutionReport]
    negatives_pct: float
    overbright_pct: float
    issues: List[str]
    brightness_coefficients: Dict[str, Dict[int, float]] = field(default_factory=dict)
    brightness_summary: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def write_json(metrics: QAMetrics, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(metrics.to_json())
    return out_path


__all__ = [
    "HeaderReport",
    "MaskReport",
    "CorrectionReport",
    "ConvolutionReport",
    "Provenance",
    "QAMetrics",
    "write_json",
]
