from __future__ import annotations
from dataclasses import dataclass
from spectralbridge.types_protocol import categorize_file

@dataclass
class Fake:
    kind: str
    is_masked: bool = False

def test_protocol_basic():
    assert categorize_file(Fake("envi")) == "ENVI"
    assert categorize_file(Fake("envi_hdr")) == "HDR"
    assert categorize_file(Fake("resampled")) == "Resampled"
    assert categorize_file(Fake("resampled_mask")) == "Mask"
    assert categorize_file(Fake("ancillary")) == "Ancillary"
    assert categorize_file(Fake("generic")) == "Generic"
    assert categorize_file(Fake("generic", True)) == "Mask"
