"""Minimal synthetic workflow demonstrating cross-sensor calibration routines."""
from __future__ import annotations

import numpy as np


def demo_calibrate(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match the mean and variance of ``target`` to ``reference`` arrays."""
    ref_mean = float(reference.mean())
    ref_std = float(reference.std() + 1e-12)
    tgt_mean = float(target.mean())
    tgt_std = float(target.std() + 1e-12)

    scale = ref_std / tgt_std
    adjusted = (target - tgt_mean) * scale + ref_mean
    return adjusted


def main() -> None:
    rng = np.random.default_rng(seed=0)
    reference = rng.normal(0, 1, size=(64, 64))
    target = rng.normal(1, 2, size=(64, 64))
    calibrated = demo_calibrate(reference, target)

    print("reference mean/std", reference.mean(), reference.std())
    print("target mean/std", target.mean(), target.std())
    print("calibrated mean/std", calibrated.mean(), calibrated.std())


if __name__ == "__main__":
    main()
