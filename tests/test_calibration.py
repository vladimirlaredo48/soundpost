from __future__ import annotations

import numpy as np

from soundpost.acoustics.calibration import (
    calibration_slope,
    d_calibration_pvalue,
    expected_calibration_error,
)


def _well_calibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    risk = rng.random(n)
    y = (rng.random(n) < risk).astype(float)
    time = np.where(y == 1.0, 1.0, 10.0)
    return risk, time, y


def test_calibration_slope_near_one_when_calibrated() -> None:
    risk, time, event = _well_calibrated(4000, 0)
    slope = calibration_slope(risk, time, event, horizon=5.0)
    assert 0.85 < slope < 1.15


def test_ece_small_when_calibrated() -> None:
    risk, time, event = _well_calibrated(4000, 1)
    ece = expected_calibration_error(risk, time, event, horizon=5.0, bins=10)
    assert ece < 0.05


def test_dcalibration_distinguishes_uniform_from_clustered() -> None:
    uniform_pit = (np.arange(500) + 0.5) / 500.0
    clustered_pit = np.full(500, 0.03)
    event = np.ones(500)
    assert d_calibration_pvalue(uniform_pit, event, bins=10) > 0.05
    assert d_calibration_pvalue(clustered_pit, event, bins=10) < 0.05
