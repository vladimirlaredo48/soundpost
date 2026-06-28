from __future__ import annotations

import math

import numpy as np

from soundpost.acoustics.discrimination import harrell_cindex, ipcw_auroc


def test_perfect_concordance() -> None:
    risk = np.array([0.9, 0.5, 0.1])
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 1.0, 0.0])
    assert math.isclose(harrell_cindex(risk, time, event), 1.0, rel_tol=1e-9)


def test_partial_concordance() -> None:
    risk = np.array([0.9, 0.1, 0.5])
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 1.0, 0.0])
    assert math.isclose(harrell_cindex(risk, time, event), 2.0 / 3.0, rel_tol=1e-9)


def test_ipcw_auroc_separates_cases_from_controls() -> None:
    risk = np.array([0.9, 0.8, 0.2, 0.1])
    time = np.array([1.0, 2.0, 5.0, 6.0])
    event = np.array([1.0, 1.0, 0.0, 0.0])
    assert math.isclose(ipcw_auroc(risk, time, event, 3.0), 1.0, rel_tol=1e-9)


def test_random_risk_is_near_half() -> None:
    rng = np.random.default_rng(0)
    risk = rng.random(400)
    time = rng.integers(1, 12, 400).astype(float)
    event = (rng.random(400) > 0.5).astype(float)
    assert abs(harrell_cindex(risk, time, event) - 0.5) < 0.08
