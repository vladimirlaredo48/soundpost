from __future__ import annotations

import numpy as np

from soundpost.acoustics.inference import (
    benjamini_hochberg,
    bootstrap_ci,
    delong_test,
    holm,
    mcnemar_test,
)


def _brute_auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1.0]
    neg = score[y == 0.0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.shape[0] * neg.shape[0]))


def test_delong_auc_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(200) > 0.5).astype(float)
    s1 = rng.random(200) + 0.4 * y
    s2 = rng.random(200) + 0.1 * y
    auc1, auc2, _ = delong_test(y, s1, s2)
    assert abs(auc1 - _brute_auc(y, s1)) < 1e-9
    assert abs(auc2 - _brute_auc(y, s2)) < 1e-9


def test_delong_identical_scores_gives_unit_pvalue() -> None:
    rng = np.random.default_rng(4)
    y = (rng.random(120) > 0.5).astype(float)
    s = rng.random(120)
    _, _, p = delong_test(y, s, s.copy())
    assert p == 1.0


def test_delong_detects_strong_difference() -> None:
    rng = np.random.default_rng(5)
    y = (rng.random(300) > 0.5).astype(float)
    strong = rng.random(300) + 1.5 * y
    weak = rng.random(300)
    _, _, p = delong_test(y, strong, weak)
    assert p < 0.01


def test_mcnemar_symmetry() -> None:
    a = np.array([True, True, False, False, True])
    b = np.array([True, False, True, False, False])
    stat, p = mcnemar_test(a, b)
    assert stat >= 0.0 and 0.0 <= p <= 1.0


def test_multiplicity_corrections_dominate_raw() -> None:
    raw = np.array([0.01, 0.02, 0.04, 0.2])
    for adjusted in (holm(raw), benjamini_hochberg(raw)):
        assert np.all(adjusted >= raw - 1e-12)
        assert np.all(adjusted <= 1.0)


def test_bootstrap_brackets_point_estimate() -> None:
    rng = np.random.default_rng(6)
    values = rng.normal(size=500)
    point, lo, hi = bootstrap_ci(lambda x: float(x.mean()), (values,), 300, seed=1)
    assert lo <= point <= hi
