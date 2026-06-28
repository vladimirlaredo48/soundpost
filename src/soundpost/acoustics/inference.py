"""Statistical inference helpers.

Ref: Methods (DeLong test for paired AUROCs, McNemar test for paired
reclassification, Holm family-wise correction, Benjamini-Hochberg FDR,
patient-level bootstrap confidence intervals).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = x.shape[0]
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _fast_delong(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    m = n_pos
    n = scores.shape[1] - m
    positive = scores[:, :m]
    negative = scores[:, m:]
    k = scores.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(positive[r])
        ty[r] = _midrank(negative[r])
        tz[r] = _midrank(scores[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> tuple[float, float, float]:
    order = np.argsort(-y_true, kind="mergesort")
    n_pos = int(np.sum(y_true == 1.0))
    stacked = np.vstack((score_a, score_b))[:, order]
    aucs, cov = _fast_delong(stacked, n_pos)
    contrast = np.array([[1.0, -1.0]])
    var = float((contrast @ cov @ contrast.T).item())
    if var <= 0.0:
        p = 1.0 if aucs[0] == aucs[1] else 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2.0 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[float, float]:
    b = float(np.sum(correct_a & ~correct_b))
    c = float(np.sum(~correct_a & correct_b))
    if b + c == 0.0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return float(stat), float(stats.chi2.sf(stat, 1))


def holm(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.shape[0]
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    m = p.shape[0]
    order = np.argsort(p)
    adjusted = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * m / (rank + 1))
        adjusted[idx] = prev
    return adjusted


def bootstrap_ci(
    metric: Callable[..., float],
    arrays: tuple[np.ndarray, ...],
    resamples: int,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = arrays[0].shape[0]
    point = metric(*arrays)
    draws = np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, n, n)
        draws[b] = metric(*(a[idx] for a in arrays))
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return float(point), lo, hi
