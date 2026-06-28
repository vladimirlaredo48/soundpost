"""Calibration metrics: slope, integrated calibration index, integrated Brier,
expected calibration error and the D-calibration goodness-of-fit p-value.

Ref: Results (calibration slope 0.94, integrated calibration index 0.042,
D-calibration p = 0.42 for CRESCO versus 0.004 likelihood-only).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from soundpost.acoustics.discrimination import _g_lookup, _km_censoring

_EPS = 1.0e-7
_integrator_name = "trapezoid" if hasattr(np, "trapezoid") else "trapz"
_trapezoid = getattr(np, _integrator_name)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _binary_window(time: np.ndarray, event: np.ndarray, horizon: float) -> np.ndarray:
    keep = (time > horizon) | (event == 1.0)
    label = ((time <= horizon) & (event == 1.0)).astype(float)
    return np.where(keep, label, np.nan)


def calibration_slope(
    risk: np.ndarray, time: np.ndarray, event: np.ndarray, horizon: float
) -> float:
    label = _binary_window(time, event, horizon)
    mask = ~np.isnan(label)
    y = label[mask]
    x = _logit(risk[mask])
    beta = np.array([0.0, 1.0])
    design = np.column_stack([np.ones_like(x), x])
    for _ in range(50):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), _EPS, None)
        grad = design.T @ (y - mu)
        hess = design.T @ (design * w[:, None])
        step = np.linalg.solve(hess + _EPS * np.eye(2), grad)
        beta = beta + step
        if np.linalg.norm(step) < 1.0e-8:
            break
    return float(beta[1])


def integrated_calibration_index(
    risk: np.ndarray, time: np.ndarray, event: np.ndarray, horizon: float, bandwidth: float = 0.1
) -> float:
    label = _binary_window(time, event, horizon)
    mask = ~np.isnan(label)
    y = label[mask]
    p = np.clip(risk[mask], _EPS, 1.0 - _EPS)
    smoothed = np.empty_like(p)
    for i, point in enumerate(p):
        weights = np.exp(-0.5 * ((p - point) / bandwidth) ** 2)
        smoothed[i] = np.sum(weights * y) / np.sum(weights)
    return float(np.mean(np.abs(p - smoothed)))


def expected_calibration_error(
    risk: np.ndarray, time: np.ndarray, event: np.ndarray, horizon: float, bins: int = 10
) -> float:
    label = _binary_window(time, event, horizon)
    mask = ~np.isnan(label)
    y = label[mask]
    p = risk[mask]
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if sel.any():
            total += np.abs(p[sel].mean() - y[sel].mean()) * sel.mean()
    return float(total)


def integrated_brier_score(
    survival: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    grid: np.ndarray,
) -> float:
    uniq, values = _km_censoring(time, event)
    scores = np.empty(grid.shape[0])
    for k, tau in enumerate(grid):
        surv_tau = survival[:, k]
        g_event = _g_lookup(uniq, values, time, left=True)
        g_tau = _g_lookup(uniq, values, np.array([tau]), left=False)[0]
        contrib = np.zeros(time.shape[0])
        early = (time <= tau) & (event == 1.0)
        late = time > tau
        contrib[early] = (surv_tau[early] ** 2) / g_event[early]
        contrib[late] = ((1.0 - surv_tau[late]) ** 2) / g_tau
        scores[k] = contrib.mean()
    return float(_trapezoid(scores, grid) / (grid[-1] - grid[0]))


def d_calibration_pvalue(cif: np.ndarray, event: np.ndarray, bins: int = 10) -> float:
    counts = np.zeros(bins)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for value, ev in zip(np.clip(cif, _EPS, 1.0 - _EPS), event):
        if ev == 1.0:
            counts[min(int(value * bins), bins - 1)] += 1.0
        else:
            survival = 1.0 - value
            for b in range(bins):
                lo, hi = edges[b], edges[b + 1]
                overlap = max(0.0, min(hi, 1.0) - max(lo, value))
                counts[b] += overlap / survival
    expected = counts.sum() / bins
    if expected <= 0:
        return 1.0
    stat = float(np.sum((counts - expected) ** 2 / expected))
    return float(stats.chi2.sf(stat, bins - 1))
