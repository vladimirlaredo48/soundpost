"""Discrimination metrics: Harrell's concordance and IPCW cumulative/dynamic AUROC.

Ref: Results and Methods (Harrell's c-index; inverse-probability-of-censoring
weighted 24-month AUROC).
"""

from __future__ import annotations

import numpy as np


def _km_censoring(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    censored = 1.0 - event
    uniq = np.unique(time)
    g = 1.0
    values = np.empty(uniq.shape[0])
    for idx, u in enumerate(uniq):
        at_risk = np.sum(time >= u)
        drops = np.sum((time == u) & (censored == 1.0))
        if at_risk > 0:
            g *= 1.0 - drops / at_risk
        values[idx] = g
    return uniq, values


def _g_lookup(uniq: np.ndarray, values: np.ndarray, query: np.ndarray, left: bool) -> np.ndarray:
    out = np.ones(query.shape[0])
    for i, q in enumerate(query):
        mask = uniq < q if left else uniq <= q
        if mask.any():
            out[i] = values[mask][-1]
    return np.clip(out, 1.0e-8, 1.0)


def harrell_cindex(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    ti = time[:, None]
    tj = time[None, :]
    comparable = (event[:, None] == 1.0) & (tj > ti)
    diff = risk[:, None] - risk[None, :]
    concordant = (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)
    denom = comparable.sum()
    if denom == 0:
        return 0.5
    return float((comparable * concordant).sum() / denom)


def ipcw_auroc(risk: np.ndarray, time: np.ndarray, event: np.ndarray, horizon: float) -> float:
    uniq, values = _km_censoring(time, event)
    cases = (time <= horizon) & (event == 1.0)
    controls = time > horizon
    if not cases.any() or not controls.any():
        return 0.5
    case_w = 1.0 / _g_lookup(uniq, values, time, left=True)
    control_w = 1.0 / _g_lookup(uniq, values, np.array([horizon]), left=False)[0]
    ri = risk[cases][:, None]
    rj = risk[controls][None, :]
    indicator = (ri > rj).astype(float) + 0.5 * (ri == rj).astype(float)
    weights = np.broadcast_to(case_w[cases][:, None] * control_w, indicator.shape)
    return float((weights * indicator).sum() / weights.sum())
