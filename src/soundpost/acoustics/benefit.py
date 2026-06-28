"""Decision-curve net benefit.

Ref: Results and Fig. 2b (net benefit 0.214 at the 0.40 adjuvant-decision
threshold versus treat-all and treat-none defaults).
"""

from __future__ import annotations

import numpy as np


def _window_labels(time: np.ndarray, event: np.ndarray, horizon: float) -> np.ndarray:
    keep = (time > horizon) | (event == 1.0)
    label = ((time <= horizon) & (event == 1.0)).astype(float)
    return np.where(keep, label, np.nan)


def net_benefit(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    thresholds: np.ndarray,
    horizon: float,
) -> dict[str, np.ndarray]:
    label = _window_labels(time, event, horizon)
    mask = ~np.isnan(label)
    y = label[mask]
    p = risk[mask]
    n = y.shape[0]
    prevalence = y.mean()
    model = np.empty(thresholds.shape[0])
    treat_all = np.empty(thresholds.shape[0])
    for i, tau in enumerate(thresholds):
        flag = p >= tau
        odds = tau / (1.0 - tau)
        tp = np.sum(flag & (y == 1.0)) / n
        fp = np.sum(flag & (y == 0.0)) / n
        model[i] = tp - fp * odds
        treat_all[i] = prevalence - (1.0 - prevalence) * odds
    return {
        "threshold": thresholds,
        "model": model,
        "treat_all": treat_all,
        "treat_none": np.zeros(thresholds.shape[0]),
    }
