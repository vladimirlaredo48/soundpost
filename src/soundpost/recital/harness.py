"""Prediction gathering and time encoding for evaluation.

Recurrence-free survivors are pushed just past the 24-month horizon so the
cumulative/dynamic operating point separates them from in-window recurrences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from soundpost.bench.loop import move_batch


@dataclass
class Predictions:
    risk: np.ndarray
    survival: np.ndarray
    hazard: np.ndarray
    time_index: np.ndarray
    time_metric: np.ndarray
    event: np.ndarray
    cif: np.ndarray
    horizon: float
    n_intervals: int


@torch.no_grad()
def gather_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device, n_intervals: int
) -> Predictions:
    model.eval().to(device)
    risk, survival, hazard, time_index, event = [], [], [], [], []
    for raw in loader:
        batch = move_batch(raw, device)
        out = model(batch)
        risk.append(out.risk.cpu().numpy())
        survival.append(out.survival.cpu().numpy())
        hazard.append(out.hazard.cpu().numpy())
        time_index.append(batch.time_index.cpu().numpy())
        event.append(batch.event.cpu().numpy())
    risk_a = np.concatenate(risk)
    surv_a = np.concatenate(survival)
    haz_a = np.concatenate(hazard)
    idx_a = np.concatenate(time_index).astype(int)
    ev_a = np.concatenate(event).astype(float)

    time_metric = idx_a.astype(float).copy()
    survivor = (ev_a == 0.0) & (idx_a == n_intervals - 1)
    time_metric[survivor] = float(n_intervals)
    horizon = (n_intervals - 1) + 0.5
    cif = 1.0 - surv_a[np.arange(surv_a.shape[0]), np.clip(idx_a, 0, n_intervals - 1)]
    return Predictions(
        risk=risk_a,
        survival=surv_a,
        hazard=haz_a,
        time_index=idx_a,
        time_metric=time_metric,
        event=ev_a,
        cif=cif,
        horizon=horizon,
        n_intervals=n_intervals,
    )
