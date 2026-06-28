"""Censoring-aware discrete-time negative log-likelihood.

Ref: Eq. (8). Event subjects contribute the log-hazard of their interval plus
the log-survival of prior intervals; censored subjects contribute log-survival
through the last observed interval.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_EPS = 1.0e-7


def nll_loss(hazard: Tensor, time_index: Tensor, event: Tensor) -> Tensor:
    n_intervals = hazard.shape[-1]
    grid = torch.arange(n_intervals, device=hazard.device)
    log_h = torch.log(hazard.clamp_min(_EPS))
    log_s = torch.log((1.0 - hazard).clamp_min(_EPS))
    before = (grid.unsqueeze(0) < time_index.unsqueeze(1)).to(hazard.dtype)
    upto = (grid.unsqueeze(0) <= time_index.unsqueeze(1)).to(hazard.dtype)
    event_at = F.one_hot(time_index, n_intervals).to(hazard.dtype)
    event_term = (event_at * log_h).sum(-1) + (before * log_s).sum(-1)
    censored_term = (upto * log_s).sum(-1)
    ll = event * event_term + (1.0 - event) * censored_term
    return -ll.mean()
