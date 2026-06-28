"""Censoring-aware discrete-time hazard head.

Ref: Eq. (1) h_t = P(recur in interval t | recurrence-free before t, X) and
Eq. (2) S(t) = prod_{u<=t} (1 - h_u). The head body produces the penultimate
features phi that FiLM modulates before the per-interval sigmoid.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from soundpost.settings import HeadConfig


class HazardHead(nn.Module):
    def __init__(self, cfg: HeadConfig, d_model: int) -> None:
        super().__init__()
        self.n_intervals = cfg.n_intervals
        layers: list[nn.Module] = []
        width = d_model
        for _ in range(cfg.body_layers):
            layers += [nn.Linear(width, cfg.body_hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            width = cfg.body_hidden
        self.body = nn.Sequential(*layers)
        self.feature_dim = cfg.body_hidden
        self.out = nn.Linear(cfg.body_hidden, cfg.n_intervals)

    def penultimate(self, s: Tensor) -> Tensor:
        return self.body(s)

    def hazard_from_features(self, phi: Tensor) -> Tensor:
        return torch.sigmoid(self.out(phi))

    def forward(self, phi: Tensor) -> Tensor:
        return self.hazard_from_features(phi)


def survival_from_hazard(hazard: Tensor) -> Tensor:
    return torch.cumprod(1.0 - hazard, dim=-1)


def risk_at_horizon(hazard: Tensor) -> Tensor:
    return 1.0 - survival_from_hazard(hazard)[..., -1]
