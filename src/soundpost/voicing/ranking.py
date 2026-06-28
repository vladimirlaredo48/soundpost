"""Censoring-aware patient-pair ranking loss.

Sharpens discrimination without adding probability mass: a pair (i, j) is
comparable when i recurs at t_i and j is still at risk past t_i, and the loss
rewards a larger 24-month risk for i than for j.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def ranking_loss(
    risk: Tensor,
    time_index: Tensor,
    event: Tensor,
    temperature: float,
) -> Tensor:
    earlier = time_index.unsqueeze(1) < time_index.unsqueeze(0)
    comparable = (event.unsqueeze(1) > 0.5) & earlier
    weight = comparable.to(risk.dtype)
    diff = risk.unsqueeze(1) - risk.unsqueeze(0)
    penalty = F.softplus(-diff / temperature)
    denom = weight.sum().clamp_min(1.0)
    return (weight * penalty).sum() / denom
