"""Differentiable D-calibration penalty.

Ref: Eq. (9), L_Dcal = sum_b (rho_b - 1/B)^2 over B equal-probability bins of
the predicted cumulative incidence at the observed time. Uncensored subjects use
a temperature-softened one-hot bin membership; censored subjects spread their
remaining mass across higher bins through a smoothed step, keeping the penalty
differentiable as required for in-training calibration.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1.0e-7


def predicted_cif(survival: Tensor, time_index: Tensor) -> Tensor:
    observed = survival.gather(1, time_index.unsqueeze(1)).squeeze(1)
    return (1.0 - observed).clamp(_EPS, 1.0 - _EPS)


def dcal_loss(
    survival: Tensor,
    time_index: Tensor,
    event: Tensor,
    bins: int,
    temperature: float,
) -> Tensor:
    pit = predicted_cif(survival, time_index)
    centers = (torch.arange(bins, device=survival.device) + 0.5) / bins
    edges = (torch.arange(bins, device=survival.device) + 1.0) / bins
    sq = ((pit.unsqueeze(1) - centers.unsqueeze(0)) / temperature) ** 2
    event_mass = torch.softmax(-sq, dim=1)
    above = torch.sigmoid((edges.unsqueeze(0) - pit.unsqueeze(1)) / temperature)
    censored_mass = above / above.sum(dim=1, keepdim=True).clamp_min(_EPS)
    membership = event.unsqueeze(1) * event_mass + (1.0 - event).unsqueeze(1) * censored_mass
    rho = membership.mean(dim=0)
    return ((rho - 1.0 / bins) ** 2).sum()
