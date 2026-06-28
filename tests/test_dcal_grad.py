from __future__ import annotations

import torch

from soundpost.voicing.dcal import dcal_loss


def _loss(survival: torch.Tensor) -> torch.Tensor:
    time_index = torch.tensor([1, 2, 0, 3])
    event = torch.tensor([1.0, 0.0, 1.0, 0.0])
    return dcal_loss(survival, time_index, event, bins=4, temperature=0.1)


def test_dcal_gradient_matches_finite_difference() -> None:
    torch.manual_seed(0)
    survival = torch.rand(4, 4, dtype=torch.float64).clamp(0.05, 0.95).requires_grad_(True)
    loss = _loss(survival)
    loss.backward()
    analytic = survival.grad.clone()
    eps = 1e-6
    numeric = torch.zeros_like(analytic)
    with torch.no_grad():
        flat = survival.view(-1)
        for k in range(flat.numel()):
            base = flat[k].item()
            flat[k] = base + eps
            plus = _loss(survival).item()
            flat[k] = base - eps
            minus = _loss(survival).item()
            flat[k] = base
            numeric.view(-1)[k] = (plus - minus) / (2 * eps)
    assert torch.allclose(analytic, numeric, atol=1e-4)


def test_dcal_is_zero_for_uniform_pit() -> None:
    bins = 5
    pit = (torch.arange(bins * 4, dtype=torch.float64) % bins + 0.5) / bins
    survival = (1.0 - pit).unsqueeze(1).repeat(1, 4)
    time_index = torch.zeros(bins * 4, dtype=torch.long)
    event = torch.ones(bins * 4)
    loss = dcal_loss(survival, time_index, event, bins=bins, temperature=0.02)
    assert loss.item() < 0.05
