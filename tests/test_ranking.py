from __future__ import annotations

import torch

from soundpost.voicing.ranking import ranking_loss


def test_correct_order_has_lower_loss() -> None:
    time_index = torch.tensor([0, 2, 1])
    event = torch.tensor([1.0, 0.0, 1.0])
    good = torch.tensor([0.9, 0.1, 0.5])
    bad = torch.tensor([0.1, 0.9, 0.5])
    assert ranking_loss(good, time_index, event, 0.1) < ranking_loss(bad, time_index, event, 0.1)


def test_no_comparable_pairs_is_zero() -> None:
    time_index = torch.tensor([3, 3])
    event = torch.tensor([0.0, 0.0])
    loss = ranking_loss(torch.tensor([0.2, 0.8]), time_index, event, 0.1)
    assert loss.item() == 0.0


def test_loss_is_nonnegative() -> None:
    torch.manual_seed(1)
    risk = torch.rand(8)
    time_index = torch.randint(0, 4, (8,))
    event = (torch.rand(8) > 0.5).float()
    assert ranking_loss(risk, time_index, event, 0.1).item() >= 0.0
