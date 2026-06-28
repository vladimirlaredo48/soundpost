from __future__ import annotations

import math

import torch

from soundpost.archings.bridge import survival_from_hazard
from soundpost.voicing.likelihood import nll_loss


def test_survival_is_cumulative_product() -> None:
    hazard = torch.tensor([[0.1, 0.2, 0.5, 0.25]])
    survival = survival_from_hazard(hazard)
    expected = torch.tensor([[0.9, 0.9 * 0.8, 0.9 * 0.8 * 0.5, 0.9 * 0.8 * 0.5 * 0.75]])
    assert torch.allclose(survival, expected, atol=1e-6)


def test_nll_matches_hand_computation() -> None:
    hazard = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4]])
    time_index = torch.tensor([2, 3])
    event = torch.tensor([1.0, 0.0])
    log = math.log
    subj0 = log(0.4) + log(1 - 0.2) + log(1 - 0.3)
    subj1 = log(1 - 0.1) + log(1 - 0.2) + log(1 - 0.3) + log(1 - 0.4)
    expected = -(subj0 + subj1) / 2.0
    value = nll_loss(hazard, time_index, event)
    assert math.isclose(value.item(), expected, rel_tol=1e-6)


def test_event_earlier_lowers_nll_for_high_hazard() -> None:
    hazard = torch.tensor([[0.05, 0.05, 0.9, 0.05]])
    early = nll_loss(hazard, torch.tensor([2]), torch.tensor([1.0]))
    late = nll_loss(hazard, torch.tensor([0]), torch.tensor([1.0]))
    assert early < late
