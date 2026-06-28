from __future__ import annotations

import torch

from soundpost.archings.instrument import Earlst, EarlstBatch
from soundpost.settings import ModelConfig


def test_forward_shapes_and_ranges(tiny_model: ModelConfig, batch: EarlstBatch) -> None:
    model = Earlst(tiny_model).eval()
    out = model(batch)
    n = batch.event.shape[0]
    t = tiny_model.head.n_intervals
    assert out.hazard.shape == (n, t)
    assert out.survival.shape == (n, t)
    assert out.risk.shape == (n,)
    assert torch.all((out.hazard > 0) & (out.hazard < 1))
    assert torch.all((out.risk >= 0) & (out.risk <= 1))


def test_survival_is_monotone_non_increasing(tiny_model: ModelConfig, batch: EarlstBatch) -> None:
    model = Earlst(tiny_model).eval()
    survival = model(batch).survival
    diffs = survival[:, 1:] - survival[:, :-1]
    assert torch.all(diffs <= 1e-6)


def test_gradients_reach_encoder_and_token(tiny_model: ModelConfig, batch: EarlstBatch) -> None:
    model = Earlst(tiny_model).train()
    out = model(batch)
    out.risk.sum().backward()
    grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert model.rst.queries.grad is not None


def test_disabling_rst_changes_state_path(tiny_model: ModelConfig, batch: EarlstBatch) -> None:
    without = tiny_model.model_copy(update={"use_rst": False})
    model = Earlst(without).eval()
    out = model(batch)
    assert out.remnant_state.shape == (batch.event.shape[0], tiny_model.encoder.d_model)
    assert torch.all(out.hbp_hat == 0)
