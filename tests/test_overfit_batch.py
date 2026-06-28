from __future__ import annotations

import torch

from soundpost.acoustics.discrimination import harrell_cindex
from soundpost.archings.instrument import Earlst, EarlstBatch
from soundpost.settings import LossConfig, ModelConfig
from soundpost.voicing.cresco import cresco_objective


def test_single_batch_overfits(tiny_model: ModelConfig, batch: EarlstBatch) -> None:
    torch.manual_seed(0)
    model = Earlst(tiny_model).train()
    loss_cfg = LossConfig(dcal_bins=4, dcal_temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for step in range(200):
        out = model(batch)
        terms = cresco_objective(out, batch, loss_cfg, progress=step / 200)
        optimizer.zero_grad()
        terms.total.backward()
        optimizer.step()
        if first is None:
            first = float(terms.nll.detach())
    model.eval()
    with torch.no_grad():
        out = model(batch)
    final = float(cresco_objective(out, batch, loss_cfg, 1.0).nll.detach())
    assert final < first * 0.6
    cindex = harrell_cindex(
        out.risk.numpy(), batch.time_index.numpy().astype(float), batch.event.numpy()
    )
    assert cindex > 0.7
