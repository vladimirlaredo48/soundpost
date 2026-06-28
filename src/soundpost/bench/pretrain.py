"""Encoder pretraining on the public liver volumes.

Ref: Implementation and training (VoCo volume-contrast self-supervision and a
MedSAM-derived tumour segmentation prior, before end-to-end fine-tuning).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.utils.data import DataLoader

from soundpost.archings.resonator import positional_grid
from soundpost.bench.optim import build_optimizer
from soundpost.settings import ModelConfig, OptimConfig, PretrainConfig
from soundpost.voicing.pretext import dice_prior, voco_infonce


class PretextModel(nn.Module):
    def __init__(self, encoder: nn.Module, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.encoder = encoder
        d_model = model_cfg.encoder.d_model
        self.grid = positional_grid(model_cfg.volume.size, model_cfg.encoder.patch_size)
        self.project = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 128)
        )
        self.seg = nn.Linear(d_model, 1)

    def forward(self, volume: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.encoder(volume)
        embedding = self.project(tokens.mean(dim=1))
        seg = self.seg(tokens).squeeze(-1)
        return embedding, seg


@dataclass
class PretrainBatch:
    volume: Tensor
    mask: Tensor


def pretrain_encoder(
    model: PretextModel,
    loader: DataLoader,
    pre_cfg: PretrainConfig,
    optim_cfg: OptimConfig,
    device: torch.device,
) -> list[float]:
    optimizer = build_optimizer(model, optim_cfg)
    model.to(device).train()
    losses: list[float] = []
    for seen, batch in enumerate(loader):
        volume = batch.volume.to(device)
        mask = rearrange(batch.mask.to(device), "b d h w -> b (d h w)")
        embedding, seg = model(volume)
        contrast = voco_infonce(embedding, pre_cfg.voco_temperature)
        prior = dice_prior(seg, mask)
        loss = contrast + pre_cfg.seg_prior_weight * prior
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if seen + 1 >= pre_cfg.steps:
            break
    return losses
