"""EARLST forward pass.

Ref: Algorithm 1 (encode -> Remnant-State Token -> gated FiLM fusion ->
discrete-time hazard -> calibrated survival curve).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from soundpost.archings.bridge import HazardHead, risk_at_horizon, survival_from_hazard
from soundpost.archings.purfling import ClinicalEmbed, FiLM, GatedFusion
from soundpost.archings.resonator import build_encoder
from soundpost.archings.soul import RemnantStateToken
from soundpost.settings import ModelConfig


@dataclass
class EarlstOutput:
    hazard: Tensor
    survival: Tensor
    risk: Tensor
    remnant_state: Tensor
    hbp_hat: Tensor


@dataclass
class EarlstBatch:
    volume: Tensor
    continuous: Tensor
    categorical: dict[str, Tensor]
    time_index: Tensor
    event: Tensor
    hbp_target: Tensor


class Earlst(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_model = cfg.encoder.d_model
        self.encoder = build_encoder(cfg.encoder, cfg.volume)
        self.use_rst = cfg.use_rst
        self.rst = RemnantStateToken(cfg.rst, d_model)
        self.clinical = ClinicalEmbed(cfg.clinical, d_model)
        self.fusion = GatedFusion(cfg.fusion, d_model)
        self.head = HazardHead(cfg.head, d_model)
        self.film = FiLM(cfg.fusion, self.head.feature_dim)

    def encode_state(self, volume: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(volume)
        if self.use_rst:
            return self.rst(features)
        pooled = features.mean(dim=1)
        return pooled, volume.new_zeros(volume.shape[0])

    def forward(self, batch: EarlstBatch) -> EarlstOutput:
        state, hbp_hat = self.encode_state(batch.volume)
        z = self.clinical(batch.continuous, batch.categorical)
        c = self.fusion(state, z)
        phi = self.head.penultimate(state)
        phi_tilde = self.film(c, phi)
        hazard = self.head.hazard_from_features(phi_tilde)
        survival = survival_from_hazard(hazard)
        risk = 1.0 - survival[..., -1]
        return EarlstOutput(
            hazard=hazard,
            survival=survival,
            risk=risk,
            remnant_state=state,
            hbp_hat=hbp_hat,
        )

    @torch.no_grad()
    def predict_risk(self, batch: EarlstBatch) -> Tensor:
        return risk_at_horizon(self.forward(batch).hazard)
