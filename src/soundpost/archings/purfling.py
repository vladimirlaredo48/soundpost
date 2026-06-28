"""Gated clinical-imaging fusion and FiLM conditioning.

Ref: Eq. (5) c = sigma(W_g[s;z]) * W_s s + (1 - sigma(W_g[s;z])) * W_z z and
Eq. (6) phi_tilde = gamma * phi + beta.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from soundpost.settings import ClinicalSpec, FusionConfig


class ClinicalEmbed(nn.Module):
    def __init__(self, spec: ClinicalSpec, d_model: int) -> None:
        super().__init__()
        self.continuous = spec.continuous
        self.categorical = list(spec.categorical.items())
        self.tables = nn.ModuleDict(
            {name: nn.Embedding(card, spec.embed_dim) for name, card in self.categorical}
        )
        self.proj = nn.Linear(spec.raw_dim, d_model)

    def forward(self, cont: Tensor, cat: dict[str, Tensor]) -> Tensor:
        parts = [cont]
        for name, _ in self.categorical:
            parts.append(self.tables[name](cat[name]))
        return self.proj(torch.cat(parts, dim=-1))


class GatedFusion(nn.Module):
    def __init__(self, cfg: FusionConfig, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)
        self.w_s = nn.Linear(d_model, d_model, bias=False)
        self.w_z = nn.Linear(d_model, d_model, bias=False)

    def forward(self, s: Tensor, z: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([s, z], dim=-1)))
        return gate * self.w_s(s) + (1.0 - gate) * self.w_z(z)


class FiLM(nn.Module):
    def __init__(self, cfg: FusionConfig, feature_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = cfg.hidden
        for _ in range(cfg.film_layers - 1):
            layers += [nn.Linear(width, cfg.film_hidden), nn.GELU()]
            width = cfg.film_hidden
        layers.append(nn.Linear(width, 2 * feature_dim))
        self.net = nn.Sequential(*layers)
        self.feature_dim = feature_dim

    def forward(self, c: Tensor, phi: Tensor) -> Tensor:
        gamma, beta = self.net(c).chunk(2, dim=-1)
        return gamma * phi + beta
