"""Remnant-State Token and its hepatobiliary-phase anchor.

Ref: Eq. (4), s = CrossAttn({q_k}, F). The anchor head g_HBP(s) regresses the
hepatobiliary-phase relative-enhancement surrogate used by Eq. (10).
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, nn

from soundpost.settings import RstConfig


class RemnantStateToken(nn.Module):
    def __init__(self, cfg: RstConfig, d_model: int) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.scale = (d_model // cfg.n_heads) ** -0.5
        self.queries = nn.Parameter(torch.randn(cfg.n_queries, d_model) * 0.02)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.anchor = nn.Sequential(
            nn.Linear(d_model, cfg.hbp_hidden),
            nn.GELU(),
            nn.Linear(cfg.hbp_hidden, 1),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        b = features.shape[0]
        q = self.to_q(self.queries).unsqueeze(0).expand(b, -1, -1)
        k = self.to_k(features)
        v = self.to_v(features)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        pooled = rearrange(attn @ v, "b h n d -> b n (h d)")
        token = self.norm(self.proj(pooled).mean(dim=1))
        hbp = self.anchor(token).squeeze(-1)
        return token, hbp
