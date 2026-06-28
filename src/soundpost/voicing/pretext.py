"""Encoder pretext objectives.

A VoCo-style volume-contrast InfoNCE pretrains the encoder on the public liver
volumes, and a Dice term distils the segmentation prior used as the spatial
anchor for the surgical-bed region of interest.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_EPS = 1.0e-6


def voco_infonce(embeddings: Tensor, temperature: float) -> Tensor:
    feats = F.normalize(embeddings, dim=-1)
    logits = feats @ feats.t() / temperature
    n = logits.shape[0]
    logits = logits - torch.eye(n, device=logits.device) * 1.0e9
    half = n // 2
    target = torch.arange(n, device=logits.device)
    target = (target + half) % n
    return F.cross_entropy(logits, target)


def dice_prior(pred: Tensor, mask: Tensor) -> Tensor:
    probs = torch.sigmoid(pred)
    dims = tuple(range(1, probs.dim()))
    inter = (probs * mask).sum(dims)
    union = probs.sum(dims) + mask.sum(dims)
    dice = (2.0 * inter + _EPS) / (union + _EPS)
    return 1.0 - dice.mean()
