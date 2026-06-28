"""Hepatobiliary-phase anchoring loss.

Ref: Eq. (10), L_HBP = || g_HBP(s) - r_HBP ||^2, tying the Remnant-State Token
to the relative-enhancement surrogate of hepatocyte transporter function.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def anchor_loss(hbp_hat: Tensor, hbp_target: Tensor) -> Tensor:
    return F.mse_loss(hbp_hat, hbp_target)
