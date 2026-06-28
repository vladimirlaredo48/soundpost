"""Exponential moving average of model parameters."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn


class ParameterEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def keys(self) -> Iterator[str]:
        return iter(self.shadow)
