"""Autocast and gradient scaling driven by the precision config."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from soundpost.settings import Precision

_DTYPE = {Precision.fp16: torch.float16, Precision.bf16: torch.bfloat16}


class AmpContext:
    def __init__(self, precision: Precision, device_type: str) -> None:
        self.precision = precision
        self.device_type = device_type
        self.enabled = precision in _DTYPE
        self.scaler = torch.cuda.amp.GradScaler(enabled=precision is Precision.fp16)

    @contextmanager
    def autocast(self) -> Iterator[None]:
        if self.enabled and self.device_type == "cuda":
            with torch.autocast(device_type="cuda", dtype=_DTYPE[self.precision]):
                yield
        else:
            yield
