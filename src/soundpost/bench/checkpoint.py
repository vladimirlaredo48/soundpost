"""Atomic checkpointing with seed provenance."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "seed": seed,
        "rng": torch.get_rng_state(),
    }
    if extra:
        payload.update(extra)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
