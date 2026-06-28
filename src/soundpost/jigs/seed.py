"""Single seeding entry point; the returned state is embedded in checkpoints."""

from __future__ import annotations

import os
import random
from typing import Any, TypedDict

import numpy as np
import torch


class SeedState(TypedDict):
    seed: int
    python: Any
    numpy: Any
    torch: Any


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_state(seed: int) -> SeedState:
    return SeedState(
        seed=seed,
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
    )


def restore_state(state: SeedState) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
