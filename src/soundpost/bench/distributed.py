"""Distributed data-parallel helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def maybe_init_distributed() -> tuple[int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
