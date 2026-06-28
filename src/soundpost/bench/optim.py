"""Optimiser and warmup-cosine schedule, plus the CRESCO progress fraction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from soundpost.settings import OptimConfig


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)


def build_scheduler(optimizer: Optimizer, cfg: OptimConfig, total_steps: int) -> LambdaLR:
    warmup = max(1, cfg.warmup_steps)
    horizon = max(total_steps, warmup + 1)

    def schedule(step: int) -> float:
        if step < warmup:
            return step / warmup
        if cfg.scheduler == "constant":
            return 1.0
        progress = (step - warmup) / (horizon - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda=schedule)


def clip_gradients(model: nn.Module, max_norm: float) -> None:
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
