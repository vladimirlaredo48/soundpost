"""CRESCO fine-tuning loop.

Ref: Algorithm 1 (forward pass followed by a gradient step on the composite
objective), with gradient accumulation, AMP, EMA and the progress-driven loss
weighting from Eq. (7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.utils.data import DataLoader

from soundpost.archings.instrument import EarlstBatch
from soundpost.bench.amp import AmpContext
from soundpost.bench.checkpoint import save_checkpoint
from soundpost.bench.ema import ParameterEma
from soundpost.bench.optim import build_optimizer, build_scheduler, clip_gradients
from soundpost.jigs.logs import get_logger
from soundpost.settings import ExperimentConfig
from soundpost.voicing.cresco import cresco_objective

_LOG = get_logger("soundpost.bench")


def move_batch(batch: EarlstBatch, device: torch.device) -> EarlstBatch:
    return replace(
        batch,
        volume=batch.volume.to(device),
        continuous=batch.continuous.to(device),
        categorical={k: v.to(device) for k, v in batch.categorical.items()},
        time_index=batch.time_index.to(device),
        event=batch.event.to(device),
        hbp_target=batch.hbp_target.to(device),
    )


@dataclass
class StepLog:
    step: int
    total: float
    nll: float
    dcal: float
    rank: float
    hbp: float


class Trainer:
    def __init__(self, model: nn.Module, cfg: ExperimentConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = build_optimizer(self.model, cfg.optim)
        self.amp = AmpContext(cfg.train.precision, device.type)
        self.ema = ParameterEma(self.model, cfg.train.ema_decay)

    def _total_steps(self, loader: DataLoader) -> int:
        per_epoch = math.ceil(len(loader) / self.cfg.train.grad_accum)
        return max(per_epoch * self.cfg.train.epochs, self.cfg.optim.max_steps, 1)

    def fit(self, loader: DataLoader) -> list[StepLog]:
        accum = self.cfg.train.grad_accum
        total = self._total_steps(loader)
        scheduler = build_scheduler(self.optimizer, self.cfg.optim, total)
        history: list[StepLog] = []
        global_step = 0
        for _ in range(self.cfg.train.epochs):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            for i, raw in enumerate(loader):
                batch = move_batch(raw, self.device)
                progress = global_step / total
                with self.amp.autocast():
                    output = self.model(batch)
                    terms = cresco_objective(output, batch, self.cfg.loss, progress)
                    loss = terms.total / accum
                self.amp.scaler.scale(loss).backward()
                if (i + 1) % accum == 0:
                    self.amp.scaler.unscale_(self.optimizer)
                    clip_gradients(self.model, self.cfg.optim.grad_clip)
                    self.amp.scaler.step(self.optimizer)
                    self.amp.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    self.ema.update(self.model)
                    history.append(
                        StepLog(
                            step=global_step,
                            total=float(terms.total.detach()),
                            nll=float(terms.nll.detach()),
                            dcal=float(terms.dcal.detach()),
                            rank=float(terms.rank.detach()),
                            hbp=float(terms.hbp.detach()),
                        )
                    )
                    if global_step % self.cfg.train.log_every == 0:
                        _LOG.info(
                            "step %d loss %.4f nll %.4f dcal %.4f rank %.4f",
                            global_step,
                            history[-1].total,
                            history[-1].nll,
                            history[-1].dcal,
                            history[-1].rank,
                        )
                    if global_step and global_step % self.cfg.train.ckpt_every == 0:
                        self.save(f"{self.cfg.train.out_dir}/step_{global_step}.ckpt", global_step)
                    global_step += 1
                    if global_step >= total:
                        break
        return history

    def save(self, path: str, step: int) -> None:
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            step,
            self.cfg.train.seed,
            extra={"ema": self.ema.state_dict(), "config": self.cfg.model_dump()},
        )
