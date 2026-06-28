from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from soundpost.archings.instrument import Earlst
from soundpost.bench.loop import Trainer
from soundpost.jigs.io import load_experiment
from soundpost.jigs.seed import set_seed
from soundpost.recital.harness import gather_predictions
from soundpost.recital.report import build_report
from soundpost.tonewood.cohort import collate_batch
from soundpost.tonewood.synth import SyntheticCohort

_SMOKE = Path(__file__).resolve().parents[1] / "configs" / "experiment" / "_smoke.yaml"


def _loader(cfg) -> DataLoader:
    dataset = SyntheticCohort(cfg.data.synthetic_size, cfg.model, cfg.data, seed=0)
    return DataLoader(
        dataset,
        batch_size=cfg.train.micro_batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )


def test_smoke_training_reduces_loss() -> None:
    set_seed(0)
    cfg = load_experiment(_SMOKE, overrides=["train.epochs=6"])
    model = Earlst(cfg.model)
    trainer = Trainer(model, cfg, torch.device("cpu"))
    history = trainer.fit(_loader(cfg))
    assert len(history) >= 2
    half = len(history) // 2
    early = sum(h.total for h in history[:half]) / half
    late = sum(h.total for h in history[half:]) / (len(history) - half)
    assert late < early


def test_smoke_report_runs() -> None:
    set_seed(0)
    cfg = load_experiment(_SMOKE)
    model = Earlst(cfg.model)
    pred = gather_predictions(model, _loader(cfg), torch.device("cpu"), cfg.model.head.n_intervals)
    report = build_report(pred, cfg.evaluation, seed=0)
    assert 0.0 <= report.auroc[0] <= 1.0
    assert report.net_benefit["model"].shape == (len(cfg.evaluation.decision_thresholds),)
