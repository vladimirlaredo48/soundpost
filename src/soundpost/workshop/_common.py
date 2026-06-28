from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from soundpost.archings.instrument import Earlst
from soundpost.jigs.io import load_experiment
from soundpost.settings import ExperimentConfig
from soundpost.tonewood.cohort import FileCohort, Sample, collate_batch
from soundpost.tonewood.synth import SyntheticCohort


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_config(config: str, override: tuple[str, ...]) -> ExperimentConfig:
    return load_experiment(config, overrides=list(override))


def build_model(cfg: ExperimentConfig) -> Earlst:
    return Earlst(cfg.model)


def build_loader(cfg: ExperimentConfig, manifest: str | None, shuffle: bool) -> DataLoader:
    dataset: Dataset[Sample]
    if cfg.data.synthetic or manifest is None:
        dataset = SyntheticCohort(cfg.data.synthetic_size, cfg.model, cfg.data, cfg.train.seed)
    else:
        dataset = FileCohort(Path(manifest), cfg.model, cfg.data)
    return DataLoader(
        dataset,
        batch_size=cfg.train.micro_batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=0,
    )
