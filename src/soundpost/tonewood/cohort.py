"""Cohort sample types, time discretisation and batching.

Ref: Problem formulation (the 24-month window is split into T contiguous
intervals; each patient carries a possibly-censored time and event indicator).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from soundpost.archings.instrument import EarlstBatch
from soundpost.settings import DataConfig, ModelConfig
from soundpost.tonewood.clinical import ClinicalEncoder, ClinicalRecord
from soundpost.tonewood.phases import prepare_volume


class Sample(TypedDict):
    volume: Tensor
    continuous: Tensor
    categorical: dict[str, Tensor]
    time_index: Tensor
    event: Tensor
    hbp_target: Tensor


def time_to_index(months: float, interval_months: int, n_intervals: int) -> int:
    idx = int(months // interval_months)
    return max(0, min(idx, n_intervals - 1))


def collate_batch(samples: list[Sample]) -> EarlstBatch:
    keys = samples[0]["categorical"].keys()
    return EarlstBatch(
        volume=torch.stack([s["volume"] for s in samples]),
        continuous=torch.stack([s["continuous"] for s in samples]),
        categorical={k: torch.stack([s["categorical"][k] for s in samples]) for k in keys},
        time_index=torch.stack([s["time_index"] for s in samples]),
        event=torch.stack([s["event"] for s in samples]),
        hbp_target=torch.stack([s["hbp_target"] for s in samples]),
    )


class FileCohort(Dataset[Sample]):
    def __init__(self, manifest: str | Path, model: ModelConfig, data: DataConfig) -> None:
        self.records = [
            json.loads(line) for line in Path(manifest).read_text().splitlines() if line
        ]
        self.model = model
        self.data = data
        self.encoder = ClinicalEncoder(model.clinical)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        rec = self.records[index]
        phases = torch.from_numpy(np.load(rec["volume"])).float()
        mask_path = rec.get("mask")
        mask = torch.from_numpy(np.load(mask_path)).float() if mask_path else None
        volume = prepare_volume(phases, mask, self.model.volume.size)
        clinical = ClinicalRecord(**rec["clinical"])
        idx = time_to_index(
            rec["time_months"], self.data.interval_months, self.model.head.n_intervals
        )
        return Sample(
            volume=volume,
            continuous=self.encoder.continuous(clinical),
            categorical=self.encoder.categorical(clinical),
            time_index=torch.tensor(idx, dtype=torch.long),
            event=torch.tensor(float(rec["event"]), dtype=torch.float32),
            hbp_target=torch.tensor(float(rec["hbp_relative_enhancement"]), dtype=torch.float32),
        )
