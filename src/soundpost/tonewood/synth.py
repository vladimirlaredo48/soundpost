"""Planted-signal synthetic cohort.

A latent severity drives the discrete-time hazard, the hepatobiliary-phase
enhancement of the surgical-bed region and the correlated clinical fields, so a
correctly wired model can recover discrimination and calibration on data that
ships with the repository.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from soundpost.settings import DataConfig, ModelConfig
from soundpost.tonewood.clinical import ClinicalEncoder, ClinicalRecord
from soundpost.tonewood.cohort import Sample


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


class SyntheticCohort(Dataset[Sample]):
    def __init__(
        self,
        n: int,
        model: ModelConfig,
        data: DataConfig,
        seed: int = 0,
        volume_size: tuple[int, int, int] | None = None,
    ) -> None:
        self.model = model
        self.encoder = ClinicalEncoder(model.clinical)
        self.size = volume_size or model.volume.size
        self.n_intervals = model.head.n_intervals
        self.channels = model.volume.n_phases * model.volume.channels
        rng = np.random.default_rng(seed)
        self.samples = [self._draw(rng) for _ in range(n)]

    def _draw(self, rng: np.random.Generator) -> Sample:
        severity = float(rng.normal())
        base = np.linspace(-2.5, -1.4, self.n_intervals)
        hazard = 1.0 / (1.0 + np.exp(-(base + 0.95 * severity)))
        time_idx = self.n_intervals - 1
        event = 0.0
        for t in range(self.n_intervals):
            if rng.random() < hazard[t]:
                time_idx, event = t, 1.0
                break
        censor = int(rng.integers(self.n_intervals // 2, self.n_intervals))
        if event == 0.0:
            time_idx = censor
        elif censor < time_idx:
            time_idx, event = censor, 0.0

        hbp = _sigmoid(-severity)
        depth, height, width = self.size
        volume = rng.normal(0.2, 0.05, size=(self.channels,) + self.size).astype(np.float32)
        volume[-1, : depth // 2, : height // 2, : width // 2] += np.float32(0.5 - hbp)
        volume[0] += np.float32(0.08 * severity)

        record = ClinicalRecord(
            afp=float(np.exp(rng.normal(3.0 + 0.6 * severity, 0.5))),
            age=float(rng.normal(58.0, 9.0)),
            tumour_size_cm=float(max(0.5, rng.normal(3.5 + 0.4 * severity, 1.2))),
            albi_grade=int(min(2, rng.binomial(2, _sigmoid(0.5 * severity)))),
            bclc_stage=int(min(3, rng.binomial(3, _sigmoid(0.4 * severity)))),
            mvi=int(rng.random() < _sigmoid(severity)),
            aetiology=int(rng.integers(0, 4)),
            resection_margin=int(rng.integers(0, 3)),
            sex=int(rng.random() < 0.8),
            vendor=int(rng.integers(0, 3)),
        )
        return Sample(
            volume=torch.from_numpy(volume),
            continuous=self.encoder.continuous(record),
            categorical=self.encoder.categorical(record),
            time_index=torch.tensor(time_idx, dtype=torch.long),
            event=torch.tensor(event, dtype=torch.float32),
            hbp_target=torch.tensor(hbp, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]
