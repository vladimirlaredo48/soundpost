"""Clinical-laboratory feature assembly.

Ref: Image acquisition and preprocessing (alpha-fetoprotein, ALBI grade, BCLC
stage, microvascular invasion, aetiology, resection margin, age, sex, vendor
combined into the feature vector z).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from soundpost.settings import ClinicalSpec


@dataclass
class ClinicalRecord:
    afp: float
    age: float
    tumour_size_cm: float
    albi_grade: int
    bclc_stage: int
    mvi: int
    aetiology: int
    resection_margin: int
    sex: int
    vendor: int


@dataclass
class ClinicalEncoder:
    spec: ClinicalSpec
    age_center: float = 58.0
    age_scale: float = 10.0
    size_center: float = 3.5
    size_scale: float = 2.0
    _names: tuple[str, ...] = field(default_factory=tuple)

    def continuous(self, record: ClinicalRecord) -> Tensor:
        afp_log = math.log1p(max(record.afp, 0.0))
        age = (record.age - self.age_center) / self.age_scale
        size = (record.tumour_size_cm - self.size_center) / self.size_scale
        return torch.tensor([afp_log, age, size], dtype=torch.float32)

    def categorical(self, record: ClinicalRecord) -> dict[str, Tensor]:
        return {
            name: torch.tensor(int(getattr(record, name)), dtype=torch.long)
            for name in self.spec.categorical
        }
