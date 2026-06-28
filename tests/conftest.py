from __future__ import annotations

import pytest
import torch

from soundpost.archings.instrument import EarlstBatch
from soundpost.settings import (
    DataConfig,
    EncoderConfig,
    FusionConfig,
    HeadConfig,
    ModelConfig,
    RstConfig,
    VolumeSpec,
)
from soundpost.tonewood.cohort import collate_batch
from soundpost.tonewood.synth import SyntheticCohort


@pytest.fixture(scope="session")
def tiny_model() -> ModelConfig:
    return ModelConfig(
        use_rst=True,
        volume=VolumeSpec(n_phases=5, channels=1, size=(16, 16, 16)),
        encoder=EncoderConfig(
            kind="ssm", patch_size=8, d_model=32, depth=2, d_state=8, expand=2, conv_kernel=3
        ),
        rst=RstConfig(n_queries=4, n_heads=2, hbp_hidden=16),
        fusion=FusionConfig(hidden=32, film_hidden=16, film_layers=2),
        head=HeadConfig(n_intervals=4, body_hidden=16, body_layers=2, dropout=0.0),
    )


@pytest.fixture(scope="session")
def tiny_data() -> DataConfig:
    return DataConfig(synthetic=True, interval_months=6, synthetic_size=24)


@pytest.fixture
def cohort(tiny_model: ModelConfig, tiny_data: DataConfig) -> SyntheticCohort:
    return SyntheticCohort(24, tiny_model, tiny_data, seed=0)


@pytest.fixture
def batch(cohort: SyntheticCohort) -> EarlstBatch:
    torch.manual_seed(0)
    return collate_batch([cohort[i] for i in range(len(cohort))])
