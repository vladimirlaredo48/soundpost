"""Typed configuration schema for the EARLST stack.

Validated config objects flow between every module so that the compute profile
in configs/experiment/main.yaml is the single source of truth.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Precision(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"


class EncoderKind(str, Enum):
    ssm = "ssm"
    swin_unetr = "swin_unetr"
    resnet3d = "resnet3d"


class VolumeSpec(_Frozen):
    n_phases: int = 5
    channels: int = 1
    size: tuple[int, int, int] = (128, 128, 128)
    spacing_mm: float = 1.0


class ClinicalSpec(_Frozen):
    continuous: tuple[str, ...] = ("afp_log", "age", "tumour_size_cm")
    categorical: dict[str, int] = Field(
        default_factory=lambda: {
            "albi_grade": 3,
            "bclc_stage": 4,
            "mvi": 2,
            "aetiology": 4,
            "resection_margin": 3,
            "sex": 2,
            "vendor": 3,
        }
    )
    embed_dim: int = 16

    @property
    def raw_dim(self) -> int:
        return len(self.continuous) + len(self.categorical) * self.embed_dim


class EncoderConfig(_Frozen):
    kind: EncoderKind = EncoderKind.ssm
    patch_size: int = 8
    d_model: int = 192
    depth: int = 12
    d_state: int = 16
    expand: int = 2
    conv_kernel: int = 4
    dropout: float = 0.0
    bidirectional: bool = True


class RstConfig(_Frozen):
    n_queries: int = 8
    n_heads: int = 4
    hbp_hidden: int = 64


class FusionConfig(_Frozen):
    hidden: int = 192
    film_hidden: int = 128
    film_layers: int = 2


class HeadConfig(_Frozen):
    n_intervals: int = 12
    body_hidden: int = 128
    body_layers: int = 2
    dropout: float = 0.1


class ModelConfig(_Frozen):
    volume: VolumeSpec = VolumeSpec()
    clinical: ClinicalSpec = ClinicalSpec()
    encoder: EncoderConfig = EncoderConfig()
    rst: RstConfig = RstConfig()
    fusion: FusionConfig = FusionConfig()
    head: HeadConfig = HeadConfig()
    use_rst: bool = True

    @model_validator(mode="after")
    def _check_dims(self) -> ModelConfig:
        if self.fusion.hidden != self.encoder.d_model:
            raise ValueError("fusion.hidden must equal encoder.d_model")
        return self


class LossConfig(_Frozen):
    lambda_cal_start: float = 0.1
    lambda_cal_end: float = 1.0
    lambda_rank_start: float = 1.0
    lambda_rank_end: float = 0.2
    lambda_aux: float = 0.1
    dcal_bins: int = 10
    dcal_temperature: float = 0.05
    rank_temperature: float = 0.1
    likelihood_only: bool = False


class OptimConfig(_Frozen):
    name: Literal["adamw"] = "adamw"
    lr: float = 2.0e-5
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    grad_clip: float = 1.0
    scheduler: Literal["cosine", "constant"] = "cosine"
    warmup_steps: int = 2000
    max_steps: int = 0


class PretrainConfig(_Frozen):
    enabled: bool = True
    voco_crops: int = 4
    voco_temperature: float = 0.07
    seg_prior_weight: float = 0.5
    steps: int = 50000


class DataConfig(_Frozen):
    root: str = "data/eob_recur"
    dev_sites: tuple[str, ...] = ("A", "B")
    external_site: str = "C"
    prospective_site: str = "D"
    t_max_months: int = 24
    interval_months: int = 2
    val_fraction: float = 0.2
    num_workers: int = 8
    synthetic: bool = False
    synthetic_size: int = 256


class TrainConfig(_Frozen):
    seed: int = 1337
    seeds: tuple[int, ...] = (1337, 7, 11, 23, 42, 71, 101, 202, 303, 404)
    epochs: int = 800
    micro_batch_size: int = 2
    grad_accum: int = 16
    world_size: int = 8
    precision: Precision = Precision.fp32
    ema_decay: float = 0.9999
    log_every: int = 50
    ckpt_every: int = 1000
    out_dir: str = "runs/earlst"

    @property
    def effective_batch(self) -> int:
        return self.micro_batch_size * self.grad_accum * self.world_size


class EvalConfig(_Frozen):
    bootstrap_resamples: int = 2000
    decision_thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    operating_threshold: float = 0.40
    abstention_fraction: float = 0.10
    recalibrate_intercept: bool = False


class ExperimentConfig(_Frozen):
    name: str = "main"
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    optim: OptimConfig = OptimConfig()
    loss: LossConfig = LossConfig()
    pretrain: PretrainConfig = PretrainConfig()
    evaluation: EvalConfig = EvalConfig()
