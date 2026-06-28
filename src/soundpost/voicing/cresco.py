"""CRESCO composite objective.

Ref: Eq. (7), L = L_NLL + lambda_cal L_Dcal + lambda_rank L_rank + lambda_aux L_HBP.
The ranking weight decays and the calibration weight grows with training
progress so ranking shapes the representation before calibration fixes its scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from soundpost.archings.instrument import EarlstBatch, EarlstOutput
from soundpost.settings import LossConfig
from soundpost.voicing.anchor import anchor_loss
from soundpost.voicing.dcal import dcal_loss
from soundpost.voicing.likelihood import nll_loss
from soundpost.voicing.ranking import ranking_loss


@dataclass
class LossTerms:
    total: Tensor
    nll: Tensor
    dcal: Tensor
    rank: Tensor
    hbp: Tensor
    lambda_cal: float
    lambda_rank: float


def _ramp(start: float, end: float, progress: float) -> float:
    p = min(max(progress, 0.0), 1.0)
    return start + (end - start) * p


def cresco_objective(
    output: EarlstOutput,
    batch: EarlstBatch,
    cfg: LossConfig,
    progress: float,
) -> LossTerms:
    nll = nll_loss(output.hazard, batch.time_index, batch.event)
    dcal = dcal_loss(
        output.survival, batch.time_index, batch.event, cfg.dcal_bins, cfg.dcal_temperature
    )
    rank = ranking_loss(output.risk, batch.time_index, batch.event, cfg.rank_temperature)
    hbp = anchor_loss(output.hbp_hat, batch.hbp_target)
    lam_cal = _ramp(cfg.lambda_cal_start, cfg.lambda_cal_end, progress)
    lam_rank = _ramp(cfg.lambda_rank_start, cfg.lambda_rank_end, progress)
    if cfg.likelihood_only:
        total = nll
    else:
        total = nll + lam_cal * dcal + lam_rank * rank + cfg.lambda_aux * hbp
    return LossTerms(
        total=total,
        nll=nll,
        dcal=dcal,
        rank=rank,
        hbp=hbp,
        lambda_cal=lam_cal,
        lambda_rank=lam_rank,
    )
