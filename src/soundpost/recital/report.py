"""Assemble the primary discrimination, calibration and decision-curve report.

Ref: Tables 2-4 and Fig. 2 (AUROC, concordance, calibration slope, integrated
calibration index, integrated Brier, D-calibration, net benefit, selective
prediction with abstention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from soundpost.acoustics.benefit import net_benefit
from soundpost.acoustics.calibration import (
    calibration_slope,
    d_calibration_pvalue,
    expected_calibration_error,
    integrated_brier_score,
    integrated_calibration_index,
)
from soundpost.acoustics.discrimination import harrell_cindex, ipcw_auroc
from soundpost.acoustics.inference import bootstrap_ci
from soundpost.recital.harness import Predictions
from soundpost.settings import EvalConfig

_EPS = 1.0e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _recalibrate_intercept(risk: np.ndarray, label: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = _logit(risk[mask])
    y = label[mask]
    offset = 0.0
    for _ in range(50):
        mu = 1.0 / (1.0 + np.exp(-(x + offset)))
        grad = np.sum(y - mu)
        hess = np.sum(mu * (1.0 - mu)) + _EPS
        offset += grad / hess
    return 1.0 / (1.0 + np.exp(-(_logit(risk) + offset)))


@dataclass
class MetricReport:
    auroc: tuple[float, float, float]
    cindex: tuple[float, float, float]
    calibration_slope: float
    integrated_calibration_index: float
    expected_calibration_error: float
    integrated_brier: float
    d_calibration_p: float
    selective_auroc: float
    net_benefit: dict[str, np.ndarray] = field(default_factory=dict)


def build_report(pred: Predictions, cfg: EvalConfig, seed: int = 0) -> MetricReport:
    risk = pred.risk
    if cfg.recalibrate_intercept:
        keep = (pred.time_metric > pred.horizon) | (pred.event == 1.0)
        label = ((pred.time_metric <= pred.horizon) & (pred.event == 1.0)).astype(float)
        risk = _recalibrate_intercept(risk, label, keep)

    def auroc_fn(r: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
        return ipcw_auroc(r, t, e, pred.horizon)

    def cindex_fn(r: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
        return harrell_cindex(r, t, e)

    arrays = (risk, pred.time_metric, pred.event)
    auroc = bootstrap_ci(auroc_fn, arrays, cfg.bootstrap_resamples, seed=seed)
    cindex = bootstrap_ci(cindex_fn, arrays, cfg.bootstrap_resamples, seed=seed + 1)

    grid = np.arange(pred.n_intervals, dtype=float)
    report = MetricReport(
        auroc=auroc,
        cindex=cindex,
        calibration_slope=calibration_slope(risk, pred.time_metric, pred.event, pred.horizon),
        integrated_calibration_index=integrated_calibration_index(
            risk, pred.time_metric, pred.event, pred.horizon
        ),
        expected_calibration_error=expected_calibration_error(
            risk, pred.time_metric, pred.event, pred.horizon
        ),
        integrated_brier=integrated_brier_score(pred.survival, pred.time_metric, pred.event, grid),
        d_calibration_p=d_calibration_pvalue(pred.cif, pred.event, 10),
        selective_auroc=_selective_auroc(risk, pred, cfg),
        net_benefit=net_benefit(
            risk, pred.time_metric, pred.event, np.asarray(cfg.decision_thresholds), pred.horizon
        ),
    )
    return report


def _selective_auroc(risk: np.ndarray, pred: Predictions, cfg: EvalConfig) -> float:
    confidence = np.abs(risk - cfg.operating_threshold)
    keep_n = int(round((1.0 - cfg.abstention_fraction) * risk.shape[0]))
    order = np.argsort(-confidence)[:keep_n]
    return ipcw_auroc(risk[order], pred.time_metric[order], pred.event[order], pred.horizon)
