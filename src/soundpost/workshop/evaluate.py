from __future__ import annotations

import click

from soundpost.bench.checkpoint import load_checkpoint
from soundpost.jigs.logs import setup_logging
from soundpost.recital.harness import gather_predictions
from soundpost.recital.report import build_report
from soundpost.workshop._common import build_loader, build_model, load_config, resolve_device


@click.command(name="evaluate")
@click.option("--config", required=True)
@click.option("--checkpoint", default=None)
@click.option("--manifest", default=None)
@click.option("--override", multiple=True)
@click.option("--device", default="auto")
def evaluate_command(
    config: str,
    checkpoint: str | None,
    manifest: str | None,
    override: tuple[str, ...],
    device: str,
) -> None:
    setup_logging()
    cfg = load_config(config, override)
    dev = resolve_device(device)
    model = build_model(cfg)
    if checkpoint is not None:
        state = load_checkpoint(checkpoint, map_location=str(dev))
        model.load_state_dict(state["model"])
    loader = build_loader(cfg, manifest, shuffle=False)
    pred = gather_predictions(model, loader, dev, cfg.model.head.n_intervals)
    report = build_report(pred, cfg.evaluation, seed=cfg.train.seed)
    point, lo, hi = report.auroc
    cpt, clo, chi = report.cindex
    click.echo(f"AUROC      {point:.3f} ({lo:.3f}-{hi:.3f})")
    click.echo(f"C-index    {cpt:.3f} ({clo:.3f}-{chi:.3f})")
    click.echo(f"cal slope  {report.calibration_slope:.3f}")
    click.echo(f"ICI        {report.integrated_calibration_index:.3f}")
    click.echo(f"IBS        {report.integrated_brier:.3f}")
    click.echo(f"D-cal p    {report.d_calibration_p:.3f}")
    click.echo(f"selective  {report.selective_auroc:.3f}")
