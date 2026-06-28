from __future__ import annotations

import json

import click
import torch

from soundpost.bench.checkpoint import load_checkpoint
from soundpost.bench.loop import move_batch
from soundpost.jigs.logs import setup_logging
from soundpost.workshop._common import build_loader, build_model, load_config, resolve_device


@click.command(name="infer")
@click.option("--config", required=True)
@click.option("--checkpoint", required=True)
@click.option("--manifest", default=None)
@click.option("--output", default="predictions.jsonl")
@click.option("--override", multiple=True)
@click.option("--device", default="auto")
def infer_command(
    config: str,
    checkpoint: str,
    manifest: str | None,
    output: str,
    override: tuple[str, ...],
    device: str,
) -> None:
    setup_logging()
    cfg = load_config(config, override)
    dev = resolve_device(device)
    model = build_model(cfg)
    state = load_checkpoint(checkpoint, map_location=str(dev))
    model.load_state_dict(state["model"])
    model.eval().to(dev)
    loader = build_loader(cfg, manifest, shuffle=False)
    rows = []
    with torch.no_grad():
        for raw in loader:
            out = model(move_batch(raw, dev))
            for r, surv, token in zip(out.risk, out.survival, out.remnant_state):
                rows.append(
                    {
                        "risk_24mo": float(r),
                        "survival_curve": [float(x) for x in surv.cpu()],
                        "remnant_state_norm": float(token.norm().cpu()),
                    }
                )
    with open(output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    click.echo(f"wrote {len(rows)} predictions to {output}")
