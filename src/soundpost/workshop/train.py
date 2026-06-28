from __future__ import annotations

import click

from soundpost.bench.loop import Trainer
from soundpost.jigs.logs import setup_logging
from soundpost.jigs.seed import set_seed
from soundpost.workshop._common import build_loader, build_model, load_config, resolve_device


@click.command(name="train")
@click.option("--config", required=True, help="path to an experiment yaml")
@click.option("--manifest", default=None, help="jsonl cohort manifest (omit to use synthetic)")
@click.option("--override", multiple=True, help="dotted key=value config override")
@click.option("--device", default="auto")
def train_command(
    config: str, manifest: str | None, override: tuple[str, ...], device: str
) -> None:
    setup_logging()
    cfg = load_config(config, override)
    set_seed(cfg.train.seed)
    dev = resolve_device(device)
    model = build_model(cfg)
    loader = build_loader(cfg, manifest, shuffle=True)
    trainer = Trainer(model, cfg, dev)
    trainer.fit(loader)
    trainer.save(f"{cfg.train.out_dir}/final.ckpt", step=-1)
    click.echo(f"wrote {cfg.train.out_dir}/final.ckpt")
