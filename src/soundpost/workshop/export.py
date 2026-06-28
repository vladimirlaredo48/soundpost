from __future__ import annotations

import inspect

import click
import torch
from torch import Tensor, nn

from soundpost.archings.instrument import Earlst
from soundpost.bench.checkpoint import load_checkpoint
from soundpost.jigs.logs import setup_logging
from soundpost.workshop._common import build_model, load_config, resolve_device


class _HeadGraph(nn.Module):
    def __init__(self, model: Earlst) -> None:
        super().__init__()
        self.fusion = model.fusion
        self.head = model.head
        self.film = model.film

    def forward(self, state: Tensor, clinical: Tensor) -> Tensor:
        c = self.fusion(state, clinical)
        phi = self.head.penultimate(state)
        modulated = self.film(c, phi)
        return self.head.hazard_from_features(modulated)


@click.command(name="export")
@click.option("--config", required=True)
@click.option("--checkpoint", default=None)
@click.option("--output", default="earlst_head.onnx")
@click.option("--override", multiple=True)
@click.option("--device", default="cpu")
def export_command(
    config: str, checkpoint: str | None, output: str, override: tuple[str, ...], device: str
) -> None:
    setup_logging()
    cfg = load_config(config, override)
    dev = resolve_device(device)
    model = build_model(cfg)
    if checkpoint is not None:
        model.load_state_dict(load_checkpoint(checkpoint, map_location=str(dev))["model"])
    graph = _HeadGraph(model).eval().to(dev)
    d_model = cfg.model.encoder.d_model
    state = torch.randn(1, d_model, device=dev)
    clinical = torch.randn(1, d_model, device=dev)
    axes = {"remnant_state": {0: "batch"}, "clinical_embedding": {0: "batch"}}
    names_in = ["remnant_state", "clinical_embedding"]
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        torch.onnx.export(
            graph,
            (state, clinical),
            output,
            input_names=names_in,
            output_names=["hazard"],
            opset_version=14,
            dynamic_axes=axes,
            dynamo=False,
        )
    else:
        torch.onnx.export(
            graph,
            (state, clinical),
            output,
            input_names=names_in,
            output_names=["hazard"],
            opset_version=14,
            dynamic_axes=axes,
        )
    click.echo(f"exported survival-head graph to {output}")
