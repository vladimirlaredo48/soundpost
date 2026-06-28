from __future__ import annotations

import click

from soundpost import __version__
from soundpost.workshop.evaluate import evaluate_command
from soundpost.workshop.export import export_command
from soundpost.workshop.infer import infer_command
from soundpost.workshop.train import train_command


@click.group()
@click.version_option(__version__)
def cli() -> None:
    pass


cli.add_command(train_command)
cli.add_command(evaluate_command)
cli.add_command(infer_command)
cli.add_command(export_command)


if __name__ == "__main__":
    cli()
