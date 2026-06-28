from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from soundpost.settings import ExperimentConfig


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(token: str) -> Any:
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def merge_overrides(payload: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    out = dict(payload)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        cursor = out
        keys = dotted.split(".")
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = _coerce(raw)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_experiment(
    path: str | Path,
    config_root: str | Path | None = None,
    overrides: Iterable[str] | None = None,
) -> ExperimentConfig:
    path = Path(path)
    payload = _read_yaml(path)
    root = Path(config_root) if config_root is not None else path.parent.parent
    composed: dict[str, Any] = {}
    for group in ("model", "data", "train", "optim", "loss", "pretrain", "evaluation"):
        ref = payload.pop(f"{group}_ref", None)
        if ref is not None:
            composed[group] = _read_yaml(root / group / f"{ref}.yaml")
    composed = _deep_merge(composed, payload)
    if overrides:
        composed = merge_overrides(composed, overrides)
    return ExperimentConfig.model_validate(composed)
