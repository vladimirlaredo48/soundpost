#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiment/main.yaml}
CKPT=${2:-runs/earlst/final.ckpt}
MANIFEST=${3:-data/eob_recur/prospective.jsonl}

python -m soundpost.workshop.evaluate \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --manifest "${MANIFEST}" \
  --device cuda
