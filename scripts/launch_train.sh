#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiment/main.yaml}
MANIFEST=${2:-data/eob_recur/train.jsonl}
NPROC=${NPROC:-8}

torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  -m soundpost.workshop.train \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --device cuda
