#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

exec python -u train.py \
  configs/data/megadepth_trainval_832.py \
  configs/woven_anchor/outdoor/train.py \
  --batch_size=8 \
  --num_workers=8 \
  --log_every_n_steps=1000 \
  "$@"
