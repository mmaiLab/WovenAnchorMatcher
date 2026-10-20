#!/usr/bin/env bash
set -euo pipefail

eval_dataset="${1:-}"
case "${eval_dataset}" in
  megadepth)
    data_config="configs/data/megadepth_test_1500.py"
    model_config="configs/woven_anchor/outdoor/eval.py"
    ;;
  scannet)
    data_config="configs/data/scannet_test_1500.py"
    model_config="configs/woven_anchor/indoor/test.py"
    ;;
  hpatches)
    data_config="configs/data/hpatches_test.py"
    model_config="configs/woven_anchor/outdoor/eval.py"
    ;;
  *)
    echo "Unknown dataset: ${eval_dataset}" >&2
    echo "Usage: bash scripts/reproduce_test/outdoor.sh <megadepth|scannet|hpatches> [additional arguments]" >&2
    exit 2
    ;;
esac
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

exec python -u test.py \
  "${data_config}" \
  "${model_config}" \
  "$@"
