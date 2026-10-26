#!/usr/bin/env bash
# Evaluate a checkpoint on the fastMRI single-coil validation split, which we use as test set
# (the official test split has no fully sampled ground truth).
#
#   GPU=0 scripts/evaluate.sh runs/fourier-pdunet_varden1d_x4/best.pt /data/fastmri
set -euo pipefail

CKPT=${1:?usage: evaluate.sh CHECKPOINT DATA_ROOT [extra mrirecon args]}
DATA_ROOT=${2:?usage: evaluate.sh CHECKPOINT DATA_ROOT [extra mrirecon args]}
shift 2

GPU=${GPU:-0}
RUN_DIR=$(dirname "$CKPT")
OUT=${OUT:-results/$(basename "$RUN_DIR")}

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU mrirecon evaluate \
    --checkpoint "$CKPT" \
    --data-dir "$DATA_ROOT/singlecoil_val" \
    --label "$(basename "$RUN_DIR")" \
    --out "$OUT" "$@" 2>&1 | tee "$OUT/evaluate.log"
