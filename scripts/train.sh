#!/usr/bin/env bash
# Train one model on fastMRI single-coil knee data on a GPU.
#
#   GPU=0 MODEL=fourier-pdunet MASK=varden1d ACCEL=4 scripts/train.sh /data/fastmri
#
# Anything after the data root is passed straight to `mrirecon train`,
# e.g. `scripts/train.sh /data/fastmri --magnitude --epochs 50`.
set -euo pipefail

DATA_ROOT=${1:?usage: train.sh DATA_ROOT [extra mrirecon args]}
shift

GPU=${GPU:-0}
MODEL=${MODEL:-fourier-pdunet}
MASK=${MASK:-varden1d}
ACCEL=${ACCEL:-4}
EPOCHS=${EPOCHS:-100}
WORKERS=${WORKERS:-8}
RUN_NAME=${RUN_NAME:-${MODEL}_${MASK}_x${ACCEL}}
OUT=${OUT:-runs/${RUN_NAME}}

mkdir -p "$OUT"
echo "[$(date '+%F %T')] training $RUN_NAME on GPU $GPU -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "$GPU" 2>/dev/null || echo "nvidia-smi not found, carrying on"

CUDA_VISIBLE_DEVICES=$GPU mrirecon train \
    --model "$MODEL" \
    --train-dir "$DATA_ROOT/singlecoil_train" \
    --mask "$MASK" --accel "$ACCEL" \
    --epochs "$EPOCHS" --workers "$WORKERS" \
    --out "$OUT" "$@" 2>&1 | tee -a "$OUT/train.log"
