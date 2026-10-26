#!/usr/bin/env bash
# Full benchmark: both models x the three sampling settings of the paper, then a summary table.
# Takes several GPU-days for 100 epochs; set EPOCHS lower for a quick look.
#
#   GPU=0 EPOCHS=100 scripts/benchmark.sh /data/fastmri
set -euo pipefail

DATA_ROOT=${1:?usage: benchmark.sh DATA_ROOT}
HERE=$(cd "$(dirname "$0")" && pwd)
SETTINGS=("varden1d 4" "varden2d 4" "varden2d 10")

for model in fourier-pd fourier-pdunet; do
    for setting in "${SETTINGS[@]}"; do
        read -r mask accel <<< "$setting"
        run=${model}_${mask}_x${accel}
        if [[ -f runs/$run/best.pt && ${RETRAIN:-0} != 1 ]]; then
            echo "skipping training of $run (checkpoint exists, RETRAIN=1 to redo)"
        else
            MODEL=$model MASK=$mask ACCEL=$accel RUN_NAME=$run "$HERE/train.sh" "$DATA_ROOT"
        fi
        "$HERE/evaluate.sh" "runs/$run/best.pt" "$DATA_ROOT"
    done
done

python "$HERE/collate_results.py" results > results/benchmark.md
cat results/benchmark.md
