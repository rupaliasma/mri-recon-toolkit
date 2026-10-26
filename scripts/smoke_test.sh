#!/usr/bin/env bash
# End-to-end smoke test on synthetic data: writes a few tiny fastMRI-style volumes, trains
# each model for one epoch on 32x32 crops and evaluates it. Checks that the code runs,
# not that it reconstructs well. Takes well under a minute on a laptop CPU.
#
#   scripts/smoke_test.sh [WORK_DIR]
set -euo pipefail

WORK=${1:-$(mktemp -d)}
mkdir -p "$WORK/singlecoil_train" "$WORK/singlecoil_val"

python - "$WORK" <<'PY'
import sys
import h5py
import numpy as np

root = sys.argv[1]
rng = np.random.default_rng(0)
for split, n in (("singlecoil_train", 4), ("singlecoil_val", 2)):
    for v in range(n):
        y, x = np.mgrid[-1:1:48j, -1:1:40j]
        images = np.stack([(((y - rng.uniform(-.3, .3)) / .5) ** 2 + (x / .4) ** 2 < 1) * np.exp(1j * x)
                           for _ in range(3)]).astype(np.complex64)
        kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(images, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
        with h5py.File(f"{root}/{split}/vol{v}.h5", "w") as f:
            f.create_dataset("kspace", data=kspace)
PY

for model in fourier-pd fourier-pdunet; do
    mrirecon train --model "$model" --train-dir "$WORK/singlecoil_train" --val-fraction 0.25 \
        --epochs 1 --crop 32 --workers 0 --device cpu --out "$WORK/runs/$model"
    mrirecon evaluate --checkpoint "$WORK/runs/$model/best.pt" --data-dir "$WORK/singlecoil_val" \
        --workers 0 --device cpu --out "$WORK/results/$model"
done
python "$(dirname "$0")/collate_results.py" "$WORK/results"
echo "smoke test passed ($WORK)"
