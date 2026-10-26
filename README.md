# mrirecon

PyTorch re-implementation of **Fourier-PD** and **Fourier-PDUNet**, two complex-valued
primal-dual networks for reconstructing undersampled MRI, packaged so that they can be
trained and evaluated on public fastMRI k-space instead of patient data.

The models come from the [NCC1701](https://github.com/soumickmj/NCC1701) pipeline and
were published in

> R. Khatun, S. Chatterjee, C. Bert, M. Wadepohl, O. J. Ott, S. Semrau, R. Fietkau,
> A. Nürnberger, U. S. Gaipl, B. Frey. *Complex-valued neural networks to speed-up MR
> thermometry during hyperthermia using Fourier PD and PDUNet.* Scientific Reports 15,
> 11765 (2025). https://doi.org/10.1038/s41598-025-96071-x

The original experiments used clinical MR thermometry data that cannot be shared. This
package keeps the architectures, loss and training schedule of the paper but swaps the
data for fastMRI, so anyone can reproduce the numbers.

## What is in here

| | |
|---|---|
| `mrirecon.models` | Fourier-PD and Fourier-PDUNet, complex or magnitude-only primal |
| `mrirecon.masks` | random / equispaced (fastMRI style), 1D and 2D variable-density (Varden) Cartesian masks |
| `mrirecon.data` | fastMRI slice dataset, single-coil and multi-coil (SVD coil compression) |
| `mrirecon.metrics` | SSIM (matches scikit-image / fastMRI), PSNR, NMSE, complex NRMSE |
| `mrirecon` CLI | `train` and `evaluate` |
| `scripts/` | Bash wrappers for GPU training, evaluation and the full benchmark |

The port was checked against the original code: with identical weights the outputs agree
to within float32 rounding (see `tests/test_ncc1701_parity.py`), and NCC1701 checkpoints
load through `convert_ncc1701_state_dict`. Details in [docs/models.md](docs/models.md).

## Installation

```bash
pip install "git+https://github.com/soumickmj/ProjRup.git#subdirectory=mri-recon-toolkit"
# or, from a clone
pip install -e "mri-recon-toolkit[dev]"
```

Python 3.10 or newer and PyTorch 2.1 or newer. Install the CUDA build of PyTorch first if
you want to train on a GPU.

A Docker image is built for every release:

```bash
docker pull ghcr.io/soumickmj/mrirecon:latest
docker run --gpus all -v /path/to/fastmri:/data -v $PWD/runs:/runs ghcr.io/soumickmj/mrirecon \
    train --train-dir /data/singlecoil_train --out /runs/pdunet
```

## Quick start

```python
import torch
from mrirecon import build_model, data_consistency, fft2c, ifft2c
from mrirecon.masks import make_mask

target = torch.randn(1, 1, 320, 320, dtype=torch.cfloat)  # your complex image
mask = torch.from_numpy(make_mask("varden2d", (320, 320), accel=10))
kspace = fft2c(target) * mask
zero_filled = ifft2c(kspace)

model = build_model("fourier-pdunet").eval()  # complex primal by default
with torch.no_grad():
    recon = data_consistency(model(zero_filled, kspace), kspace, mask)
```

## Data

Request access to fastMRI at <https://fastmri.med.nyu.edu/> and download the single-coil
knee data. The scripts expect

```
/data/fastmri/
    singlecoil_train/*.h5
    singlecoil_val/*.h5
```

The official test split has no fully sampled reference, so the protocol here is:
train on `singlecoil_train` with 10% of its volumes held out for validation (split by
volume, fixed seed), and report results on `singlecoil_val`. Multi-coil files (brain or
knee) also work: each slice is compressed to a single virtual coil first.

Every slice is cropped to 320 x 320 in image space, transformed back to k-space and then
undersampled, so the target and the measured data are consistent. Masks are drawn afresh
at every training step; for validation and testing they are fixed per slice (seeded by
file name and slice index), so every model sees exactly the same measurements.

## Smoke test

```bash
scripts/smoke_test.sh
```

Writes a few tiny synthetic fastMRI-style volumes, trains each model for one epoch on
32 x 32 crops on the CPU and evaluates it. It only checks that data loading, training,
checkpointing and evaluation run end to end (in well under a minute); with one epoch on
toy data the scores stay at the zero-filled level, because the primal blocks start as
an identity map.

## Training and evaluation

```bash
# one model, one setting
GPU=0 MODEL=fourier-pdunet MASK=varden1d ACCEL=4 scripts/train.sh /data/fastmri
GPU=0 scripts/evaluate.sh runs/fourier-pdunet_varden1d_x4/best.pt /data/fastmri

# both models x the three sampling settings of the paper, then a Markdown table
GPU=0 scripts/benchmark.sh /data/fastmri
```

or directly:

```bash
mrirecon train --model fourier-pd --train-dir /data/fastmri/singlecoil_train \
    --mask varden2d --accel 10 --out runs/pd_varden2d_x10
mrirecon evaluate --checkpoint runs/pd_varden2d_x10/best.pt \
    --data-dir /data/fastmri/singlecoil_val --out results/pd_varden2d_x10
```

Defaults follow the paper: L1 loss on the real and imaginary parts, Adam with a learning
rate of 1e-4 divided by 10 every 50 epochs, 100 epochs, batch size 1. Data consistency
is applied at inference only. `--magnitude` trains the magnitude-only variants instead
(the image-space blocks then never see phase).

`evaluate` writes `per_slice.csv` and `summary.json` with SSIM, PSNR, NMSE and complex
NRMSE for the model and for zero-filling. Metrics are computed per slice on magnitude
images with the data range taken from each target slice; NRMSE is computed on complex
values and so also reflects phase errors.

## Results

Run `scripts/benchmark.sh` to fill this table; it writes the same table to
`results/benchmark.md`. Sampling settings are those of the paper: 1D Varden 25%, 2D
Varden 25% and 2D Varden 10% (theoretical acceleration 4, 4 and 10).

| Sampling | Method | SSIM | PSNR (dB) | NMSE |
|---|---|---|---|---|
| varden1d x4 | zero-filled | | | |
| varden1d x4 | Fourier-PD | | | |
| varden1d x4 | Fourier-PDUNet | | | |
| varden2d x4 | zero-filled | | | |
| varden2d x4 | Fourier-PD | | | |
| varden2d x4 | Fourier-PDUNet | | | |
| varden2d x10 | zero-filled | | | |
| varden2d x10 | Fourier-PD | | | |
| varden2d x10 | Fourier-PDUNet | | | |

For reference, on the clinical thermometry data the paper reports magnitude SSIM of
0.886 (Fourier-PD) and 0.864 (Fourier-PDUNet) for 2D Varden 10%, against 0.385 for
zero-filling. Those numbers are not directly comparable with fastMRI knee scans.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

The tests run on CPU in under a minute and cover the FFT helpers, masks, metrics, the
dataset (on small synthetic fastMRI-style files), both models, an end-to-end
train/evaluate run and a numerical regression snapshot (`tests/data/regression.json`).
To compare against the original implementation, clone NCC1701, install
[torchcomplex](https://github.com/soumickmj/pytorch-complex) and run

```bash
NCC1701_PATH=/path/to/NCC1701 pytest tests/test_ncc1701_parity.py
```

CI (`.github/workflows/mrirecon.yml` at the repository root) lints and tests every push
and pull request, builds the wheel, and on a `mrirecon-v*` tag publishes the Docker image
to GHCR.

## Citation

```bibtex
@article{khatun2025complex,
  title   = {Complex-valued neural networks to speed-up {MR} thermometry during hyperthermia using {Fourier PD} and {PDUNet}},
  author  = {Khatun, Rupali and Chatterjee, Soumick and Bert, Christoph and Wadepohl, Martin and Ott, Oliver J. and Semrau, Sabine and Fietkau, Rainer and N{\"u}rnberger, Andreas and Gaipl, Udo S. and Frey, Benjamin},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {11765},
  year    = {2025},
  doi     = {10.1038/s41598-025-96071-x}
}
```

If you use fastMRI, please also cite Zbontar et al., *fastMRI: An Open Dataset and
Benchmarks for Accelerated MRI*, arXiv:1811.08839 (2018).
