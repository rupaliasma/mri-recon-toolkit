# Models

Both networks unroll the learned primal-dual algorithm of Adler and Öktem (IEEE TMI,
2018) with the centred, orthonormal 2D Fourier transform as forward operator. Each of
the `n_iterations` steps does

```
h  <- h + w_d * Dual([h, mean(F f) / s, g])          # k-space, complex convs
f  <- f + w_p * Primal([f, mean(F^-1 (s h))])        # image space
```

where `g` is the measured k-space divided by its largest magnitude `s`, `h` holds
`n_dual` k-space channels, `f` holds `n_primal` image channels and the output is the
channel mean of `f`. `w_d` and `w_p` are learnt per-channel weights; `w_p` starts at
(almost) zero so that every primal block is initially the identity.

| | Fourier-PD | Fourier-PDUNet |
|---|---|---|
| iterations | 10 | 10 |
| primal / dual channels | 5 / 5 | 5 / 5 |
| dual block | 3 complex 3x3 convs, 32 features, CReLU | same |
| primal block | 3 convs (3x3, 32 features) | U-Net, depth 4, 32 to 256 features |
| primal initialisation | zeros | zero-filled image |
| parameters (complex primal) | 0.25 M | 18.2 M |

Parameter counts treat a complex weight as one parameter.

The dual convolutions pad circularly along the phase-encode axis and by replication
along the readout axis, a habit inherited from the sinogram version of the network;
the primal convolutions replicate on all sides.

## Complex or magnitude primal

`complex_primal=True` (the default, and the setting of the thermometry paper) keeps the
image estimate complex, so the output has magnitude and phase. With
`complex_primal=False` the primal blocks are real-valued and only ever see magnitudes;
these are the variants NCC1701 calls `PDOrigMagPrimCReLU` and
`PDUNetMagPrimCReLU10IT`, used in its fastMRI experiments.

## Differences from NCC1701

The code follows `Bridge/WarpDrives/PDUNet/pd_crelu.py` closely and keeps its parameter
names. Things that differ:

* **No torchcomplex.** Complex convolutions use PyTorch's native complex tensors.
  torchcomplex adds a convolution's complex bias `b` to each of its four real
  sub-convolutions, which works out as an effective bias of `b(1+i)`. The two are the
  same family of functions, but loading an NCC1701 checkpoint needs the biases rescaled;
  `convert_ncc1701_state_dict` does that.
* **Per-sample k-space scaling.** NCC1701 divides by the largest magnitude in the whole
  batch, which couples the samples of a batch. Here every sample is scaled on its own.
  At batch size 1, the setting of the paper, the two are identical.
* **Complex U-Net batch norm** (only for Fourier-PDUNet with a complex primal)
  normalises the real and imaginary parts separately instead of using the 2x2 whitening
  of Trabelsi et al. Checkpoints of that one variant therefore do not transfer.
* Radon-transform support, the non-residual variants and the large-kernel experiment
  were left out.

`tests/test_ncc1701_parity.py` loads randomly perturbed NCC1701 weights into this
implementation and checks that the outputs match for Fourier-PD (complex and magnitude)
and magnitude Fourier-PDUNet.

## Undersampling masks

`varden1d` and `varden2d` reproduce the sampling patterns described in the paper: a
fully sampled centre (8 lines for 1D, a disc covering 2.5% of k-space for 2D) and the
rest drawn without replacement from a Gaussian density centred on k = 0 until the target
fraction `1/accel` is reached. The width of the Gaussian (a quarter of the matrix size)
is a reasonable choice, not a value taken from the paper. `random` and `equispaced`
follow the fastMRI mask functions for comparison with the fastMRI literature.
