"""Fourier Primal-Dual network (Fourier-PD) and its U-Net variant (Fourier-PDUNet).

The learned primal-dual scheme of Adler & Oktem (2018) unrolls a primal-dual
hybrid gradient method and replaces both proximal operators with small CNNs.
Here the forward operator is the 2D Fourier transform, so the dual blocks
operate on complex k-space with complex-valued convolutions and the primal
blocks operate in image space. Fourier-PDUNet swaps the three-layer primal CNN
for a U-Net. See Khatun, Chatterjee et al., Sci Rep 15, 11765 (2025),
https://doi.org/10.1038/s41598-025-96071-x.

The module layout mirrors ``Bridge/WarpDrives/PDUNet/pd_crelu.py`` in NCC1701,
so state dicts trained there can be loaded directly (see ``docs/models.md``
for the exceptions).
"""

import torch
import torch.nn.functional as F
from torch import nn

from mrirecon.fft import fft2c, ifft2c
from mrirecon.models.unet import CReLU, UNet


def _conv(in_ch, out_ch, complex):
    conv = nn.Conv2d(in_ch, out_ch, 3, dtype=torch.cfloat if complex else None)
    with torch.no_grad():
        conv.bias.zero_()
    return conv


class PrePadDualConv2D(nn.Module):
    """3x3 conv on k-space: circular padding along the phase-encode axis, replicate along readout."""

    def __init__(self, in_ch, out_ch, complex=True):
        super().__init__()
        self.conv = _conv(in_ch, out_ch, complex)

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        return self.conv(F.pad(x, (0, 0, 1, 1), mode="replicate"))


class PrePadPrimalConv2D(nn.Module):
    """3x3 conv on the image with replicate padding on all sides."""

    def __init__(self, in_ch, out_ch, complex=False):
        super().__init__()
        self.conv = _conv(in_ch, out_ch, complex)

    def forward(self, x):
        return self.conv(F.pad(x, (1, 1, 1, 1), mode="replicate"))


def _activation(complex, ch):
    return CReLU() if complex else nn.PReLU(ch)


class DualBlock(nn.Module):
    def __init__(self, features, complex=True):
        super().__init__()
        self.layers = nn.Sequential(
            PrePadDualConv2D(features + 2, 32, complex),
            _activation(complex, 32),
            PrePadDualConv2D(32, 32, complex),
            _activation(complex, 32),
            PrePadDualConv2D(32, features, complex),
        )
        dtype = torch.cfloat if complex else torch.float32
        self.diff_weight = nn.Parameter(torch.ones(1, features, 1, 1, dtype=dtype))

    def forward(self, h, f_ksp, g):
        x = torch.cat([h, f_ksp.mean(dim=1, keepdim=True), g], 1)
        return h + self.diff_weight * self.layers(x)


def _primal_diff_weight(features, complex):
    # Starts (almost) at zero so every primal block is initially an identity map.
    if complex:
        return nn.Parameter(torch.zeros(1, features, 1, 1, dtype=torch.cfloat) + torch.finfo(torch.float32).eps)
    return nn.Parameter(torch.zeros(1, features, 1, 1))


class PrimalBlock(nn.Module):
    def __init__(self, features, complex=False):
        super().__init__()
        self.layers = nn.Sequential(
            PrePadPrimalConv2D(features + 1, 32, complex),
            _activation(complex, 32),
            PrePadPrimalConv2D(32, 32, complex),
            _activation(complex, 32),
            PrePadPrimalConv2D(32, features, complex),
        )
        self.diff_weight = _primal_diff_weight(features, complex)

    def forward(self, h_img, f):
        x = torch.cat([f, h_img.mean(dim=1, keepdim=True)], 1)
        return f + self.diff_weight * self.layers(x)


class PrimalUNetBlock(nn.Module):
    def __init__(self, features, complex=False):
        super().__init__()
        self.layers = UNet(features + 1, features, wf=5, complex=complex)
        self.diff_weight = _primal_diff_weight(features, complex)

    def forward(self, h_img, f):
        x = torch.cat([f, h_img.mean(dim=1, keepdim=True)], 1)
        return f + self.diff_weight * self.layers(x)


class PrimalDualNetwork(nn.Module):
    """Unrolled primal-dual reconstruction with the Fourier transform as forward operator.

    Args:
        n_primal: number of image-space channels carried between iterations.
        n_dual: number of k-space channels carried between iterations.
        n_iterations: number of unrolled primal/dual updates.
        unet_primal: use a U-Net as primal block (Fourier-PDUNet) instead of
            the three-layer CNN (Fourier-PD).
        complex_primal: keep the image estimate complex (magnitude and phase).
            With ``False`` the primal blocks see only the magnitude, as in the
            magnitude-only fastMRI experiments.
        init_from_input: initialise the primal channels with the zero-filled
            image rather than with zeros. NCC1701 uses zeros for Fourier-PD and
            the input for Fourier-PDUNet.

    ``forward(image, kspace)`` takes the zero-filled image and the undersampled
    k-space, both shaped (B, 1, H, W), and returns a (B, 1, H, W) image. The
    k-space is scaled per sample by its maximum magnitude before entering the
    network and the scaling is undone on the way out.
    """

    def __init__(
        self,
        n_primal: int = 5,
        n_dual: int = 5,
        n_iterations: int = 10,
        unet_primal: bool = False,
        complex_primal: bool = True,
        init_from_input: bool = False,
    ):
        super().__init__()
        primal_cls = PrimalUNetBlock if unet_primal else PrimalBlock
        self.primal_blocks = nn.ModuleList(primal_cls(n_primal, complex_primal) for _ in range(n_iterations))
        self.dual_blocks = nn.ModuleList(DualBlock(n_dual, complex=True) for _ in range(n_iterations))
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.complex_primal = complex_primal
        self.init_from_input = init_from_input

    def forward(self, image: torch.Tensor, kspace: torch.Tensor | None = None) -> torch.Tensor:
        if kspace is None:
            kspace = fft2c(image.to(torch.cfloat))
        if self.complex_primal:
            image = image.to(torch.cfloat)
        elif image.is_complex():
            image = image.abs()

        scale = kspace.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        g = kspace / scale

        B, _, H, W = g.shape
        h = torch.zeros(B, self.n_dual, H, W, dtype=g.dtype, device=g.device)
        if self.init_from_input:
            f = image.repeat(1, self.n_primal, 1, 1)
        else:
            f = torch.zeros(B, self.n_primal, H, W, dtype=image.dtype, device=g.device)

        for primal, dual in zip(self.primal_blocks, self.dual_blocks):
            h = dual(h, fft2c(f) / scale, g)
            h_img = ifft2c(h * scale)
            if not self.complex_primal:
                h_img = h_img.abs()
            f = primal(h_img, f)

        return f.mean(dim=1, keepdim=True)


def fourier_pd(**overrides) -> PrimalDualNetwork:
    """Fourier-PD as published: 5 primal / 5 dual channels, 10 iterations."""
    kwargs = dict(n_primal=5, n_dual=5, n_iterations=10, unet_primal=False, init_from_input=False)
    return PrimalDualNetwork(**{**kwargs, **overrides})


def fourier_pdunet(**overrides) -> PrimalDualNetwork:
    """Fourier-PDUNet as published: 5 primal / 5 dual channels, 10 iterations, U-Net primal."""
    kwargs = dict(n_primal=5, n_dual=5, n_iterations=10, unet_primal=True, init_from_input=True)
    return PrimalDualNetwork(**{**kwargs, **overrides})


def convert_ncc1701_state_dict(state_dict: dict) -> dict:
    """Map a state dict saved by NCC1701 onto this implementation.

    torchcomplex adds the complex bias ``b`` of a convolution once to each of
    the four real convolutions, so its effective bias is ``b * (1 + 1j)``. Plain
    PyTorch complex convolutions add ``b`` as is, hence the rescaling.
    """
    out = {}
    for key, value in state_dict.items():
        if key.endswith("bias") and value.is_complex():
            value = value * (1 + 1j)
        out[key] = value
    return out
