"""Centred, orthonormal 2D Fourier transforms over the last two axes."""

import torch

DIMS = (-2, -1)


def fft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=DIMS)
    x = torch.fft.fft2(x, dim=DIMS, norm="ortho")
    return torch.fft.fftshift(x, dim=DIMS)


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    k = torch.fft.ifftshift(k, dim=DIMS)
    k = torch.fft.ifft2(k, dim=DIMS, norm="ortho")
    return torch.fft.fftshift(k, dim=DIMS)


def data_consistency(image: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace the sampled k-space locations of ``image`` with the measured values.

    ``mask`` is 1 where k-space was acquired and must broadcast against ``kspace``.
    A real-valued ``image`` is treated as having zero phase.
    """
    k_est = fft2c(image.to(kspace.dtype))
    return ifft2c(k_est * (1 - mask) + kspace * mask)
