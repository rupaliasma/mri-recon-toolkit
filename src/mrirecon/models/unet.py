"""Small U-Net used as the primal (image-space) block of Fourier-PDUNet.

Follows the layout of the U-Net in NCC1701 (itself adapted from
https://discuss.pytorch.org/t/unet-implementation/426): average-pool
downsampling, bilinear upsampling followed by a 1x1 conv, ReLU then BatchNorm.
Parameter names match NCC1701 for the real-valued variant so that old
checkpoints load as-is.

With ``complex=True`` every convolution is complex-valued and the activation
is CReLU (ReLU applied separately to the real and imaginary parts, Trabelsi
et al. 2018).
"""

import torch
import torch.nn.functional as F
from torch import nn


def crelu(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return torch.complex(F.relu(x.real), F.relu(x.imag))
    return F.relu(x)


class CReLU(nn.Module):
    def forward(self, x):
        return crelu(x)


def _per_part(fn, x, *args, **kwargs):
    """Apply a real-only linear op to the real and imaginary parts separately."""
    if x.is_complex():
        return torch.complex(fn(x.real, *args, **kwargs), fn(x.imag, *args, **kwargs))
    return fn(x, *args, **kwargs)


class ComplexBatchNorm2d(nn.Module):
    # ponytail: normalises real and imaginary parts independently ("naive" complex BN).
    # The 2x2 whitening of Trabelsi et al. is more faithful; swap in if it matters.
    def __init__(self, num_features: int):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return torch.complex(self.bn_r(x.real), self.bn_i(x.imag))


class Upsample(nn.Module):
    def forward(self, x):
        return _per_part(F.interpolate, x, scale_factor=2, mode="bilinear", align_corners=False)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, complex=False):
        super().__init__()
        dtype = torch.cfloat if complex else None
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1, dtype=dtype),
            CReLU() if complex else nn.ReLU(),
            ComplexBatchNorm2d(out_size) if complex else nn.BatchNorm2d(out_size),
            nn.Conv2d(out_size, out_size, 3, padding=1, dtype=dtype),
            CReLU() if complex else nn.ReLU(),
            ComplexBatchNorm2d(out_size) if complex else nn.BatchNorm2d(out_size),
        )

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, complex=False):
        super().__init__()
        dtype = torch.cfloat if complex else None
        self.up = nn.Sequential(Upsample(), nn.Conv2d(in_size, out_size, 1, dtype=dtype))
        self.conv_block = UNetConvBlock(in_size, out_size, complex)

    @staticmethod
    def embed(x, size):
        # Odd input sizes leave the upsampled map a pixel short; pad it back.
        dh, dw = size[0] - x.shape[-2], size[1] - x.shape[-1]
        x = F.pad(x, (dw // 2, dw - dw // 2, 0, 0), mode="circular") if dw else x
        return F.pad(x, (0, 0, dh // 2, dh - dh // 2)) if dh else x

    def forward(self, x, bridge):
        up = self.embed(self.up(x), bridge.shape[-2:])
        return self.conv_block(torch.cat([up, bridge], 1))


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, wf=6, complex=False):
        super().__init__()
        self.down_path = nn.ModuleList()
        prev = in_channels
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev, 2 ** (wf + i), complex))
            prev = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev, 2 ** (wf + i), complex))
            prev = 2 ** (wf + i)

        self.last = nn.Conv2d(prev, out_channels, 1, dtype=torch.cfloat if complex else None)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < len(self.down_path) - 1:
                skips.append(x)
                x = _per_part(F.avg_pool2d, x, 2)
        for up, skip in zip(self.up_path, reversed(skips)):
            x = up(x, skip)
        return self.last(x)
