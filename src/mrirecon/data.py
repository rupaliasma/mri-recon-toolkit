"""fastMRI slice dataset with retrospective Cartesian undersampling.

Works with both the single-coil knee files (``kspace`` shaped (S, H, W)) and
the multi-coil files (``kspace`` shaped (S, C, H, W)). Multi-coil k-space is
reduced to a single virtual coil by SVD coil compression (Buehrer et al.
2007), which keeps a complex image with meaningful phase; this matches the
single-channel setting the models were designed for.

Each slice is cropped to ``crop`` x ``crop`` in image space, transformed back
to k-space and undersampled, so that the fully sampled target and the
measured data are consistent. Everything is scaled by the maximum magnitude of
the zero-filled image.
"""

import zlib
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mrirecon.fft import fft2c, ifft2c
from mrirecon.masks import make_mask


def compress_coils(kspace: np.ndarray) -> np.ndarray:
    """Project (C, H, W) k-space onto its dominant virtual coil, returning (H, W)."""
    c = kspace.shape[0]
    m = kspace.reshape(c, -1)
    _, vecs = np.linalg.eigh(m @ m.conj().T)
    return (vecs[:, -1].conj() @ m).reshape(kspace.shape[1:])


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    ch, cw = min(size, h), min(size, w)
    top, left = (h - ch) // 2, (w - cw) // 2
    return x[..., top : top + ch, left : left + cw]


def split_volumes(files, val_fraction: float, seed: int = 0):
    """Deterministic volume-level train/validation split."""
    files = sorted(files)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    n_val = int(round(len(files) * val_fraction))
    val = {files[i] for i in order[:n_val]}
    return [f for f in files if f not in val], [f for f in files if f in val]


class FastMRISlices(Dataset):
    """Yields dicts with ``image``, ``kspace``, ``mask``, ``target`` (all (1, H, W)) and metadata.

    With ``fixed_mask=True`` the mask of a slice depends only on its file name
    and index, so evaluation is reproducible; during training a fresh mask is
    drawn every time.
    """

    def __init__(
        self,
        files,
        mask: str = "varden1d",
        accel: float = 4,
        crop: int = 320,
        fixed_mask: bool = False,
        skip_edge_slices: int = 0,
        mask_kwargs: dict | None = None,
    ):
        self.mask, self.accel, self.crop = mask, accel, crop
        self.fixed_mask = fixed_mask
        self.mask_kwargs = mask_kwargs or {}
        self.slices = []
        for path in map(Path, files):
            with h5py.File(path, "r") as f:
                n = f["kspace"].shape[0]
            self.slices += [(path, s) for s in range(skip_edge_slices, n - skip_edge_slices)]

    def __len__(self):
        return len(self.slices)

    def _rng(self, path: Path, s: int):
        if self.fixed_mask:
            return np.random.default_rng(zlib.crc32(f"{path.name}:{s}".encode()))
        return np.random.default_rng()

    def __getitem__(self, idx):
        path, s = self.slices[idx]
        with h5py.File(path, "r") as f:
            kspace = f["kspace"][s]
        if kspace.ndim == 3:
            kspace = compress_coils(kspace)

        target = center_crop(ifft2c(torch.from_numpy(kspace).to(torch.cfloat)), self.crop)
        full = fft2c(target)
        mask = torch.from_numpy(make_mask(self.mask, full.shape, self.accel, self._rng(path, s), **self.mask_kwargs))
        under = full * mask
        zero_filled = ifft2c(under)
        scale = zero_filled.abs().max().clamp_min(1e-12)

        return {
            "image": (zero_filled / scale)[None],
            "kspace": (under / scale)[None],
            "mask": mask[None],
            "target": (target / scale)[None],
            "scale": scale,
            "fname": path.name,
            "slice": s,
        }
