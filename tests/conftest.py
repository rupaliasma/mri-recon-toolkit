import h5py
import numpy as np
import pytest
import torch

from mrirecon.fft import fft2c


def phantom(h, w, rng):
    """A few random ellipses with a smooth phase ramp: enough structure for SSIM to mean something."""
    y, x = np.mgrid[-1 : 1 : h * 1j, -1 : 1 : w * 1j]
    img = np.zeros((h, w))
    for _ in range(6):
        cy, cx, ry, rx = rng.uniform(-0.5, 0.5, 2).tolist() + rng.uniform(0.1, 0.4, 2).tolist()
        img += rng.uniform(0.2, 1.0) * (((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1)
    phase = np.pi * (0.3 * x + 0.2 * y)
    return (img * np.exp(1j * phase)).astype(np.complex64)


def write_volume(path, n_slices=3, shape=(40, 36), coils=None, seed=0):
    rng = np.random.default_rng(seed)
    images = np.stack([phantom(*shape, rng) for _ in range(n_slices)])
    if coils:
        sens = np.stack([phantom(*shape, rng) + 0.5 for _ in range(coils)])
        images = images[:, None] * sens[None]
    kspace = fft2c(torch.from_numpy(images)).numpy()
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=kspace)
    return path


@pytest.fixture
def fastmri_dir(tmp_path):
    """Four tiny single-coil volumes laid out like a fastMRI split."""
    for i in range(4):
        write_volume(tmp_path / f"file{i:03d}.h5", seed=i)
    return tmp_path
