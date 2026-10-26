import numpy as np
import pytest
import torch

from conftest import write_volume
from mrirecon.data import FastMRISlices, center_crop, compress_coils, split_volumes
from mrirecon.fft import fft2c, ifft2c


def test_singlecoil_sample(fastmri_dir):
    ds = FastMRISlices(sorted(fastmri_dir.glob("*.h5")), mask="random", accel=4, crop=32)
    assert len(ds) == 4 * 3
    s = ds[0]
    for key in ("image", "kspace", "mask", "target"):
        assert s[key].shape == (1, 32, 32), key
    # Measured k-space is the undersampled transform of the target, and the zero-filled image its inverse.
    torch.testing.assert_close(s["kspace"], fft2c(s["target"]) * s["mask"], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(s["image"], ifft2c(s["kspace"]), atol=1e-5, rtol=1e-4)
    assert s["image"].abs().max().item() == pytest.approx(1.0)


def test_fixed_masks_are_deterministic(fastmri_dir):
    files = sorted(fastmri_dir.glob("*.h5"))
    a = FastMRISlices(files, mask="varden2d", accel=8, crop=32, fixed_mask=True)
    b = FastMRISlices(files, mask="varden2d", accel=8, crop=32, fixed_mask=True)
    assert torch.equal(a[5]["mask"], b[5]["mask"])
    assert not torch.equal(a[5]["mask"], a[6]["mask"])


def test_multicoil_is_compressed(tmp_path):
    path = write_volume(tmp_path / "mc.h5", n_slices=2, shape=(30, 28), coils=4)
    s = FastMRISlices([path], mask="random", accel=2, crop=320)[1]
    assert s["target"].shape == (1, 30, 28)
    assert s["target"].is_complex()


def test_coil_compression_recovers_rank_one_data():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((20, 20)) + 1j * rng.standard_normal((20, 20))
    weights = rng.standard_normal(6) + 1j * rng.standard_normal(6)
    virtual = compress_coils(weights[:, None, None] * signal)
    # For rank-one data the virtual coil equals the signal up to a global complex factor.
    factor = virtual.ravel() @ signal.ravel().conj() / (signal.ravel() @ signal.ravel().conj())
    np.testing.assert_allclose(virtual, factor * signal, atol=1e-8)


def test_center_crop_handles_small_inputs():
    x = torch.arange(30).reshape(5, 6)
    assert center_crop(x, 4).shape == (4, 4)
    assert center_crop(x, 10).shape == (5, 6)


def test_split_volumes_is_disjoint_and_stable():
    files = [f"f{i}.h5" for i in range(20)]
    train, val = split_volumes(files, 0.2, seed=1)
    assert len(val) == 4 and not set(train) & set(val)
    assert split_volumes(files, 0.2, seed=1) == (train, val)
