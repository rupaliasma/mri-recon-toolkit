import numpy as np
import pytest
import torch

from mrirecon.metrics import nmse, nrmse, psnr, ssim


def test_identical_images():
    x = torch.rand(3, 64, 64)
    torch.testing.assert_close(ssim(x, x), torch.ones(3, dtype=torch.float64))
    assert torch.isinf(psnr(x, x)).all()
    assert (nmse(x, x) == 0).all()


def test_ssim_decreases_with_noise():
    x = torch.rand(64, 64)
    scores = [ssim(x + s * torch.randn(64, 64), x).item() for s in (0.01, 0.1, 0.5)]
    assert scores[0] > scores[1] > scores[2]


def test_psnr_known_value():
    target = torch.ones(16, 16)
    pred = target + 0.1
    assert psnr(pred, target).item() == pytest.approx(20.0, abs=1e-4)


def test_complex_inputs_use_magnitude_except_nrmse():
    x = torch.rand(32, 32) + 0.1
    rotated = x * torch.exp(torch.tensor(1j))
    assert ssim(rotated, x.to(torch.cfloat)).item() == pytest.approx(1.0)
    assert nrmse(rotated, x.to(torch.cfloat)).item() > 0.5


def test_ssim_matches_scikit_image():
    skm = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(0)
    gt = rng.random((80, 72))
    pred = gt + 0.05 * rng.standard_normal(gt.shape)
    expected = skm.structural_similarity(gt, pred, data_range=gt.max())
    assert ssim(torch.from_numpy(pred), torch.from_numpy(gt)).item() == pytest.approx(expected, abs=1e-6)
