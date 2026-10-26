import torch

from mrirecon.fft import data_consistency, fft2c, ifft2c


def test_roundtrip_and_parseval():
    x = torch.randn(2, 1, 32, 27, dtype=torch.cfloat)
    k = fft2c(x)
    torch.testing.assert_close(ifft2c(k), x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k.abs().pow(2).sum(), x.abs().pow(2).sum(), rtol=1e-4, atol=1e-4)


def test_dc_is_centred():
    x = torch.ones(1, 1, 16, 16, dtype=torch.cfloat)
    k = fft2c(x)
    assert k.abs().argmax().item() == 8 * 16 + 8


def test_data_consistency_keeps_measured_samples():
    target = torch.randn(1, 1, 24, 24, dtype=torch.cfloat)
    mask = (torch.rand(1, 1, 24, 24) > 0.5).float()
    measured = fft2c(target) * mask
    estimate = torch.randn_like(target)
    out = data_consistency(estimate, measured, mask)
    torch.testing.assert_close(fft2c(out) * mask, measured, atol=1e-5, rtol=1e-5)
    # Unmeasured locations come from the estimate.
    torch.testing.assert_close(fft2c(out) * (1 - mask), fft2c(estimate) * (1 - mask), atol=1e-5, rtol=1e-5)


def test_data_consistency_with_full_mask_returns_measurement():
    target = torch.randn(1, 1, 16, 16, dtype=torch.cfloat)
    out = data_consistency(torch.zeros(1, 1, 16, 16), fft2c(target), torch.ones(1, 1, 16, 16))
    torch.testing.assert_close(out, target, atol=1e-5, rtol=1e-5)
