import pytest
import torch

from mrirecon.fft import fft2c
from mrirecon.models import MODELS, build_model

SMALL = dict(n_iterations=2)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("complex_primal", [True, False])
@pytest.mark.parametrize("shape", [(32, 32), (33, 27)])
def test_forward_backward(name, complex_primal, shape):
    model = build_model(name, complex_primal=complex_primal, **SMALL)
    image = torch.randn(2, 1, *shape, dtype=torch.cfloat)
    out = model(image, fft2c(image))
    assert out.shape == image.shape
    assert out.is_complex() == complex_primal
    out.abs().mean().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_kspace_defaults_to_fft_of_image():
    model = build_model("fourier-pd", **SMALL).eval()
    image = torch.randn(1, 1, 16, 16, dtype=torch.cfloat)
    with torch.no_grad():
        torch.testing.assert_close(model(image), model(image, fft2c(image)))


def test_pdunet_starts_as_identity_on_its_input():
    # Primal residual weights start at ~0, so an untrained PDUNet returns its zero-filled input.
    model = build_model("fourier-pdunet", **SMALL).eval()
    image = torch.randn(1, 1, 16, 16, dtype=torch.cfloat)
    with torch.no_grad():
        torch.testing.assert_close(model(image), image, atol=1e-5, rtol=1e-4)


def test_published_sizes():
    pd, pdunet = build_model("fourier-pd"), build_model("fourier-pdunet")
    assert len(pd.primal_blocks) == len(pd.dual_blocks) == 10
    assert pd.n_primal == pd.n_dual == 5
    assert pdunet.init_from_input and not pd.init_from_input


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("ista-net")
