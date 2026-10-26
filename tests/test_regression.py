"""Regression tests: fixed weights and inputs must keep producing the stored outputs.

The reference values live in tests/data/regression.json. After an intentional
change to the numerics, regenerate them with

    MRIRECON_UPDATE_REGRESSION=1 pytest tests/test_regression.py

and explain the change in the commit message.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from mrirecon.fft import data_consistency, fft2c, ifft2c
from mrirecon.metrics import all_metrics
from mrirecon.models import build_model

REFERENCE = Path(__file__).parent / "data" / "regression.json"
UPDATE = os.environ.get("MRIRECON_UPDATE_REGRESSION") == "1"
CASES = {
    "fourier-pd-complex": ("fourier-pd", {"complex_primal": True}),
    "fourier-pd-magnitude": ("fourier-pd", {"complex_primal": False}),
    "fourier-pdunet-complex": ("fourier-pdunet", {"complex_primal": True}),
    "fourier-pdunet-magnitude": ("fourier-pdunet", {"complex_primal": False}),
}


def _seeded_model(name, kwargs):
    # Weights are drawn from our own generator rather than PyTorch's initialisers,
    # whose algorithms are free to change between releases.
    model = build_model(name, n_iterations=3, **kwargs).eval()
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.copy_(0.1 * torch.randn(p.shape, dtype=p.dtype, generator=gen))
    return model


def _inputs():
    gen = torch.Generator().manual_seed(42)
    target = torch.randn(1, 1, 24, 20, dtype=torch.cfloat, generator=gen)
    mask = (torch.rand(1, 1, 1, 20, generator=gen) < 0.4).float()
    mask[..., 8:12] = 1
    kspace = fft2c(target) * mask
    return target, kspace, mask


def _summary(x):
    x = torch.view_as_real(x) if x.is_complex() else x
    return [round(v, 6) for v in x.flatten()[::7].tolist()]


@pytest.fixture(scope="module")
def reference():
    return {} if UPDATE else json.loads(REFERENCE.read_text())


@pytest.fixture(scope="module", autouse=True)
def maybe_write(reference):
    yield
    if UPDATE:
        REFERENCE.parent.mkdir(exist_ok=True)
        REFERENCE.write_text(json.dumps(reference, indent=1, sort_keys=True))


@pytest.mark.parametrize("case", CASES)
def test_model_outputs(case, reference):
    target, kspace, mask = _inputs()
    with torch.no_grad():
        out = _seeded_model(*CASES[case])(ifft2c(kspace), kspace)
        out = data_consistency(out, kspace, mask)
    got = _summary(out)
    if UPDATE:
        reference[case] = got
    else:
        torch.testing.assert_close(torch.tensor(got), torch.tensor(reference[case]), rtol=1e-4, atol=1e-5)


def test_metrics(reference):
    gen = torch.Generator().manual_seed(7)
    target = torch.rand(2, 32, 32, generator=gen)
    pred = target + 0.05 * torch.randn(2, 32, 32, generator=gen)
    got = {k: [round(x, 8) for x in v.tolist()] for k, v in all_metrics(pred, target).items()}
    if UPDATE:
        reference["metrics"] = got
    else:
        for key, values in got.items():
            assert values == pytest.approx(reference["metrics"][key], rel=1e-6), key
