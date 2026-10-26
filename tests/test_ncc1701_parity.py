"""Check this port against the original NCC1701 implementation.

Skipped unless ``NCC1701_PATH`` points at a checkout of
https://github.com/soumickmj/NCC1701 and ``torchcomplex`` is installed.
Weights are copied from the original model into ours with ``strict=True``,
so this also guards checkpoint compatibility.
"""

import os
import sys

import pytest
import torch

from mrirecon.fft import fft2c
from mrirecon.models import build_model, convert_ncc1701_state_dict

NCC1701 = os.environ.get("NCC1701_PATH")
pytest.importorskip("torchcomplex")
pytestmark = pytest.mark.skipif(not NCC1701, reason="NCC1701_PATH not set")

# (our model, our kwargs, NCC1701 kwargs) - NCC1701 model IDs 44, 33 and 555.
CASES = {
    "pd-magnitude": (
        "fourier-pd",
        dict(complex_primal=False),
        dict(
            n_primary=5,
            n_dual=5,
            n_iterations=10,
            use_original_block=True,
            use_original_init=True,
            use_complex_primal=False,
        ),
    ),
    "pd-complex": (
        "fourier-pd",
        dict(complex_primal=True),
        dict(
            n_primary=5,
            n_dual=5,
            n_iterations=10,
            use_original_block=True,
            use_original_init=True,
            use_complex_primal=True,
        ),
    ),
    "pdunet-magnitude": (
        "fourier-pdunet",
        dict(complex_primal=False),
        dict(
            n_primary=5,
            n_dual=5,
            n_iterations=10,
            use_original_block=False,
            use_original_init=False,
            use_complex_primal=False,
        ),
    ),
}


@pytest.fixture(scope="module")
def reference_cls():
    sys.path.insert(0, NCC1701)
    from Bridge.WarpDrives.PDUNet.pd_crelu import PrimalDualNetwork

    return PrimalDualNetwork


@pytest.mark.parametrize("case", CASES)
def test_matches_ncc1701(case, reference_cls):
    name, ours_kwargs, ref_kwargs = CASES[case]
    torch.manual_seed(0)
    ref = reference_cls(**ref_kwargs, g_normtype="magmax", transform="Fourier").eval()
    # The primal residual weights start at zero, which would make the comparison
    # trivial; jitter every parameter first.
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ours = build_model(name, **ours_kwargs).eval()
    ours.load_state_dict(convert_ncc1701_state_dict(ref.state_dict()), strict=True)

    image = torch.randn(1, 1, 48, 40, dtype=torch.cfloat)
    kspace = fft2c(image)
    f0 = image if ours.complex_primal else image.abs()
    with torch.no_grad():
        expected = ref(f0, kspace)
        actual = ours(image, kspace)
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=1e-4, atol=1e-5)
