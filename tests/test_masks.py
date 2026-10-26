import numpy as np
import pytest

from mrirecon.masks import KINDS, make_mask

SHAPE = (320, 368)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("accel", [4, 8, 10])
def test_sampling_fraction(kind, accel):
    rng = np.random.default_rng(0)
    fractions = [make_mask(kind, SHAPE, accel, rng).mean() for _ in range(20)]
    assert make_mask(kind, SHAPE, accel, rng).shape == SHAPE
    assert abs(np.mean(fractions) - 1 / accel) < 0.02


@pytest.mark.parametrize("kind", ["random", "equispaced", "varden1d"])
def test_1d_masks_select_whole_columns(kind):
    mask = make_mask(kind, SHAPE, 4, np.random.default_rng(1))
    assert (mask == mask[:1]).all()


@pytest.mark.parametrize("kind", KINDS)
def test_centre_always_sampled(kind):
    mask = make_mask(kind, SHAPE, 8, np.random.default_rng(2))
    assert mask[SHAPE[0] // 2, SHAPE[1] // 2] == 1
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_seeded_masks_are_reproducible():
    a = make_mask("varden2d", SHAPE, 10, np.random.default_rng(3))
    b = make_mask("varden2d", SHAPE, 10, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_variable_density_favours_low_frequencies():
    mask = make_mask("varden1d", SHAPE, 4, np.random.default_rng(4))[0]
    centre, edge = mask[SHAPE[1] // 2 - 40 : SHAPE[1] // 2 + 40].mean(), np.r_[mask[:40], mask[-40:]].mean()
    assert centre > 2 * edge


def test_bad_arguments():
    with pytest.raises(ValueError):
        make_mask("spiral", SHAPE, 4)
    with pytest.raises(ValueError):
        make_mask("random", SHAPE, 0.5)
