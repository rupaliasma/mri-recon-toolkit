"""Cartesian undersampling masks.

All masks are float32 arrays of shape (H, W) with 1 at acquired locations.
1D masks select whole columns, i.e. the phase-encode direction is the last
axis, which is the convention of the fastMRI k-space files.

* ``random`` / ``equispaced``: fastMRI-style, a fully sampled centre band of
  ``center_fraction`` columns plus the remaining columns at random or at a
  regular spacing (Zbontar et al. 2018).
* ``varden1d``: columns drawn from a Gaussian density centred on k=0 with a
  fully sampled centre of ``center_lines`` columns ("1D Varden" in NCC1701).
* ``varden2d``: individual k-space points drawn from a 2D Gaussian density
  with a fully sampled central disc covering ``center_area`` of k-space
  ("2D Varden").

For the Varden masks ``accel`` sets the overall sampled fraction, 1/accel,
so accel=4 samples 25% of k-space and accel=10 samples 10%.
"""

import numpy as np

KINDS = ("random", "equispaced", "varden1d", "varden2d")


def _centre_band(n: int, n_low: int) -> np.ndarray:
    keep = np.zeros(n, dtype=bool)
    start = (n - n_low + 1) // 2
    keep[start : start + n_low] = True
    return keep


def _columns_to_mask(cols: np.ndarray, shape) -> np.ndarray:
    return np.broadcast_to(cols[None, :], shape).astype(np.float32)


def random_mask(shape, accel, rng, center_fraction=0.08):
    n = shape[-1]
    n_low = round(n * center_fraction)
    centre = _centre_band(n, n_low)
    # Probability chosen so that the expected total number of columns is n / accel.
    prob = (n / accel - n_low) / (n - n_low)
    cols = centre | (rng.uniform(size=n) < prob)
    return _columns_to_mask(cols, shape)


def equispaced_mask(shape, accel, rng, center_fraction=0.08):
    n = shape[-1]
    n_low = round(n * center_fraction)
    cols = _centre_band(n, n_low)
    # Spacing that compensates for the extra centre lines (as in fastMRI).
    spacing = accel * (n_low - n) / (n_low * accel - n) if n_low * accel != n else accel
    offset = rng.integers(0, max(1, round(spacing)))
    picks = np.round(np.arange(offset, n - 1, spacing)).astype(int)
    cols[picks] = True
    return _columns_to_mask(cols, shape)


def varden1d_mask(shape, accel, rng, center_lines=8, sigma_fraction=0.25):
    n = shape[-1]
    target = max(round(n / accel), center_lines)
    cols = _centre_band(n, center_lines)
    x = np.arange(n) - n / 2
    density = np.exp(-0.5 * (x / (sigma_fraction * n)) ** 2)
    density[cols] = 0
    picks = rng.choice(n, size=target - cols.sum(), replace=False, p=density / density.sum())
    cols[picks] = True
    return _columns_to_mask(cols, shape)


def varden2d_mask(shape, accel, rng, center_area=0.025, sigma_fraction=0.25):
    h, w = shape
    ky, kx = np.meshgrid(np.arange(h) - h / 2, np.arange(w) - w / 2, indexing="ij")
    r2 = (ky / h) ** 2 + (kx / w) ** 2
    # Normalised radius whose disc covers `center_area` of the (unit) k-space square.
    centre = r2 <= center_area / np.pi
    density = np.exp(-0.5 * r2 / sigma_fraction**2).ravel()
    density[centre.ravel()] = 0
    target = max(round(h * w / accel), int(centre.sum()))
    picks = rng.choice(h * w, size=target - centre.sum(), replace=False, p=density / density.sum())
    mask = centre.ravel().copy()
    mask[picks] = True
    return mask.reshape(shape).astype(np.float32)


_BUILDERS = {
    "random": random_mask,
    "equispaced": equispaced_mask,
    "varden1d": varden1d_mask,
    "varden2d": varden2d_mask,
}


def make_mask(kind: str, shape, accel: float, rng: np.random.Generator | None = None, **kwargs) -> np.ndarray:
    if kind not in _BUILDERS:
        raise ValueError(f"unknown mask {kind!r}, choose from {KINDS}")
    if accel < 1:
        raise ValueError("accel must be >= 1")
    rng = rng if rng is not None else np.random.default_rng()
    return _BUILDERS[kind](tuple(shape), accel, rng, **kwargs)
