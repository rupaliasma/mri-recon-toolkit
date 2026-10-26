"""Image-quality metrics, computed per image on (..., H, W) tensors.

SSIM follows the fastMRI evaluation (scikit-image defaults: 7x7 uniform window,
K1=0.01, K2=0.03, sample covariance, border of half a window excluded) with the
data range taken from the target. PSNR and NMSE are as defined in the fastMRI
paper. Complex inputs are compared by magnitude, except for ``nrmse`` which is
computed on the complex values and is therefore also sensitive to phase errors.
"""

import torch
import torch.nn.functional as F


def _batch(x: torch.Tensor) -> torch.Tensor:
    x = x.abs() if x.is_complex() else x
    return x.reshape(-1, *x.shape[-2:]).double()


def _data_range(target: torch.Tensor) -> torch.Tensor:
    return target.amax(dim=(-2, -1))


def ssim(pred: torch.Tensor, target: torch.Tensor, win: int = 7, k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    shape = pred.shape[:-2]
    x, y = _batch(pred)[:, None], _batch(target)[:, None]
    L = _data_range(y[:, 0])[:, None, None, None]
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2

    mean = lambda t: F.avg_pool2d(t, win, stride=1)  # noqa: E731
    mx, my = mean(x), mean(y)
    cov_norm = win * win / (win * win - 1)
    vx = cov_norm * (mean(x * x) - mx * mx)
    vy = cov_norm * (mean(y * y) - my * my)
    vxy = cov_norm * (mean(x * y) - mx * my)

    s = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return s.mean(dim=(1, 2, 3)).reshape(shape)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x, y = _batch(pred), _batch(target)
    mse = ((x - y) ** 2).mean(dim=(-2, -1))
    return (10 * torch.log10(_data_range(y) ** 2 / mse)).reshape(pred.shape[:-2])


def nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x, y = _batch(pred), _batch(target)
    return (((x - y) ** 2).sum(dim=(-2, -1)) / (y**2).sum(dim=(-2, -1))).reshape(pred.shape[:-2])


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """||pred - target|| / ||target||, on complex values when given."""
    diff = torch.linalg.vector_norm(pred - target, dim=(-2, -1))
    return diff / torch.linalg.vector_norm(target, dim=(-2, -1))


def all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "ssim": ssim(pred, target),
        "psnr": psnr(pred, target),
        "nmse": nmse(pred, target),
        "nrmse": nrmse(pred, target),
    }
