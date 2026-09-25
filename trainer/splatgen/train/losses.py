import torch
import torch.nn.functional as F

_WINDOWS = {}


def _window(channels, device, dtype, size=11, sigma=1.5):
    key = (channels, device, dtype)
    if key not in _WINDOWS:
        coords = torch.arange(size, dtype=dtype, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g = g / g.sum()
        _WINDOWS[key] = (g[:, None] @ g[None, :]).expand(channels, 1, size, size).contiguous()
    return _WINDOWS[key]


def ssim(a, b):
    """Mean SSIM of two HxWx3 images in [0, 1]."""
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    w = _window(x.shape[1], x.device, x.dtype)
    pad = w.shape[-1] // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y, w, padding=pad, groups=x.shape[1])
    sxx = F.conv2d(x * x, w, padding=pad, groups=x.shape[1]) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=x.shape[1]) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=x.shape[1]) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    value = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return value.mean()


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).clamp_min(1e-10)
    return float(-10.0 * torch.log10(mse))
