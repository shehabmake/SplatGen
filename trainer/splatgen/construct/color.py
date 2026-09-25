"""Step 5 - colour: view-dependent colour fitted directly from every camera.

Each splat's samples are grouped by the view they came from, giving one
observed colour per (splat, camera direction). Spherical-harmonic
coefficients are then solved per splat by weighted, regularised least
squares - no gradient descent. Higher bands are damped more, and damped
further on rough materials (raw roughness pass), so matte surfaces stay
view-independent while glossy ones keep their reflections.
"""

import torch

from ..model.sh import C0, eval_sh


def sh_basis(degree, dirs):
    k = (degree + 1) ** 2
    eye = torch.eye(k, device=dirs.device).expand(dirs.shape[0], k, k)
    return eval_sh(degree, eye, dirs)                         # (M, K)


def fit_sh(samples, splat, n_splats, means, view_centers, config, color=None, block=100_000):
    """SH coefficients (N, K, 3) of every splat and the number of views that saw it."""
    dev = means.device
    degree = int(config.sh_degree)
    K = (degree + 1) ** 2
    n_views = view_centers.shape[0]
    color = samples["col"] if color is None else color
    key = splat * n_views + samples["view"].long()
    keys, inverse = torch.unique(key, return_inverse=True)       # sorted: grouped by splat
    w = samples["w"]
    obs_w = torch.zeros(len(keys), device=dev).index_add_(0, inverse, w)
    obs_c = torch.zeros(len(keys), 3, device=dev).index_add_(0, inverse, w[:, None] * color)
    obs_c = obs_c / obs_w.clamp_min(1e-12)[:, None]
    obs_splat = keys // n_views
    obs_view = keys % n_views
    rough = torch.zeros(n_splats, device=dev).index_add_(0, splat, w * samples["rough"])
    total = torch.zeros(n_splats, device=dev).index_add_(0, splat, w)
    rough = rough / total.clamp_min(1e-12)
    band = torch.tensor([l for l in range(degree + 1) for _ in range(2 * l + 1)], device=dev, dtype=torch.float32)
    band_reg = config.sh_regularization * band * (band + 1)
    # One observation per view: damp the pixel count so no view dominates.
    ow = obs_w.sqrt()
    bounds = torch.searchsorted(obs_splat, torch.arange(0, n_splats + block, block, device=dev).clamp_max(n_splats))
    coeffs = torch.zeros(n_splats, K, 3, device=dev)
    for i in range(len(bounds) - 1):
        s0 = i * block
        s1 = min(n_splats, s0 + block)
        if s0 >= s1:
            break
        o = slice(int(bounds[i]), int(bounds[i + 1]))
        local = obs_splat[o] - s0
        dirs = torch.nn.functional.normalize(means[obs_splat[o]] - view_centers[obs_view[o]], dim=-1)
        B = sh_basis(degree, dirs) * ow[o, None].sqrt()
        A = torch.zeros(s1 - s0, K, K, device=dev).index_add_(0, local, B[:, :, None] * B[:, None, :])
        b = torch.zeros(s1 - s0, K, 3, device=dev).index_add_(
            0, local, B[:, :, None] * ((obs_c[o] - 0.5) * ow[o, None].sqrt())[:, None, :])
        scale = A[:, 0, 0].clamp_min(1e-9)[:, None] / (C0 * C0)
        reg = band_reg * (1 + config.roughness_regularization * rough[s0:s1, None]) + 1e-6
        A = A + torch.diag_embed(reg * scale)
        coeffs[s0:s1] = torch.linalg.solve(A, b)
    views_seen = torch.bincount(obs_splat, minlength=n_splats)
    return coeffs, views_seen
