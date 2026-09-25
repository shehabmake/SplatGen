"""The trainable Gaussian set.

Parameters are stored in their optimisation space, exactly as the reference
3DGS implementation and the PLY format do:

    means      (N, 3)   world position
    scales     (N, 3)   log of the axis standard deviations
    quats      (N, 4)   rotation, w x y z (normalised when used)
    opacities  (N,)     logit of opacity
    sh0        (N, 1, 3) DC colour coefficient
    shN        (N, K-1, 3) higher-order SH coefficients
"""

import math

import numpy as np
import torch

from . import sh as shmod

PARAM_NAMES = ("means", "scales", "quats", "opacities", "sh0", "shN")


def knn_mean_distance(points, k=3, chunk=4096):
    """Mean distance to the k nearest neighbours, chunked to bound memory."""
    points = points.float()
    n = points.shape[0]
    if n <= 1:
        return torch.ones(n, device=points.device)
    k = min(k, n - 1)
    out = torch.empty(n, device=points.device)
    for start in range(0, n, chunk):
        block = points[start:start + chunk]
        dist = torch.cdist(block, points)
        dist[torch.arange(len(block)), torch.arange(start, start + len(block))] = float("inf")
        out[start:start + chunk] = dist.topk(k, largest=False).values.mean(dim=1)
    return out


def init_params(xyz, rgb, sh_degree, init_opacity=0.1, init_scale=1.0):
    """Initial parameters from points (N,3) and colours in [0,1] (N,3)."""
    xyz = torch.as_tensor(np.asarray(xyz), dtype=torch.float32)
    rgb = torch.as_tensor(np.asarray(rgb), dtype=torch.float32)
    n = xyz.shape[0]
    dist = knn_mean_distance(xyz).clamp_min(1e-7)
    scales = torch.log(dist * init_scale).unsqueeze(-1).repeat(1, 3)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), float(init_opacity)))
    k = shmod.num_coeffs(sh_degree)
    sh0 = shmod.rgb_to_sh(rgb).unsqueeze(1)
    shN = torch.zeros(n, k - 1, 3)
    return {"means": xyz, "scales": scales, "quats": quats,
            "opacities": opacities, "sh0": sh0, "shN": shN}


def random_points(count, center, radius, seed=0):
    generator = np.random.default_rng(seed)
    xyz = (generator.random((count, 3)) * 2 - 1) * radius + np.asarray(center)
    rgb = generator.random((count, 3))
    return xyz, rgb


class GaussianModel:
    """A dict of ``torch.nn.Parameter`` plus activation helpers."""

    def __init__(self, params, device="cpu"):
        self.params = {name: torch.nn.Parameter(torch.as_tensor(params[name]).float().to(device))
                       for name in PARAM_NAMES}

    def __len__(self):
        return self.params["means"].shape[0]

    @property
    def device(self):
        return self.params["means"].device

    @property
    def max_sh_degree(self):
        return int(math.isqrt(self.params["shN"].shape[1] + 1) - 1)

    def activated(self):
        p = self.params
        return {
            "means": p["means"],
            "scales": torch.exp(p["scales"]),
            "quats": torch.nn.functional.normalize(p["quats"], dim=-1),
            "opacities": torch.sigmoid(p["opacities"]),
            "sh": torch.cat([p["sh0"], p["shN"]], dim=1),
        }

    def state_dict(self):
        return {name: value.detach().cpu() for name, value in self.params.items()}

    @classmethod
    def from_state_dict(cls, state, device="cpu"):
        return cls(state, device=device)
