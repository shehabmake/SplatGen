"""Background: the sky / world seen behind the geometry, as a far shell.

Pixels with no surface (depth at the sky value) only carry a direction.
They are binned on a cube map around the scene centre; every observed bin
becomes one flat splat on a sphere well outside the geometry, facing the
centre, with the bin's mean colour. Pixels are binned by where their camera
ray meets that sphere, so cameras away from the centre agree. The pixels of each bin are returned too,
so the check-&-repeat rounds can correct the shell like any other splat.
"""

import numpy as np
import torch

from .samples import SKY
from .shape import rotmat_to_quat


def _sky_pixels(raw, view, center, radius):
    depth = raw.load_pass("depth", view)
    sky = ~(np.isfinite(depth) & (depth < SKY * 0.5) & (depth > 0))
    # Drop sky pixels touching geometry: their colour is a blend with the edge.
    solid = ~sky
    grown = solid.copy()
    grown[1:] |= solid[:-1]
    grown[:-1] |= solid[1:]
    grown[:, 1:] |= solid[:, :-1]
    grown[:, :-1] |= solid[:, 1:]
    idx = np.flatnonzero((sky & ~grown).reshape(-1))
    H, W = depth.shape
    v, u = np.divmod(idx, W)
    rays = np.stack([(u + 0.5 - view.cx) / view.fx, (v + 0.5 - view.cy) / view.fy, np.ones(len(idx))], -1)
    dirs = rays @ view.c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    # Where each ray leaves the shell, seen from the centre (the camera is inside).
    o = view.center - center
    b = dirs @ o
    t = -b + np.sqrt(np.maximum(b * b - (o @ o - radius * radius), 0))
    hit = o[None] + t[:, None] * dirs
    dirs = hit / np.linalg.norm(hit, axis=-1, keepdims=True)
    colour = raw.load_image(view).reshape(-1, 3)[idx]
    return idx, dirs, colour


def _cube_bin(dirs, res):
    a = np.abs(dirs)
    axis = a.argmax(-1)
    sign = np.take_along_axis(dirs, axis[:, None], -1)[:, 0] >= 0
    face = axis * 2 + (~sign)
    major = np.take_along_axis(a, axis[:, None], -1)[:, 0]
    other = np.stack([dirs[np.arange(len(dirs)), (axis + 1) % 3], dirs[np.arange(len(dirs)), (axis + 2) % 3]], -1)
    uv = other / major[:, None]                                   # [-1, 1]
    cell = np.clip(((uv + 1) * 0.5 * res).astype(np.int64), 0, res - 1)
    return (face * res + cell[:, 0]) * res + cell[:, 1]


def _bin_direction(bins, res):
    face, rest = np.divmod(bins, res * res)
    i, j = np.divmod(rest, res)
    axis, negative = face // 2, face % 2 == 1
    d = np.zeros((len(bins), 3))
    n = np.arange(len(bins))
    d[n, axis] = np.where(negative, -1.0, 1.0)
    d[n, (axis + 1) % 3] = (i + 0.5) / res * 2 - 1
    d[n, (axis + 2) % 3] = (j + 0.5) / res * 2 - 1
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def build_background(raw, views, center, radius, config, device, log=print):
    """Shell splats and their pixel observations, or None when no sky is visible."""
    res = int(config.background_resolution)
    obs_view, obs_pix, obs_bin, obs_col = [], [], [], []
    for view in views:
        idx, dirs, colour = _sky_pixels(raw, view, np.asarray(center), radius)
        if len(idx) == 0:
            continue
        obs_view.append(np.full(len(idx), view.index, np.int64))
        obs_pix.append(idx)
        obs_bin.append(_cube_bin(dirs, res))
        obs_col.append(colour)
    if not obs_view:
        return None
    obs_bin = np.concatenate(obs_bin)
    bins, inverse = np.unique(obs_bin, return_inverse=True)
    colour = np.zeros((len(bins), 3))
    np.add.at(colour, inverse, np.concatenate(obs_col))
    colour /= np.bincount(inverse, minlength=len(bins))[:, None]
    direction = _bin_direction(bins, res)
    means = np.asarray(center)[None] + radius * direction
    # A cube cell spans about 2/res of the face at unit distance.
    sigma = 1.2 * radius / res
    normal = -direction
    helper = np.where(np.abs(normal[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
    t1 = np.cross(helper, normal)
    t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    t2 = np.cross(normal, t1)
    R = torch.tensor(np.stack([t1, t2, normal], -1), dtype=torch.float32)
    log(f"background: {len(bins):,} shell splats from sky pixels, radius {radius:.3g}")
    to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    return {
        "means": to(means),
        "scales": torch.log(to([[sigma, sigma, sigma * 0.05]]).expand(len(bins), 3).clone()),
        "quats": rotmat_to_quat(R).to(device),
        "color": to(colour),
        "obs_view": to(np.concatenate(obs_view), torch.long),
        "obs_pix": to(np.concatenate(obs_pix), torch.long),
        "obs_splat": to(inverse, torch.long),
    }
