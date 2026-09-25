"""Differentiable Gaussian rasterizer in plain PyTorch.

Runs anywhere PyTorch runs (CPU, CUDA, Apple MPS). It is far slower than
gsplat's CUDA kernels and meant for small scenes, tests and machines without
an NVIDIA GPU. It follows the same maths as the reference implementation:
EWA-projected 2D covariances with a 0.3 px low-pass, per-tile depth-sorted
front-to-back alpha compositing, alpha capped at 0.99 and cut below 1/255.
"""

import math

import torch

from ..model.sh import eval_sh

NAME = "torch"
TILE = 16


def available():
    return True


def quat_to_rotmat(quats):
    w, x, y, z = quats.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(quats.shape[:-1] + (3, 3))


def project(means, quats, scales, viewmat, K, width, height, near=0.01, eps2d=0.3):
    """Screen-space means, conics, radii and depths for every Gaussian."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    pc = means @ R.T + t
    x, y, z = pc.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    zc = z.clamp_min(near)
    # Clamp the Jacobian's view angle as the reference does, for stability of
    # Gaussians far outside the frustum.
    lim_x = 1.3 * (0.5 * width / fx)
    lim_y = 1.3 * (0.5 * height / fy)
    tx = (x / zc).clamp(-lim_x - cx / fx, lim_x + (width - cx) / fx) * zc
    ty = (y / zc).clamp(-lim_y - cy / fy, lim_y + (height - cy) / fy) * zc
    J = torch.zeros(means.shape[0], 2, 3, device=means.device, dtype=means.dtype)
    J[:, 0, 0] = fx / zc
    J[:, 0, 2] = -fx * tx / (zc * zc)
    J[:, 1, 1] = fy / zc
    J[:, 1, 2] = -fy * ty / (zc * zc)
    rot = quat_to_rotmat(quats)
    M = rot * scales.unsqueeze(-2)
    cov3 = M @ M.transpose(-1, -2)
    cov_cam = R @ cov3 @ R.T
    cov2 = J @ cov_cam @ J.transpose(-1, -2)
    a = cov2[:, 0, 0] + eps2d
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + eps2d
    det = (a * c - b * b).clamp_min(1e-12)
    conic = torch.stack([c / det, -b / det, a / det], dim=-1)
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
    radius = torch.ceil(3.0 * torch.sqrt(lam))
    mean2d = torch.stack([fx * x / zc + cx, fy * y / zc + cy], dim=-1)
    with torch.no_grad():
        inside = ((z > near)
                  & (mean2d[:, 0] + radius > 0) & (mean2d[:, 0] - radius < width)
                  & (mean2d[:, 1] + radius > 0) & (mean2d[:, 1] - radius < height))
        radii = torch.where(inside, radius, torch.zeros_like(radius)).to(torch.int32)
    return mean2d, conic, radii, z


def rasterize(means, quats, scales, opacities, sh, sh_degree, viewmat, K,
              width, height, background):
    """Render one view. Returns (rgb HxWx3, alpha HxW, info)."""
    device = means.device
    mean2d, conic, radii, depth = project(means, quats, scales, viewmat, K, width, height)
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    campos = -R.T @ t
    dirs = torch.nn.functional.normalize(means - campos, dim=-1)
    rgb = (eval_sh(sh_degree, sh[:, : (sh_degree + 1) ** 2], dirs) + 0.5).clamp_min(0.0)

    visible = torch.nonzero(radii > 0).squeeze(-1)
    order = visible[torch.argsort(depth[visible].detach())]
    image = background.to(device).expand(height, width, 3).clone()
    alpha_img = torch.zeros(height, width, device=device)
    if len(order):
        m2 = mean2d[order]
        con = conic[order]
        opa = opacities[order]
        col = rgb[order]
        rad = radii[order].float()
        lo = (m2.detach() - rad[:, None])
        hi = (m2.detach() + rad[:, None])
        tiles_x = math.ceil(width / TILE)
        tiles_y = math.ceil(height / TILE)
        rows_out = []
        for ty in range(tiles_y):
            y0, y1 = ty * TILE, min(height, (ty + 1) * TILE)
            row_rgb, row_alpha = [], []
            band = (hi[:, 1] >= y0) & (lo[:, 1] < y1)
            for tx in range(tiles_x):
                x0, x1 = tx * TILE, min(width, (tx + 1) * TILE)
                sel = torch.nonzero(band & (hi[:, 0] >= x0) & (lo[:, 0] < x1)).squeeze(-1)
                h, w = y1 - y0, x1 - x0
                if len(sel) == 0:
                    row_rgb.append(background.to(device).expand(h, w, 3))
                    row_alpha.append(torch.zeros(h, w, device=device))
                    continue
                ys, xs = torch.meshgrid(
                    torch.arange(y0, y1, device=device, dtype=means.dtype) + 0.5,
                    torch.arange(x0, x1, device=device, dtype=means.dtype) + 0.5,
                    indexing="ij")
                pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
                d = pix[:, None, :] - m2[sel][None]
                cs = con[sel]
                power = (-0.5 * (cs[:, 0] * d[..., 0] ** 2 + cs[:, 2] * d[..., 1] ** 2)
                         - cs[:, 1] * d[..., 0] * d[..., 1])
                alpha = (opa[sel] * torch.exp(power)).clamp(max=0.99)
                alpha = torch.where((power > 0) | (alpha < 1.0 / 255.0),
                                    torch.zeros_like(alpha), alpha)
                trans = torch.cumprod(1.0 - alpha, dim=1)
                trans_before = torch.cat([torch.ones_like(trans[:, :1]), trans[:, :-1]], dim=1)
                weights = alpha * trans_before
                color = weights @ col[sel] + trans[:, -1:] * background.to(device)
                row_rgb.append(color.reshape(h, w, 3))
                row_alpha.append((1.0 - trans[:, -1]).reshape(h, w))
            rows_out.append((torch.cat(row_rgb, dim=1), torch.cat(row_alpha, dim=1)))
        image = torch.cat([r for r, _ in rows_out], dim=0)
        alpha_img = torch.cat([a for _, a in rows_out], dim=0)
    return image, alpha_img, {"means2d": mean2d, "radii": radii,
                              "width": width, "height": height}
