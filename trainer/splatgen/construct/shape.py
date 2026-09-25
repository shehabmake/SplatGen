"""Step 4 - shape: one flat Gaussian per leaf, two across a colour edge.

For every splat the weighted sums give the sample mean and covariance. The
disk lies in the plane of the mean normal; its two in-plane axes and sizes
come from the in-plane covariance (so a partly covered cell gives a smaller
or stretched splat), scaled so neighbours overlap without holes, and never
smaller than half a pixel footprint.

Edges: a finest-level leaf that still has too much colour detail is fitted
with a linear colour model c(x) = c0 + G (x - mu) from the same sums. Its
samples are split in two by colour along the dominant colour change, each
half becoming its own splat with its own colour and its own (long, thin)
shape - so the border between them follows the real edge.
This is what resolves text and sharp borders below the cell size.
"""

import torch



def group_moments(samples, group, n):
    """Exact per-group statistics in two passes (means, then centred products).

    One-pass sums of world coordinates lose the variance of small cells to
    float32 cancellation (x^2 ~ 100 against a variance of 1e-6), so shapes
    are always computed from centred values."""
    dev = samples["pos"].device
    w = samples["w"]
    wsum = torch.zeros(n, device=dev).index_add_(0, group, w)
    inv = 1.0 / wsum.clamp_min(1e-12)
    mu = torch.zeros(n, 3, device=dev).index_add_(0, group, w[:, None] * samples["pos"]) * inv[:, None]
    color = torch.zeros(n, 3, device=dev).index_add_(0, group, w[:, None] * samples["col"]) * inv[:, None]
    normal = torch.nn.functional.normalize(
        torch.zeros(n, 3, device=dev).index_add_(0, group, w[:, None] * samples["nrm"]), dim=-1)
    d = samples["pos"] - mu[group]
    dc = samples["col"] - color[group]
    cov = torch.zeros(n, 9, device=dev).index_add_(
        0, group, w[:, None] * (d[:, :, None] * d[:, None, :]).reshape(-1, 9)).reshape(n, 3, 3) * inv[:, None, None]
    cx = torch.zeros(n, 9, device=dev).index_add_(
        0, group, w[:, None] * (dc[:, :, None] * d[:, None, :]).reshape(-1, 9)).reshape(n, 3, 3) * inv[:, None, None]
    return wsum, mu, cov, normal, color, cx


def tangent_basis(normal):
    helper = torch.where(normal[:, 2:3].abs() < 0.9,
                         torch.tensor([0.0, 0.0, 1.0], device=normal.device).expand_as(normal),
                         torch.tensor([1.0, 0.0, 0.0], device=normal.device).expand_as(normal))
    t1 = torch.nn.functional.normalize(torch.cross(helper, normal, dim=-1), dim=-1)
    t2 = torch.cross(normal, t1, dim=-1)
    return torch.stack([t1, t2], dim=-1)          # (N, 3, 2)


def edge_directions(moments):
    """World-space dominant colour-gradient direction and the colour change
    across one standard deviation along it."""
    _w, _mu, cov, normal, _color, cx = moments
    T = tangent_basis(normal)
    c2 = T.transpose(1, 2) @ cov @ T                                    # (N,2,2)
    eps = 1e-12 + 1e-3 * c2.diagonal(dim1=1, dim2=2).sum(-1)
    c2 = c2 + eps[:, None, None] * torch.eye(2, device=cov.device)
    G = (cx @ T) @ torch.linalg.inv(c2)                                  # (N,3,2) colour per unit length
    S = G.transpose(1, 2) @ G
    evals, evecs = torch.linalg.eigh(S)
    e = evecs[:, :, 1]                                                   # major in-plane direction
    spread = torch.sqrt((e[:, None, :] @ c2 @ e[:, :, None]).squeeze(-1).squeeze(-1).clamp_min(0))
    contrast = evals[:, 1].clamp_min(0).sqrt() * spread
    return (T @ e[:, :, None]).squeeze(-1), contrast


def rotmat_to_quat(R):
    """Batched rotation matrices -> quaternions (w, x, y, z)."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    trace = m00 + m11 + m22
    q = torch.zeros(R.shape[0], 4, device=R.device)
    s = torch.sqrt((1 + torch.stack([trace, m00 - m11 - m22, -m00 + m11 - m22, -m00 - m11 + m22], -1)).clamp_min(1e-12)) * 2
    best = torch.stack([trace, m00, m11, m22], -1).argmax(-1)
    for k in range(4):
        sel = best == k
        if not sel.any():
            continue
        Rs, ss = R[sel], s[sel, k]
        if k == 0:
            q[sel] = torch.stack([0.25 * ss, (Rs[:, 2, 1] - Rs[:, 1, 2]) / ss,
                                  (Rs[:, 0, 2] - Rs[:, 2, 0]) / ss, (Rs[:, 1, 0] - Rs[:, 0, 1]) / ss], -1)
        elif k == 1:
            q[sel] = torch.stack([(Rs[:, 2, 1] - Rs[:, 1, 2]) / ss, 0.25 * ss,
                                  (Rs[:, 0, 1] + Rs[:, 1, 0]) / ss, (Rs[:, 0, 2] + Rs[:, 2, 0]) / ss], -1)
        elif k == 2:
            q[sel] = torch.stack([(Rs[:, 0, 2] - Rs[:, 2, 0]) / ss, (Rs[:, 0, 1] + Rs[:, 1, 0]) / ss,
                                  0.25 * ss, (Rs[:, 1, 2] + Rs[:, 2, 1]) / ss], -1)
        else:
            q[sel] = torch.stack([(Rs[:, 1, 0] - Rs[:, 0, 1]) / ss, (Rs[:, 0, 2] + Rs[:, 2, 0]) / ss,
                                  (Rs[:, 1, 2] + Rs[:, 2, 1]) / ss, 0.25 * ss], -1)
    return torch.nn.functional.normalize(q, dim=-1)


def _colour_sides(col, w, leaf, n_leaves, mean_color, cx, direction, iterations=3):
    """Which side of its edge each sample is on, decided by colour.

    Samples are projected on the leaf's main colour-change axis (the colour
    that changes along ``direction``) and split in two by a few rounds of 1-D
    k-means, so the halves follow the real border wherever it crosses the
    cell rather than cutting the cell at its centre."""
    axis = torch.nn.functional.normalize((cx @ direction[:, :, None]).squeeze(-1), dim=-1)
    proj = ((col - mean_color[leaf]) * axis[leaf]).sum(-1)
    threshold = torch.zeros(n_leaves, device=col.device)
    for _ in range(iterations):
        side = proj > threshold[leaf]
        hi = _weighted_mean(proj, w, leaf, side, n_leaves)
        lo = _weighted_mean(proj, w, leaf, ~side, n_leaves)
        threshold = 0.5 * (hi + lo)
    return proj > threshold[leaf]


def _weighted_mean(values, w, leaf, mask, n):
    total = torch.zeros(n, device=values.device).index_add_(0, leaf[mask], w[mask] * values[mask])
    weight = torch.zeros(n, device=values.device).index_add_(0, leaf[mask], w[mask])
    return total / weight.clamp_min(1e-12)


def assign_splats(tree, leaves, sample_leaf, config):
    """Splat id per sample: the leaf, or leaf x2 + side for edge leaves."""
    samples = tree.samples
    wanted = torch.cat([flag for _l, _c, flag in leaves])
    n_leaves = len(wanted)
    splat = sample_leaf.clone()
    edges = torch.zeros(n_leaves, dtype=torch.bool, device=wanted.device)
    if config.edge_split:
        moments = group_moments(samples, sample_leaf, n_leaves)
        direction, contrast = edge_directions(moments)
        edges = wanted & (contrast > config.edge_contrast) & (moments[0] > 0)
        if edges.any():
            _w, _mu, _cov, _n, color, cx = moments
            on_edge = edges[sample_leaf]
            idx = sample_leaf[on_edge]
            side = _colour_sides(samples["col"][on_edge], samples["w"][on_edge], idx, n_leaves,
                                 color, cx, direction)
            # Edge leaves take ids n_leaves.. so the first half keeps the leaf id.
            edge_rank = torch.cumsum(edges.long(), 0) - 1
            splat[on_edge] = torch.where(side, n_leaves + edge_rank[idx], idx)
    n_splats = n_leaves + int(edges.sum())
    is_edge = torch.zeros(n_splats, dtype=torch.bool, device=wanted.device)
    is_edge[:n_leaves] = edges
    is_edge[n_leaves:] = True
    return splat, n_splats, is_edge


def fit_shapes(tree, splat_of_sample, n_splats, is_edge, config):
    """Means, log-scales and quaternions of every splat."""
    samples = tree.samples
    dev = samples["pos"].device
    foot = torch.full((n_splats,), float("inf"), device=dev).scatter_reduce(
        0, splat_of_sample, samples["foot"], "amin")
    w, mu, cov, normal, color, _cx = group_moments(samples, splat_of_sample, n_splats)
    T = tangent_basis(normal)
    c2 = T.transpose(1, 2) @ cov @ T
    evals, evecs = torch.linalg.eigh(c2)                     # ascending
    axes = T @ evecs                                         # (N,3,2): minor, major
    minor, major = axes[:, :, 0], axes[:, :, 1]
    sig_min = config.min_splat_px * foot.clamp_max(1e3)
    cover_minor = torch.where(is_edge, torch.tensor(1.1, device=dev), torch.tensor(config.coverage, device=dev))
    s_major = torch.maximum(config.coverage * evals[:, 1].clamp_min(0).sqrt(), sig_min)
    s_minor = torch.maximum(cover_minor * evals[:, 0].clamp_min(0).sqrt(), sig_min)
    s_normal = (config.thickness * s_minor).clamp_min(1e-7)
    R = torch.stack([major, minor, normal], dim=-1)
    flip = torch.linalg.det(R) < 0
    R[flip, :, 1] *= -1
    return {
        "means": mu,
        "scales": torch.log(torch.stack([s_major, s_minor, s_normal], -1)),
        "quats": rotmat_to_quat(R),
        "color": color,
        "normal": normal,
        "weight": w,
    }
