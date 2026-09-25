"""Step 1 - surface samples: every usable pixel becomes a point on the surface.

A sample carries position, shading normal, colour (from the display image
the trainer would fit), a weight, the pixel footprint (world size of one
pixel there), the view it came from, object / material ids and roughness.

Pixels whose values are blends of two surfaces are dropped: silhouettes
(a neighbour is background or far away along the surface) and id borders.
"""

import math

import numpy as np
import torch

SKY = 1e9


def _neighbours_ok(position, valid, ids, footprint, ratio):
    """Pixels whose 4 neighbours lie on the same continuous surface."""
    ok = valid.copy()
    H, W = valid.shape
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ys = slice(max(0, dy), H + min(0, dy))
        yn = slice(max(0, -dy), H + min(0, -dy))
        xs = slice(max(0, dx), W + min(0, dx))
        xn = slice(max(0, -dx), W + min(0, -dx))
        here = np.ones_like(valid)       # no neighbour this way (image border): no objection
        there_valid = valid[ys, xs]
        gap = np.linalg.norm(position[ys, xs] - position[yn, xn], axis=-1)
        same = there_valid & (gap < ratio * footprint[yn, xn])
        if ids is not None:
            same &= ids[ys, xs] == ids[yn, xn]
        here[yn, xn] = same
        ok &= here
    return ok


def _position_normals(position):
    """Normals from the cross product of position differences (central, one-sided at borders)."""
    du = np.gradient(position, axis=1)
    dv = np.gradient(position, axis=0)
    n = np.cross(du, dv)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def _surface_normals(raw, view, position, valid):
    """Best available normal per pixel: the geometric (true) normal AOV, then
    Cycles' shading normal, then normals derived from the position pass.

    AOVs are only written by materials the exporter could extend; elsewhere the
    AOV reads as zero or partial coverage, so each source is used only where it
    is a unit vector."""
    out = np.zeros(position.shape[:2] + (3,))
    done = np.zeros(position.shape[:2], bool)
    for key in ("true_normal", "normal"):
        data = raw.load_pass(key, view)
        if data is None or data.shape[:2] != out.shape[:2]:
            continue
        n = data[..., :3].astype(np.float64)
        good = np.abs(np.linalg.norm(n, axis=-1) - 1) < 0.2
        if data.shape[-1] == 4:
            good &= data[..., 3] > 0.99
        take = good & ~done
        out[take] = n[take]
        done |= take
    rest = valid & ~done
    if rest.any():
        out[rest] = _position_normals(position)[rest]
    return out


def view_samples(raw, view, config, stride=1):
    """Samples of one view as a dict of numpy arrays (N, ...)."""
    depth = raw.load_pass("depth", view)
    if depth is None:
        raise RuntimeError(f"{view.stem}: the raw dataset has no depth pass")
    image = raw.load_image(view)
    H, W = depth.shape
    if image.shape[:2] != (H, W):
        raise RuntimeError(f"{view.stem}: image and depth sizes differ")
    valid = np.isfinite(depth) & (depth < SKY * 0.5) & (depth > 0)
    c2w = view.c2w
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    position = raw.load_pass("position", view)
    if position is None or position.shape[:2] != (H, W):
        cam = np.stack([(u + 0.5 - view.cx) / view.fx * depth, (v + 0.5 - view.cy) / view.fy * depth, depth], -1)
        position = cam @ c2w[:3, :3].T + c2w[:3, 3]
    position = position[..., :3].astype(np.float64)
    normal = _surface_normals(raw, view, position, valid)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    valid &= length[..., 0] > 0.5
    normal = normal / np.maximum(length, 1e-8)
    ids = raw.load_pass("object_id", view)
    materials = raw.load_pass("material_id", view)
    roughness = raw.load_pass("roughness", view)

    to_cam = c2w[:3, 3] - position
    dist = np.linalg.norm(to_cam, axis=-1)
    to_cam /= np.maximum(dist[..., None], 1e-9)
    cos = np.abs((normal * to_cam).sum(-1))
    # Face the normals towards the camera that saw them.
    flip = (normal * to_cam).sum(-1) < 0
    normal[flip] *= -1
    footprint = np.where(valid, depth, 0) / view.fx
    keep = _neighbours_ok(position, valid, ids, np.maximum(footprint, 1e-9) / np.maximum(cos, 0.1), config.edge_px_ratio)
    keep &= cos > 0.05
    if stride > 1:
        grid = np.zeros_like(keep)
        oy, ox = (view.index * 7) % stride, (view.index * 3) % stride
        grid[oy::stride, ox::stride] = True
        keep &= grid
    idx = np.flatnonzero(keep.reshape(-1))
    flat = lambda a: a.reshape(-1, *a.shape[2:])[idx]
    f_iso = footprint / np.sqrt(np.maximum(cos, 0.1))
    out = {
        "pos": flat(position).astype(np.float32),
        "nrm": flat(normal).astype(np.float32),
        "col": flat(image).astype(np.float32),
        "foot": flat(f_iso).astype(np.float32) * stride,
        "cos": flat(cos).astype(np.float32),
        "view": np.full(len(idx), view.index, np.int32),
        "pix": idx.astype(np.int64),
        "obj": (flat(ids).astype(np.int32) if ids is not None else np.zeros(len(idx), np.int32)),
        "mat": (flat(materials).astype(np.int32) if materials is not None else np.zeros(len(idx), np.int32)),
        "rough": (flat(roughness).astype(np.float32) if roughness is not None else np.full(len(idx), 0.5, np.float32)),
    }
    return out, {"valid": int(valid.sum()), "kept": int(len(idx)), "height": H, "width": W}


def collect(raw, views, config, device, log=print):
    """All samples of the given views, concatenated as torch tensors."""
    total_px = sum(v.width * v.height for v in views)
    stride = max(1, math.ceil(math.sqrt(total_px * 0.7 / max(1, config.max_samples))))
    parts, stats = [], []
    for view in views:
        data, info = view_samples(raw, view, config, stride)
        parts.append(data)
        stats.append(info)
    samples = {k: torch.from_numpy(np.concatenate([p[k] for p in parts])).to(device) for k in parts[0]}
    # Colour weight: frontal and close observations resolve the surface best.
    ref = torch.quantile(samples["foot"][: 1_000_000].float(), 0.5).clamp_min(1e-9)
    rel = (samples["foot"] / ref).clamp(0.25, 16.0)
    samples["w"] = samples["cos"].clamp_min(0.05) / (rel * rel)
    log(f"{len(samples['pos']):,} surface samples from {len(views)} views (pixel stride {stride})")
    return samples, {"stride": stride, "views": stats}
