"""The compact ``.splat`` format (32 bytes per Gaussian, SH degree 0).

position 3xf32 | scale 3xf32 (linear) | rgba 4xu8 | rotation 4xu8
(quaternion w x y z mapped to 0..255). Sorted by size x opacity so viewers
that stream the file show the important splats first. Used by the built-in
web viewer and many web splat viewers.
"""

from pathlib import Path

import numpy as np
import torch

from ..model.sh import C0


def to_splat_bytes(params):
    p = {k: v.detach().cpu().float() for k, v in params.items()}
    scales = torch.exp(p["scales"])
    opacity = torch.sigmoid(p["opacities"])
    order = torch.argsort(-(scales.prod(dim=1) * opacity))
    rgb = (0.5 + C0 * p["sh0"].reshape(-1, 3)).clamp(0, 1)
    rgba = torch.cat([rgb, opacity[:, None]], dim=1)
    quats = torch.nn.functional.normalize(p["quats"], dim=-1)
    n = p["means"].shape[0]
    out = np.empty(n, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3),
                             ("rgba", "u1", 4), ("rot", "u1", 4)])
    out["pos"] = p["means"][order].numpy()
    out["scale"] = scales[order].numpy()
    out["rgba"] = (rgba[order] * 255).round().clamp(0, 255).byte().numpy()
    out["rot"] = (quats[order] * 128 + 128).round().clamp(0, 255).byte().numpy()
    return out.tobytes()


def write_splat(params, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_bytes(to_splat_bytes(params))
    pending.replace(path)
    return path
