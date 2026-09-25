"""A synthetic COLMAP dataset rendered from known Gaussians.

Eight cameras circle a cloud of coloured Gaussians; the images are rendered
with the PyTorch rasterizer, so a correct trainer must be able to fit them.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from splatgen.model import GaussianModel
from splatgen.render import torch_backend

W, H, F = 48, 36, 40.0


def look_at_w2c(eye, target=(0, 0, 0), up=(0, 0, 1)):
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = -R @ eye
    return m


def rotmat_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def gt_params(n=300, seed=1):
    g = torch.Generator().manual_seed(seed)
    means = (torch.rand(n, 3, generator=g) - 0.5) * 1.6
    means[:, 2] *= 0.5
    return {
        "means": means,
        "scales": torch.log(torch.full((n, 3), 0.09)),
        "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "opacities": torch.full((n,), 2.0),
        "sh0": ((torch.rand(n, 3, generator=g) - 0.5) / 0.2820948).unsqueeze(1),
        "shN": torch.zeros(n, 0, 3),
    }


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("colmap")
    (root / "images").mkdir()
    model = GaussianModel(gt_params())
    a = model.activated()
    cams, imgs = [], []
    for i in range(8):
        ang = 2 * math.pi * i / 8
        eye = (3.2 * math.cos(ang), 3.2 * math.sin(ang), 1.0)
        w2c = look_at_w2c(eye)
        K = torch.tensor([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], dtype=torch.float32)
        with torch.no_grad():
            rgb, _al, _info = torch_backend.rasterize(
                a["means"], a["quats"], a["scales"], a["opacities"], a["sh"], 0,
                torch.tensor(w2c, dtype=torch.float32), K, W, H, torch.zeros(3))
        name = f"view_{i:02d}.png"
        Image.fromarray((rgb.clamp(0, 1) * 255).byte().numpy()).save(root / "images" / name)
        q = rotmat_to_quat(w2c[:3, :3])
        t = w2c[:3, 3]
        imgs.append(f"{i + 1} {' '.join(f'{v:.10f}' for v in q)} {' '.join(f'{v:.10f}' for v in t)} 1 {name}\n\n")
    (root / "cameras.txt").write_text(f"# c\n1 PINHOLE {W} {H} {F} {F} {W / 2} {H / 2}\n")
    (root / "images.txt").write_text("# i\n" + "".join(imgs))
    rng = np.random.default_rng(0)
    pts = model.params["means"].detach().numpy() + rng.normal(0, 0.05, (300, 3))
    lines = [f"{i + 1} {p[0]} {p[1]} {p[2]} 128 128 128 0\n" for i, p in enumerate(pts)]
    (root / "points3D.txt").write_text("# p\n" + "".join(lines))
    return Path(root)
