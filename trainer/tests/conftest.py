"""A synthetic COLMAP dataset rendered from known Gaussians.

Eight cameras circle a cloud of coloured Gaussians; the images are rendered
with the PyTorch rasterizer, so a correct trainer must be able to fit them.
"""

import json
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


# -- a synthetic SplatGen build with raw data ------------------------------------------

RW, RH, RF = 96, 72, 90.0


def _texture(x, y):
    """Stripes, a red square and a plain border on the ground plane (z = 0)."""
    colour = np.full(x.shape + (3,), 0.85)
    stripes = (np.floor(x / 0.4) % 2 == 0) & (np.abs(y) < 0.8)
    colour[stripes] = 0.1
    square = (np.abs(x - 0.9) < 0.3) & (np.abs(y + 1.2) < 0.3)
    colour[square] = (0.8, 0.1, 0.1)
    return colour


def _write_exr(path, array):
    import OpenEXR
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(array.astype(np.float32))
    channels = {"Y": array} if array.ndim == 2 else {"RGB": array}
    with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, channels) as f:
        f.write(str(path))


@pytest.fixture(scope="session")
def raw_build(tmp_path_factory):
    """A build folder (Dataset(Default) + Dataset(Raw)) of a textured 4x4 plane."""
    pytest.importorskip("OpenEXR")
    build = tmp_path_factory.mktemp("SplatGen_test") / "2026-01-01_00-00-00"
    legacy = build / "Dataset(Default)"
    raw = build / "Dataset(Raw)"
    (legacy / "images").mkdir(parents=True)
    sky = np.array([0.3, 0.4, 0.6])
    frames, imgs = [], []
    v, u = np.mgrid[0:RH, 0:RW] + 0.5
    for i in range(12):
        ang = 2 * math.pi * i / 12
        eye = np.array([3.0 * math.cos(ang), 3.0 * math.sin(ang), 1.6 + 0.4 * (i % 2)])
        w2c = look_at_w2c(eye)
        c2w = np.linalg.inv(w2c)
        rays = np.stack([(u - RW / 2) / RF, (v - RH / 2) / RF, np.ones_like(u)], -1) @ c2w[:3, :3].T
        t = -eye[2] / np.where(rays[..., 2] < -1e-6, rays[..., 2], -1e-6)
        hit = eye + t[..., None] * rays
        on = (rays[..., 2] < -1e-6) & (np.abs(hit[..., 0]) < 2) & (np.abs(hit[..., 1]) < 2)
        # 4x supersampled colour, like a renderer's pixel filter
        colour = np.zeros((RH, RW, 3))
        for oy in (-0.375, -0.125, 0.125, 0.375):
            for ox in (-0.375, -0.125, 0.125, 0.375):
                r = np.stack([(u + ox - RW / 2) / RF, (v + oy - RH / 2) / RF, np.ones_like(u)], -1) @ c2w[:3, :3].T
                tt = -eye[2] / np.where(r[..., 2] < -1e-6, r[..., 2], -1e-6)
                h = eye + tt[..., None] * r
                inside = (r[..., 2] < -1e-6) & (np.abs(h[..., 0]) < 2) & (np.abs(h[..., 1]) < 2)
                colour += np.where(inside[..., None], _texture(h[..., 0], h[..., 1]), sky)
        colour /= 16
        stem = f"frame_{i:04d}"
        Image.fromarray((colour * 255).round().astype(np.uint8)).save(legacy / "images" / f"{stem}.png")
        depth = np.where(on, t, 1e10)                 # z-depth: rays have unit camera z
        _write_exr(raw / "geometry/depth" / f"{stem}_depth.exr", depth)
        _write_exr(raw / "geometry/position" / f"{stem}_position.exr", np.where(on[..., None], hit, 0))
        _write_exr(raw / "geometry/normal" / f"{stem}_normal.exr",
                   np.where(on[..., None], np.array([0.0, 0.0, 1.0]), 0))
        _write_exr(raw / "ids/object_id" / f"{stem}_object_id.exr", np.where(on, 1.0, 0.0))
        frames.append({"stem": stem, "camera_name": f"Cam{i}", "legacy_image": f"../Dataset(Default)/images/{stem}.png",
                       "width": RW, "height": RH, "fx": RF, "fy": RF, "cx": RW / 2, "cy": RH / 2,
                       "c2w_opencv": c2w.tolist()})
        q = rotmat_to_quat(w2c[:3, :3])
        imgs.append(f"{i + 1} {' '.join(f'{x:.10f}' for x in q)} {' '.join(f'{x:.10f}' for x in w2c[:3, 3])} 1 {stem}.png\n\n")
    (legacy / "cameras.txt").write_text(f"# c\n1 PINHOLE {RW} {RH} {RF} {RF} {RW / 2} {RH / 2}\n")
    (legacy / "images.txt").write_text("# i\n" + "".join(imgs))
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.uniform(-2, 2, (400, 2)), np.zeros((400, 1))], 1)
    (legacy / "points3D.txt").write_text("# p\n" + "".join(
        f"{i + 1} {p[0]} {p[1]} {p[2]} 128 128 128 0\n" for i, p in enumerate(pts)))
    (raw / "cameras").mkdir(parents=True, exist_ok=True)
    (raw / "cameras" / "cameras.json").write_text(json.dumps({"frames": frames}))
    passes = {key: {"path_pattern": f"{folder}/{key}/{{stem}}_{key}.exr", "frames_present": 12}
              for folder, key in (("geometry", "depth"), ("geometry", "position"),
                                  ("geometry", "normal"), ("ids", "object_id"))}
    (raw / "manifest.json").write_text(json.dumps({"format_version": 2, "passes": passes,
                                                   "resolution": [RW, RH]}))
    return build
