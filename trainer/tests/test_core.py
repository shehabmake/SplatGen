import struct

import numpy as np
import pytest
import torch

from splatgen.config import TrainConfig, preset_config
from splatgen.data import colmap, load_scene
from splatgen.io import read_ply, to_splat_bytes, write_ply
from splatgen.model import GaussianModel, init_params
from splatgen.model.sh import C0, eval_sh
from splatgen.render import torch_backend
from splatgen.train.trainer import Trainer

from conftest import F, H, W, look_at_w2c


def test_scene_loads(dataset):
    scene = load_scene(dataset)
    assert len(scene.cameras) == 8
    cam = scene.cameras[0]
    assert (cam.width, cam.height, cam.fx, cam.cx) == (W, H, F, W / 2)
    expected = look_at_w2c((3.2, 0.0, 1.0))
    assert np.allclose(cam.world_to_camera, expected, atol=1e-6)
    assert len(scene.points_xyz) == 300
    assert scene.format == "colmap"


def test_binary_model_matches_text(dataset, tmp_path):
    model = colmap.read_model(dataset)
    with open(tmp_path / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, W, H) + struct.pack("<4d", F, F, W / 2, H / 2))
    with open(tmp_path / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(model.images)))
        for im in model.images:
            f.write(struct.pack("<i7di", im.id, *im.qvec, *im.tvec, im.camera_id))
            f.write(im.name.encode() + b"\x00" + struct.pack("<Q", 0))
    with open(tmp_path / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(model.points_xyz)))
        for i, (p, c) in enumerate(zip(model.points_xyz, model.points_rgb)):
            f.write(struct.pack("<Q3d3Bd", i + 1, *p, *c, 0.0) + struct.pack("<Q", 0))
    binary = colmap.read_model(tmp_path)
    assert [im.name for im in binary.images] == [im.name for im in model.images]
    assert np.allclose(binary.images[3].world_to_camera(), model.images[3].world_to_camera())
    assert np.allclose(binary.points_xyz, model.points_xyz)


def test_sh_degree_zero_is_dc():
    coeffs = torch.randn(5, 16, 3)
    dirs = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    assert torch.allclose(eval_sh(0, coeffs, dirs), C0 * coeffs[:, 0])


def test_single_gaussian_projects_to_principal_point():
    means = torch.tensor([[0.0, 0.0, 3.0]])
    params = {"means": means, "quats": torch.tensor([[1.0, 0, 0, 0]]),
              "scales": torch.full((1, 3), 0.05), "opacities": torch.tensor([0.8])}
    sh = torch.zeros(1, 1, 3)
    K = torch.tensor([[50.0, 0, 16], [0, 50.0, 12], [0, 0, 1]])
    rgb, alpha, info = torch_backend.rasterize(
        params["means"], params["quats"], params["scales"], params["opacities"], sh, 0,
        torch.eye(4), K, 32, 24, torch.zeros(3))
    assert torch.allclose(info["means2d"][0], torch.tensor([16.0, 12.0]))
    peak = torch.nonzero(alpha == alpha.max())[0]
    assert abs(int(peak[0]) - 11.5) <= 0.5 and abs(int(peak[1]) - 15.5) <= 0.5
    # Peak pixel centre is 0.5 px off in x and y; variance = (50 * 0.05 / 3)^2 + 0.3.
    var = (50 * 0.05 / 3) ** 2 + 0.3
    expected = 0.8 * np.exp(-0.5 * (0.25 + 0.25) / var)
    assert abs(float(alpha.max()) - expected) < 1e-4


def test_training_improves_held_out_views(dataset, tmp_path):
    torch.manual_seed(0)
    scene = load_scene(dataset)
    config = preset_config("preview", {"steps": 150, "downscale": 1, "test_every": 4,
                                       "densify_start": 40, "densify_stop": 120,
                                       "densify_every": 40, "log_every": 50})
    trainer = Trainer(scene, config, tmp_path, log=lambda m: None)
    trainer.setup()
    before = trainer.evaluate()["psnr"]
    assert trainer.run() == "finished"
    after = trainer.evaluate()["psnr"]
    assert after > before + 3.0, (before, after)
    for name, param in trainer.model.params.items():
        state = trainer.optimizers[name].state[param]
        assert state["exp_avg"].shape == param.shape, name


def test_densification_changes_count_and_keeps_adam_aligned(dataset, tmp_path):
    scene = load_scene(dataset)
    config = preset_config("preview", {"steps": 60, "downscale": 1, "densify_start": 10,
                                       "densify_every": 20, "densify_stop": 60,
                                       "densify_grad_threshold": 1e-7, "log_every": 1000})
    trainer = Trainer(scene, config, tmp_path, log=lambda m: None)
    trainer.setup()
    start = len(trainer.model)
    trainer.run()
    assert len(trainer.model) > start
    for name, param in trainer.model.params.items():
        assert trainer.optimizers[name].param_groups[0]["params"][0] is param
        assert trainer.optimizers[name].state[param]["exp_avg_sq"].shape == param.shape


def test_checkpoint_resume_continues(dataset, tmp_path):
    scene = load_scene(dataset)
    config = preset_config("preview", {"steps": 20, "log_every": 1000})
    trainer = Trainer(scene, config, tmp_path, log=lambda m: None)
    trainer.setup()
    trainer.run()
    path = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    config.steps = 30
    again = Trainer(scene, config, tmp_path, log=lambda m: None)
    again.setup(path)
    assert again.step == 20 and len(again.model) == len(trainer.model)
    again.run()
    assert again.step == 30


def test_ply_roundtrip_and_splat_bytes(tmp_path):
    params = init_params(np.random.rand(50, 3), np.random.rand(50, 3), sh_degree=3)
    params["quats"] = torch.nn.functional.normalize(torch.randn(50, 4), dim=-1)
    params["shN"] = torch.randn(50, 15, 3)
    write_ply(params, tmp_path / "a.ply")
    back = read_ply(tmp_path / "a.ply")
    for key in params:
        assert torch.allclose(back[key], params[key].float(), atol=1e-6), key
    header = (tmp_path / "a.ply").read_bytes()[:2000].decode("ascii", "ignore")
    assert "property float f_rest_44" in header and "property float rot_3" in header
    assert len(to_splat_bytes(params)) == 50 * 32


def test_init_from_ply(dataset, tmp_path):
    params = init_params(np.random.rand(40, 3), np.random.rand(40, 3), sh_degree=1)
    write_ply(params, tmp_path / "seed.ply")
    scene = load_scene(dataset)
    config = preset_config("preview", {"steps": 1, "init": "ply", "init_ply": str(tmp_path / "seed.ply"),
                                       "sh_degree": 3})
    trainer = Trainer(scene, config, tmp_path, log=lambda m: None)
    trainer.setup()
    assert len(trainer.model) == 40 and trainer.model.params["shN"].shape[1] == 15


def test_config_roundtrip():
    config = preset_config("high", {"steps": 123})
    assert TrainConfig.from_dict(config.to_dict()) == config
    assert TrainConfig.from_dict({"unknown": 1, "steps": 5}).steps == 5
