"""The 3D Gaussian Splatting training loop.

``Trainer`` is usable on its own (CLI, scripts, tests) and by the app's job
manager. It never blocks on the UI: progress goes out through a callback,
and ``stop`` / ``pause`` are threading events checked between steps. A lock
is held around each step so previews can render safely in between.
"""

import json
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .. import render as renderers
from ..config import TrainConfig
from ..model import GaussianModel, init_params, random_points
from . import losses
from .images import ImageCache
from .strategy import DensityStrategy


class Trainer:
    def __init__(self, scene, config=None, out_dir=None, on_progress=None, log=None):
        self.scene = scene
        self.cfg = config or TrainConfig()
        self.out_dir = Path(out_dir) if out_dir else None
        self.on_progress = on_progress
        self.log = log or (lambda message: print(f"[splatgen] {message}"))
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.step = 0
        self.metrics = []
        self.eval_results = {}
        self.model = None

    # -- setup ---------------------------------------------------------------

    def setup(self, checkpoint=None):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = renderers.pick_device(cfg.device)
        self.backend = renderers.pick_backend(cfg.backend, self.device)
        self.log(f"device {self.device}, renderer {self.backend.NAME}")
        self.extent = self.scene.extent()
        self.train_cams, self.test_cams = self.scene.split(cfg.test_every)
        self.images = ImageCache(self.scene.cameras, cfg.downscale,
                                 masks=cfg.mask_mode != "none", device=self.device)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.model = GaussianModel.from_state_dict(state["params"], self.device)
            self.step = int(state.get("step", 0))
            self.metrics = list(state.get("metrics", []))
        else:
            self.model = GaussianModel(self._initial_params(), self.device)
        self._make_optimizers()
        if checkpoint is not None and "optimizers" in state:
            for name, opt_state in state["optimizers"].items():
                try:
                    self.optimizers[name].load_state_dict(opt_state)
                except (ValueError, KeyError):
                    pass
        self.strategy = DensityStrategy(cfg, self.extent)
        self.strategy.reset_state(len(self.model), self.device)
        self._order = []
        self.log(f"{len(self.model):,} initial Gaussians, {len(self.train_cams)} training / "
                 f"{len(self.test_cams)} test views, scene extent {self.extent:.3f}")

    def _initial_params(self):
        cfg = self.cfg
        scene = self.scene
        if cfg.init == "ply" and cfg.init_ply:
            from ..io.ply import read_ply
            params = read_ply(cfg.init_ply)
            return _fit_sh_degree(params, cfg.sh_degree)
        if cfg.init == "points" and len(scene.points_xyz) > 0:
            xyz, rgb = scene.points_xyz, scene.points_rgb / 255.0
        else:
            centers = scene.camera_centers()
            center = centers.mean(axis=0) if len(centers) else np.zeros(3)
            xyz, rgb = random_points(cfg.init_random_count, center, self.extent, cfg.seed)
        return init_params(xyz, rgb, cfg.sh_degree, cfg.init_opacity, cfg.init_scale)

    def _make_optimizers(self):
        cfg = self.cfg
        rates = {
            "means": cfg.lr_means * self.extent, "scales": cfg.lr_scales,
            "quats": cfg.lr_quats, "opacities": cfg.lr_opacities,
            "sh0": cfg.lr_sh0, "shN": cfg.lr_shN,
        }
        self.base_lr = rates
        self.optimizers = {name: torch.optim.Adam([self.model.params[name]], lr=lr, eps=1e-15)
                           for name, lr in rates.items()}

    # -- one step ------------------------------------------------------------------

    def _next_camera(self):
        if not self._order:
            self._order = list(self.train_cams)
            random.shuffle(self._order)
        return self._order.pop()

    def _background(self):
        cfg = self.cfg
        if cfg.background == "white":
            return torch.ones(3, device=self.device)
        if cfg.background == "random":
            return torch.rand(3, device=self.device)
        return torch.zeros(3, device=self.device)

    def active_sh_degree(self, step=None):
        step = self.step if step is None else step
        return min(self.model.max_sh_degree, step // max(1, self.cfg.sh_degree_interval))

    def render(self, cam, background=None, size=None, sh_degree=None):
        a = self.model.activated()
        width, height = size or self.images.size(cam)
        scale = width / cam.width
        viewmat = torch.tensor(cam.world_to_camera, dtype=torch.float32, device=self.device)
        K = torch.tensor(cam.K(scale), dtype=torch.float32, device=self.device)
        bg = background if background is not None else torch.zeros(3, device=self.device)
        return self.backend.rasterize(
            a["means"], a["quats"], a["scales"], a["opacities"], a["sh"],
            self.active_sh_degree() if sh_degree is None else sh_degree,
            viewmat, K, width, height, bg)

    def train_step(self):
        cfg = self.cfg
        with self.lock:
            self.step += 1
            step = self.step
            cam = self._next_camera()
            gt, mask, size = self.images.get(cam)
            bg = self._background()
            rgb, _alpha, info = self.render(cam, bg, size)
            info["means2d"].retain_grad()
            if mask is not None and cfg.mask_mode == "ignore":
                weight = mask[..., None].float()
                l1 = (torch.abs(rgb - gt) * weight).sum() / (weight.sum() * 3).clamp_min(1)
                ssim_value = losses.ssim(rgb * weight, gt * weight)
            else:
                l1 = torch.abs(rgb - gt).mean()
                ssim_value = losses.ssim(rgb, gt)
            loss = (1.0 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1.0 - ssim_value)
            loss.backward()
            report = self.strategy.step(step, self.model, self.optimizers, info)
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # Exponential decay of the position learning rate.
            t = min(1.0, step / max(1, cfg.steps))
            self.optimizers["means"].param_groups[0]["lr"] = (
                self.base_lr["means"] * cfg.lr_means_final ** t)
            with torch.no_grad():
                train_psnr = losses.psnr(rgb.detach(), gt)
        return {"loss": float(loss.detach()), "l1": float(l1.detach()),
                "psnr": train_psnr, "report": report}

    # -- evaluation, preview ---------------------------------------------------

    @torch.no_grad()
    def evaluate(self, cams=None):
        cams = self.test_cams if cams is None else cams
        if not cams:
            return {}
        with self.lock:
            values = {"psnr": [], "ssim": [], "l1": []}
            bg = torch.zeros(3, device=self.device)
            if self.cfg.background == "white":
                bg = torch.ones(3, device=self.device)
            for cam in cams:
                gt, _mask, size = self.images.get(cam)
                rgb, _a, _i = self.render(cam, bg, size)
                rgb = rgb.clamp(0, 1)
                values["psnr"].append(losses.psnr(rgb, gt))
                values["ssim"].append(float(losses.ssim(rgb, gt)))
                values["l1"].append(float(torch.abs(rgb - gt).mean()))
        result = {key: float(np.mean(v)) for key, v in values.items()}
        result["views"] = len(cams)
        result["step"] = self.step
        self.eval_results = result
        return result

    @torch.no_grad()
    def preview(self, cam, max_size=960, background=None):
        """Render a dataset camera to a uint8 HxWx3 array (thread-safe)."""
        with self.lock:
            scale = min(1.0, max_size / max(cam.width, cam.height))
            size = (max(1, round(cam.width * scale)), max(1, round(cam.height * scale)))
            bg = background if background is not None else torch.zeros(3, device=self.device)
            rgb, _a, _i = self.render(cam, bg, size)
            return (rgb.clamp(0, 1) * 255).byte().cpu().numpy()

    # -- persistence -----------------------------------------------------------

    def checkpoint_state(self):
        with self.lock:
            return {
                "params": self.model.state_dict(),
                "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
                "step": self.step,
                "config": self.cfg.to_dict(),
                "metrics": self.metrics[-2000:],
                "scene_extent": self.extent,
            }

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp")
        torch.save(self.checkpoint_state(), pending)
        pending.replace(path)
        return path

    def export(self, fmt, path):
        from ..io import export_model
        with self.lock:
            return export_model(self.model, fmt, path)

    # -- the loop ---------------------------------------------------------------

    def run(self):
        cfg = self.cfg
        start = time.time()
        last_log = 0.0
        rate = None
        last_step_time = time.time()
        loss_avg = psnr_avg = None
        self.status = "training"
        while self.step < cfg.steps:
            if self.stop_event.is_set():
                self.status = "stopped"
                break
            if self.pause_event.is_set():
                self.status = "paused"
                time.sleep(0.1)
                last_step_time = time.time()
                continue
            self.status = "training"
            result = self.train_step()
            now = time.time()
            dt = now - last_step_time
            last_step_time = now
            rate = 1.0 / dt if rate is None else 0.95 * rate + 0.05 / max(dt, 1e-6)
            loss_avg = result["loss"] if loss_avg is None else 0.9 * loss_avg + 0.1 * result["loss"]
            psnr_avg = result["psnr"] if psnr_avg is None else 0.9 * psnr_avg + 0.1 * result["psnr"]
            if cfg.eval_every and self.step % cfg.eval_every == 0:
                self.evaluate()
            if cfg.checkpoint_every and self.step % cfg.checkpoint_every == 0 and self.out_dir:
                self.save_checkpoint(self.out_dir / "checkpoint.pt")
            if self.step % cfg.log_every == 0 or self.step == cfg.steps or now - last_log > 2.0:
                last_log = now
                entry = {
                    "step": self.step, "loss": round(loss_avg, 6), "psnr": round(psnr_avg, 3),
                    "gaussians": len(self.model), "sh_degree": self.active_sh_degree(),
                    "rate": round(rate, 3), "elapsed": round(now - start, 2),
                    "eta": round((cfg.steps - self.step) / max(rate, 1e-6), 1),
                }
                if result["report"]:
                    entry["densify"] = result["report"]
                self.metrics.append(entry)
                if self.on_progress:
                    self.on_progress(entry)
        else:
            self.status = "finished"
        return self.status


def _fit_sh_degree(params, degree):
    """Pad or cut higher-order SH so an imported PLY matches the config."""
    want = (degree + 1) ** 2 - 1
    shN = params["shN"]
    if shN.shape[1] > want:
        params["shN"] = shN[:, :want]
    elif shN.shape[1] < want:
        pad = torch.zeros(shN.shape[0], want - shN.shape[1], 3)
        params["shN"] = torch.cat([shN, pad], dim=1)
    return params


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, indent=2), encoding="utf-8")
    pending.replace(path)


