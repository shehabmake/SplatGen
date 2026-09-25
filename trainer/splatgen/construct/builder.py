"""The direct construction pipeline: raw data in, finished splats out.

    1. samples      every usable pixel of every training view -> surface point
    2. detail map   octree statistics (colour variation, curvature, object mix)
    3. split        detailed cells subdivide down to the pixel footprint
    4. shape        flat disks fitted to each cell's samples, two across edges
    5. colour       spherical harmonics solved per splat from all cameras
    6. check        render, compare with the images, correct colours and force
                    splits where the error stays high; repeat for a few rounds

No step uses gradient descent. Held-out views are never used to build and
give an honest score at the end; the result loads into the trainer as-is
(``init = "ply"``) when a short polish is wanted.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .. import render as renderers
from ..data.raw import RawDataset
from ..model.sh import C0, rgb_to_sh
from ..train import losses
from .background import build_background
from .color import fit_sh
from .config import ConstructConfig
from .octree import Octree
from .samples import collect
from .shape import assign_splats, fit_shapes


class Stopped(Exception):
    pass


def split_views(views, test_every):
    """Same rule as ``Scene.split``: every Nth view is held out."""
    if not test_every or test_every < 2 or len(views) < 3:
        return list(views), []
    return ([v for v in views if v.index % test_every != 0],
            [v for v in views if v.index % test_every == 0])


def spread(items, count):
    if count <= 0 or len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


class Builder:
    def __init__(self, raw_path, config=None, out_dir=None, on_progress=None, log=None, stop_event=None):
        self.cfg = config or ConstructConfig()
        self.raw = RawDataset(raw_path)
        self.out_dir = Path(out_dir) if out_dir else None
        self.on_progress = on_progress
        self.log = log or (lambda message: print(f"[splatgen] {message}"))
        self.stop_event = stop_event
        self.params = None
        self.report = {"rounds": []}
        self._images = {}

    # -- helpers -------------------------------------------------------------------

    def _check_stop(self):
        if self.stop_event is not None and self.stop_event.is_set():
            raise Stopped()

    def _progress(self, stage, fraction, **extra):
        if self.on_progress:
            self.on_progress({"stage": stage, "progress": round(float(fraction), 4), **extra})

    def image(self, view):
        if view.index not in self._images:
            self._images[view.index] = torch.from_numpy(self.raw.load_image(view)).to(self.device)
        return self._images[view.index]

    def render(self, view, params=None, sh0=None):
        """Render a dataset view; pass ``sh0`` (requiring grad) to differentiate
        the image with respect to the base colours."""
        p = params or self.params
        viewmat = torch.tensor(view.w2c, dtype=torch.float32, device=self.device)
        K = torch.tensor([[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]],
                         dtype=torch.float32, device=self.device)
        sh = torch.cat([p["sh0"] if sh0 is None else sh0, p["shN"]], dim=1)
        with torch.set_grad_enabled(sh0 is not None):
            rgb, _alpha, _info = self.backend.rasterize(
                p["means"], torch.nn.functional.normalize(p["quats"], dim=-1), torch.exp(p["scales"]),
                torch.sigmoid(p["opacities"]), sh, self.cfg.sh_degree, viewmat, K,
                view.width, view.height, torch.zeros(3, device=self.device))
        return rgb if sh0 is not None else rgb.clamp(0, 1)

    # -- the pipeline ------------------------------------------------------------------

    def run(self):
        cfg = self.cfg
        t0 = time.time()
        self.device = renderers.pick_device(cfg.device)
        self.backend = renderers.pick_backend(cfg.backend, self.device)
        self.log(f"device {self.device}, renderer {self.backend.NAME}")
        self.train_views, self.test_views = split_views(self.raw.views, cfg.test_every)
        self.log(f"{len(self.train_views)} build views, {len(self.test_views)} held out")

        self._progress("samples", 0.0)
        samples, info = collect(self.raw, self.train_views, cfg, self.device, self.log)
        self._check_stop()
        self.samples = samples
        self.report["samples"] = {"count": len(samples["pos"]), "stride": info["stride"]}
        self.view_centers = torch.tensor(np.array([v.center for v in self.raw.views]),
                                         dtype=torch.float32, device=self.device)
        # Samples are in view order: remember each view's slice for the checks.
        counts = torch.bincount(samples["view"].long(), minlength=len(self.raw.views)).cpu()
        ends = torch.cumsum(counts, 0)
        self.view_slices = {i: slice(int(ends[i] - counts[i]), int(ends[i])) for i in range(len(counts))}

        self.background = None
        if cfg.background:
            centers = np.array([v.center for v in self.raw.views])
            center = centers.mean(axis=0)
            pos = samples["pos"]
            reach = float((pos - torch.tensor(center, dtype=torch.float32, device=self.device)).norm(dim=1).max())
            cams = float(np.linalg.norm(centers - center, axis=1).max())
            self.background = build_background(self.raw, self.train_views, center,
                                               3.0 * max(reach, cams, 1e-3), cfg, self.device, self.log)
        self._check_stop()

        self._progress("detail", 0.05)
        tree = Octree(samples, cfg, self.log)
        self._check_stop()

        check = spread(self.train_views, cfg.check_views)
        rounds = max(1, int(cfg.rounds))
        forced = {}
        for r in range(rounds):
            base = 0.1 + 0.8 * r / rounds
            self._progress("split", base, round=r + 1, rounds=rounds)
            leaves = tree.select(forced)
            if cfg.balance:
                leaves = tree.balance(leaves)
            sample_leaf = tree.leaf_of_samples(leaves)
            splat, n_splats, is_edge = assign_splats(tree, leaves, sample_leaf, cfg)
            self._check_stop()
            self._progress("shape", base + 0.1 / rounds, round=r + 1, rounds=rounds)
            shapes = fit_shapes(tree, splat, n_splats, is_edge, cfg)
            self._progress("colour", base + 0.2 / rounds, round=r + 1, rounds=rounds)
            coeffs, _seen = fit_sh(samples, splat, n_splats, shapes["means"], self.view_centers, cfg)
            self.params = self._assemble(shapes, coeffs)
            self.splat_of_sample = splat
            self._check_stop()

            # Check: render, correct colours directly, measure what is left.
            self._progress("check", base + 0.4 / rounds, round=r + 1, rounds=rounds)
            error, psnr = self._check_and_correct(check, passes=max(1, int(cfg.correction_passes)))
            n_leaves = sum(len(c) for _l, c, _f in leaves)
            leaf_error = torch.zeros(n_leaves, device=self.device).scatter_reduce_(
                0, sample_leaf, error, "amax")
            entry = {"round": r + 1, "splats": len(self.params["means"]), "edge_splats": int(is_edge.sum()),
                     "leaves": n_leaves, "check_psnr": round(psnr, 3),
                     "high_error_leaves": int((leaf_error > cfg.error_threshold).sum()),
                     "elapsed": round(time.time() - t0, 2)}
            self.report["rounds"].append(entry)
            self.log(f"round {r + 1}/{rounds}: {entry['splats']:,} splats ({entry['edge_splats']:,} on edges), "
                     f"check PSNR {psnr:.2f} dB, {entry['high_error_leaves']:,} cells over the error limit")
            self._progress("check", base + 0.8 / rounds, round=r + 1, rounds=rounds, splats=entry["splats"],
                           psnr=entry["check_psnr"])
            if r + 1 < rounds:
                offset = 0
                added = 0
                for level, cells, _flag in leaves:
                    bad = leaf_error[offset:offset + len(cells)] > cfg.error_threshold
                    offset += len(cells)
                    if bad.any():
                        keys = tree.cells[level]["keys"][cells[bad]]
                        forced[level] = torch.unique(torch.cat([forced[level], keys])) if level in forced else keys
                        added += int(bad.sum())
                if added == 0:
                    self.log("no cell left over the error limit - stopping early")
                    break
            self._check_stop()

        self._progress("evaluate", 0.95)
        self.report["evaluation"] = self.evaluate()
        self.report["time"] = round(time.time() - t0, 2)
        self.report["splats"] = len(self.params["means"])
        self.report["config"] = cfg.to_dict()
        if self.report["evaluation"]:
            e = self.report["evaluation"]
            self.log(f"held-out views: PSNR {e['psnr']:.2f} dB, SSIM {e['ssim']:.4f}")
        self.log(f"built {self.report['splats']:,} splats in {self.report['time']:.1f} s")
        self._progress("done", 1.0, splats=self.report["splats"])
        return self.report

    def _assemble(self, shapes, coeffs):
        cfg = self.cfg
        n = len(shapes["means"])
        opacity = torch.logit(torch.tensor(cfg.opacity, device=self.device)).expand(n).clone()
        params = {
            "means": shapes["means"], "scales": shapes["scales"], "quats": shapes["quats"],
            "opacities": opacity, "sh0": coeffs[:, :1].clone(), "shN": coeffs[:, 1:].clone(),
        }
        self.n_surface = n
        bg = self.background
        if bg is not None:
            m = len(bg["means"])
            K = coeffs.shape[1]
            params["means"] = torch.cat([params["means"], bg["means"]])
            params["scales"] = torch.cat([params["scales"], bg["scales"]])
            params["quats"] = torch.cat([params["quats"], bg["quats"]])
            params["opacities"] = torch.cat([params["opacities"], torch.full((m,), 6.0, device=self.device)])
            params["sh0"] = torch.cat([params["sh0"], rgb_to_sh(bg["color"])[:, None]])
            params["shN"] = torch.cat([params["shN"], torch.zeros(m, K - 1, 3, device=self.device)])
        return params

    def _check_and_correct(self, views, passes=2):
        """Direct colour solve on the rendered images.

        With the geometry fixed, every pixel is a fixed blend of splat colours:
        ``image = W c``. The base colours are refined by solving the linear
        least-squares problem ``min |W c - gt|^2`` over the check views with a
        Jacobi-preconditioned iteration

            c <- c + a * (W^T r) / (W^T 1),     r = gt - W c

        where both products come from one backward pass through the renderer
        (the blend weights W never change, so ``W^T 1`` is computed once).
        It converges for 0 < a < 2 because the blend weights of a pixel sum to
        at most one. Returns the per-sample error after the last pass and the
        check PSNR."""
        n = len(self.params["means"])
        dev = self.device
        alpha = float(self.cfg.color_correction)
        # W^T 1: the total blend weight of every splat over the check views.
        weight = torch.zeros(n, device=dev)
        for view in views:
            self._check_stop()
            sh0 = self.params["sh0"].detach().clone().requires_grad_(True)
            rgb = self.render(view, sh0=sh0)
            rgb[..., 0].sum().backward()
            weight += sh0.grad[:, 0, 0] / C0
        error = torch.zeros(len(self.samples["pos"]), device=dev)
        psnr = 0.0
        for p in range(passes + 1):
            last = p == passes
            step = torch.zeros(n, 3, device=dev)
            mse = []
            for view in views:
                self._check_stop()
                gt = self.image(view)
                sh0 = self.params["sh0"].detach().clone().requires_grad_(not last)
                if last:
                    rgb = self.render(view)
                else:
                    raw = self.render(view, sh0=sh0)
                    residual = (gt - raw).detach()
                    (raw * residual).sum().backward()
                    step += sh0.grad[:, 0] / C0
                    rgb = raw.detach().clamp(0, 1)
                mse.append(float(((rgb - gt) ** 2).mean()))
                if last:
                    sl = self.view_slices.get(view.index, slice(0, 0))
                    pix = self.samples["pix"][sl]
                    error[sl] = (gt.reshape(-1, 3)[pix] - rgb.reshape(-1, 3)[pix]).abs().mean(-1)
            psnr = float(np.mean([-10 * math.log10(max(m, 1e-10)) for m in mse]))
            if not last:
                self.params["sh0"][:, 0] += alpha * step / weight.clamp_min(1e-6)[:, None] / C0
        return error, psnr

    @torch.no_grad()
    def evaluate(self, views=None):
        views = self.test_views if views is None else views
        if not views:
            return {}
        values = {"psnr": [], "ssim": [], "l1": []}
        for view in views:
            gt = self.image(view)
            rgb = self.render(view)
            values["psnr"].append(losses.psnr(rgb, gt))
            values["ssim"].append(float(losses.ssim(rgb, gt)))
            values["l1"].append(float(torch.abs(rgb - gt).mean()))
        result = {k: float(np.mean(v)) for k, v in values.items()}
        result["views"] = len(views)
        return result

    # -- output -------------------------------------------------------------------------

    def state_dict(self):
        return {k: v.detach().float().cpu() for k, v in self.params.items()}

    def save(self, out_dir=None, formats=("ply",)):
        from ..io import export_model
        out = Path(out_dir or self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = {}
        state = self.state_dict()
        for fmt in formats:
            path = out / ("point_cloud.ply" if fmt == "ply" else f"model.{fmt}")
            export_model(state, fmt, path)
            written[fmt] = str(path)
        (out / "construct.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        return written


def build_splats(raw_path, config=None, out_dir=None, on_progress=None, log=None, stop_event=None,
                 formats=("ply",)):
    """Run the whole construction; returns (builder, report)."""
    if isinstance(config, dict):
        config = ConstructConfig.from_dict(config)
    builder = Builder(raw_path, config, out_dir, on_progress, log, stop_event)
    report = builder.run()
    if out_dir:
        report["files"] = builder.save(out_dir, formats)
    return builder, report


def polish_config(steps, ply_path, test_every=8, overrides=None, sh_degree=3):
    """Training settings for a short polish that starts from constructed splats.

    Positions are already on the surface, so they move slowly; every SH band
    is active from the first step; densification runs in a short window and
    opacity is never reset (that would throw away the construction)."""
    from ..config import TrainConfig

    steps = int(steps)
    data = TrainConfig().to_dict()
    data.update({
        "steps": steps, "init": "ply", "init_ply": str(ply_path), "test_every": test_every,
        "sh_degree": sh_degree, "sh_degree_interval": 1,
        "lr_means": 4e-5, "lr_means_final": 0.1,
        "densify_start": min(100, steps), "densify_stop": int(steps * 0.6),
        "densify_every": 100, "reset_opacity_every": 10 ** 9,
    })
    data.update(overrides or {})
    return TrainConfig.from_dict(data)
