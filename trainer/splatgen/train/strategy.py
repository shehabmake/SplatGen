"""Adaptive density control from the 3DGS paper.

Every ``densify_every`` steps between ``densify_start`` and ``densify_stop``:

* Gaussians whose average screen-space position gradient exceeds the
  threshold grow: small ones are cloned, large ones are split in two.
* Nearly transparent Gaussians are pruned (and, after the first opacity
  reset, ones that grew too large in world space).

Every ``reset_opacity_every`` steps opacities are clamped low so the
optimiser has to re-earn them, which removes floaters.

Works with any renderer that returns ``info["means2d"]`` (pixels, with its
gradient retained) and ``info["radii"]``.
"""

import torch

from ..model.gaussians import PARAM_NAMES
from ..render.torch_backend import quat_to_rotmat


class DensityStrategy:
    def __init__(self, config, scene_extent):
        self.cfg = config
        self.extent = float(scene_extent)
        self.grad2d = None
        self.count = None

    # -- bookkeeping ----------------------------------------------------------

    def reset_state(self, n, device):
        self.grad2d = torch.zeros(n, device=device)
        self.count = torch.zeros(n, device=device)

    def accumulate(self, info):
        means2d = info["means2d"]
        if means2d.grad is None:
            return
        grads = means2d.grad.detach()
        if grads.dim() == 3:
            grads = grads[0]
        grads = grads * torch.tensor([info["width"] / 2.0, info["height"] / 2.0],
                                     device=grads.device)
        radii = info["radii"]
        visible = radii > 0
        n = grads.shape[0]
        if self.grad2d is None or self.grad2d.shape[0] != n:
            self.reset_state(n, grads.device)
        self.grad2d[visible] += grads[visible].norm(dim=-1)
        self.count[visible] += 1

    # -- the step hook ---------------------------------------------------------

    def step(self, step, model, optimizers, info):
        """Call after backward and before the optimiser step."""
        cfg = self.cfg
        report = {}
        if step >= cfg.densify_stop:
            return report
        self.accumulate(info)
        if step > cfg.densify_start and step % cfg.densify_every == 0:
            report["cloned"], report["split"] = self._grow(model, optimizers)
            report["pruned"] = self._prune(model, optimizers, step)
            self.reset_state(len(model), model.device)
        if cfg.reset_opacity_every and step > 0 and step % cfg.reset_opacity_every == 0:
            self._reset_opacity(model, optimizers)
            report["opacity_reset"] = True
        return report

    def _grow(self, model, optimizers):
        cfg = self.cfg
        p = model.params
        avg = self.grad2d / self.count.clamp_min(1)
        high = avg > cfg.densify_grad_threshold
        if cfg.max_gaussians:
            budget = cfg.max_gaussians - len(model)
            if budget <= 0:
                return 0, 0
            if int(high.sum()) > budget:
                # Keep only the strongest gradients within the budget.
                threshold = avg[high].topk(budget).values.min()
                high = high & (avg >= threshold)
        scale_max = torch.exp(p["scales"].detach()).max(dim=-1).values
        small = scale_max <= cfg.grow_scale3d * self.extent
        clone = high & small
        split = high & ~small
        n_clone, n_split = int(clone.sum()), int(split.sum())
        if n_clone:
            _duplicate(model, optimizers, clone)
            split = torch.cat([split, torch.zeros(n_clone, dtype=torch.bool, device=split.device)])
        if n_split:
            _split(model, optimizers, split)
        return n_clone, n_split

    def _prune(self, model, optimizers, step):
        cfg = self.cfg
        p = model.params
        prune = torch.sigmoid(p["opacities"].detach()) < cfg.prune_opacity
        if cfg.reset_opacity_every and step > cfg.reset_opacity_every:
            too_big = torch.exp(p["scales"].detach()).max(dim=-1).values > cfg.prune_scale3d * self.extent
            prune = prune | too_big
        n = int(prune.sum())
        if n and n < len(model):
            _remove(model, optimizers, prune)
        return n

    def _reset_opacity(self, model, optimizers):
        value = torch.logit(torch.tensor(self.cfg.prune_opacity * 2.0)).item()
        opacities = model.params["opacities"]
        _replace(model, optimizers, "opacities",
                 opacities.detach().clamp_max(value),
                 lambda state: torch.zeros_like(state))


# -- parameter surgery that keeps Adam's moments aligned ----------------------

def _replace(model, optimizers, name, new_value, state_fn):
    old = model.params[name]
    new = torch.nn.Parameter(new_value.contiguous())
    optimizer = optimizers[name]
    state = optimizer.state.pop(old, None)
    optimizer.param_groups[0]["params"] = [new]
    if state is not None:
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key] = state_fn(state[key])
        optimizer.state[new] = state
    model.params[name] = new


def _duplicate(model, optimizers, mask):
    for name in PARAM_NAMES:
        value = model.params[name].detach()
        _replace(model, optimizers, name, torch.cat([value, value[mask]]),
                 lambda s, mask=mask: torch.cat([s, torch.zeros_like(s[mask])]))


def _split(model, optimizers, mask, samples=2):
    p = {name: model.params[name].detach() for name in PARAM_NAMES}
    scales = torch.exp(p["scales"][mask])
    quats = torch.nn.functional.normalize(p["quats"][mask], dim=-1)
    rot = quat_to_rotmat(quats)
    noise = torch.randn(samples, *scales.shape, device=scales.device)
    offsets = torch.einsum("nij,snj->sni", rot, scales[None] * noise)
    new = {
        "means": (p["means"][mask][None] + offsets).reshape(-1, 3),
        "scales": torch.log(scales / 1.6).repeat(samples, 1),
    }
    keep = ~mask
    n_new = int(mask.sum()) * samples
    for name in PARAM_NAMES:
        rows = new.get(name)
        if rows is None:
            rows = p[name][mask].repeat(samples, *([1] * (p[name].dim() - 1)))
        _replace(model, optimizers, name, torch.cat([p[name][keep], rows]),
                 lambda s, keep=keep, n_new=n_new: torch.cat(
                     [s[keep], torch.zeros((n_new,) + s.shape[1:], device=s.device, dtype=s.dtype)]))


def _remove(model, optimizers, mask):
    keep = ~mask
    for name in PARAM_NAMES:
        _replace(model, optimizers, name, model.params[name].detach()[keep],
                 lambda s, keep=keep: s[keep])
