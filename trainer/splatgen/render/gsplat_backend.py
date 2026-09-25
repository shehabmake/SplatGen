"""gsplat CUDA rasterizer (https://github.com/nerfstudio-project/gsplat).

The fast path: requires an NVIDIA GPU and the gsplat package. Returns the
same (rgb, alpha, info) contract as the PyTorch backend so the trainer and
densification never know which one ran.
"""

NAME = "gsplat"


def available():
    try:
        import torch
        import gsplat  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


def version():
    try:
        import gsplat
        return getattr(gsplat, "__version__", "unknown")
    except Exception:
        return None


def rasterize(means, quats, scales, opacities, sh, sh_degree, viewmat, K,
              width, height, background):
    from gsplat import rasterization

    colors, alphas, meta = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=sh, viewmats=viewmat[None], Ks=K[None],
        width=width, height=height, sh_degree=sh_degree,
        backgrounds=background[None], packed=False, render_mode="RGB",
    )
    radii = meta["radii"][0]
    if radii.dim() == 2:          # gsplat >= 1.5: per-axis radii
        radii = radii.max(dim=-1).values
    means2d = meta["means2d"]     # [1, N, 2]; gradients flow into it
    return colors[0], alphas[0, ..., 0], {
        "means2d": means2d, "radii": radii, "width": width, "height": height,
        "batch_view": True,
    }
