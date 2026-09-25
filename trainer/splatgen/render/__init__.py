"""Rasterizer backends behind one call: ``rasterize(...) -> (rgb, alpha, info)``.

``info["means2d"]`` carries the screen-space means whose gradient drives
densification, and ``info["radii"]`` (N,) marks visible Gaussians (> 0).
"""

import torch

from . import gsplat_backend, torch_backend

BACKENDS = {"gsplat": gsplat_backend, "torch": torch_backend}


def pick_device(requested="auto"):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_backend(requested="auto", device=None):
    if requested and requested != "auto":
        backend = BACKENDS[requested]
        if not backend.available():
            raise RuntimeError(f"Renderer '{requested}' is not available on this machine")
        return backend
    if device is not None and torch.device(device).type == "cuda" and gsplat_backend.available():
        return gsplat_backend
    return torch_backend


def describe():
    cuda = torch.cuda.is_available()
    return {
        "torch": torch.__version__,
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "gsplat": gsplat_backend.version(),
        "backends": {name: module.available() for name, module in BACKENDS.items()},
        "default_device": str(pick_device()),
        "default_backend": pick_backend("auto", pick_device()).NAME,
    }
