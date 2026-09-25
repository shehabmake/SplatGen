"""Dataset loaders. Each loader exposes ``can_load(path)`` and ``load(path)``.

New input formats (for example the SplatGen raw dataset) are added by
registering another loader here; everything downstream consumes ``Scene``.
"""

from . import legacy
from .scene import Camera, Scene

LOADERS = {
    "splatgen-legacy": legacy,   # also reads plain COLMAP datasets
}


def load_scene(path, loader=None):
    if loader is not None:
        return LOADERS[loader].load(path)
    for module in LOADERS.values():
        if module.can_load(path):
            return module.load(path)
    raise FileNotFoundError(
        f"No supported dataset found in {path}. Pick a SplatGen build folder, "
        "its Dataset(Default) folder, or a COLMAP dataset folder.")


__all__ = ["Camera", "Scene", "load_scene", "LOADERS"]
