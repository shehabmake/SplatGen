"""Reader for the SplatGen raw dataset (``Dataset(Raw)``, see docs/RAW_DATASET.md).

Only per-view passes that are present are loaded; ``manifest.json`` gives
every pass's path pattern, so both the v1 (``frame_NNNN.exr``) and v2
(``frame_NNNN_<pass>.exr``) layouts work. Colour targets are the legacy
display images (``Dataset(Default)/images``) - the same pixels a standard
trainer fits - so constructed and trained splats are directly comparable.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

RAW_FOLDER = "Dataset(Raw)"


def find_root(path):
    """``Dataset(Raw)`` for a build folder, the raw folder itself, or a legacy dataset."""
    path = Path(path).expanduser()
    for candidate in (path, path / RAW_FOLDER, path.parent / RAW_FOLDER):
        if (candidate / "manifest.json").is_file() and (candidate / "cameras" / "cameras.json").is_file():
            return candidate
    return None


def read_exr(path):
    """Channels of a single-part EXR as float32 (H, W) or (H, W, C)."""
    try:
        import OpenEXR
    except ImportError as exc:
        raise RuntimeError("Reading the raw dataset needs the OpenEXR package: pip install OpenEXR") from exc
    with OpenEXR.File(str(path)) as handle:
        channels = handle.channels()
        if len(channels) == 1:
            return np.asarray(next(iter(channels.values())).pixels, dtype=np.float32)
        # Separate X/Y/Z style channels (v1 vector passes): stack in order.
        order = [k for k in ("R", "G", "B", "A", "X", "Y", "Z", "V") if k in channels]
        return np.stack([np.asarray(channels[k].pixels, dtype=np.float32) for k in order], axis=-1)


@dataclass
class RawView:
    index: int
    stem: str
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray        # 4x4 OpenCV camera-to-world
    image_path: Path

    @property
    def w2c(self):
        return np.linalg.inv(self.c2w)

    @property
    def center(self):
        return self.c2w[:3, 3]


class RawDataset:
    def __init__(self, path):
        root = find_root(path)
        if root is None:
            raise FileNotFoundError(f"No Dataset(Raw) found at {path}")
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        cameras = json.loads((root / "cameras" / "cameras.json").read_text(encoding="utf-8"))
        self.views = []
        for frame in cameras["frames"]:
            if "c2w_opencv" not in frame:
                continue
            image = (root / frame["legacy_image"]).resolve()
            self.views.append(RawView(
                index=len(self.views), stem=frame["stem"], name=frame["camera_name"],
                width=int(frame["width"]), height=int(frame["height"]),
                fx=float(frame["fx"]), fy=float(frame["fy"]),
                cx=float(frame["cx"]), cy=float(frame["cy"]),
                c2w=np.asarray(frame["c2w_opencv"], dtype=np.float64), image_path=image))
        self.passes = {k: v for k, v in self.manifest.get("passes", {}).items()
                       if v.get("frames_present", 0) > 0}

    def has(self, key):
        return key in self.passes

    def pass_path(self, key, view):
        return self.root / self.passes[key]["path_pattern"].replace("{stem}", view.stem)

    def load_pass(self, key, view):
        if not self.has(key):
            return None
        path = self.pass_path(key, view)
        return read_exr(path) if path.is_file() else None

    def load_image(self, view):
        with Image.open(view.image_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    def summary(self):
        return {"root": str(self.root), "views": len(self.views),
                "passes": sorted(self.passes), "resolution": self.manifest.get("resolution")}


def can_load(path):
    return find_root(path) is not None
