"""The in-memory scene a trainer consumes, independent of the file format."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    index: int
    name: str
    image_path: Path
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray       # 4x4, OpenCV axes (+Z forward, +Y down)
    mask_path: Path = None

    @property
    def camera_to_world(self):
        return np.linalg.inv(self.world_to_camera)

    @property
    def center(self):
        return self.camera_to_world[:3, 3]

    def K(self, scale=1.0):
        return np.array([[self.fx * scale, 0, self.cx * scale],
                         [0, self.fy * scale, self.cy * scale],
                         [0, 0, 1]], dtype=np.float64)

    def to_dict(self):
        return {
            "index": self.index, "name": self.name,
            "width": self.width, "height": self.height,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "camera_to_world": self.camera_to_world.tolist(),
            "has_mask": self.mask_path is not None,
        }


@dataclass
class Scene:
    """Cameras, their images and an optional sparse point cloud."""

    root: Path
    format: str                        # loader id, e.g. "splatgen-legacy"
    cameras: list
    points_xyz: np.ndarray
    points_rgb: np.ndarray
    up_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    warnings: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def camera_centers(self):
        return np.array([c.center for c in self.cameras]).reshape(-1, 3)

    def extent(self):
        """Radius of the camera rig, as in the 3DGS paper (x1.1)."""
        centers = self.camera_centers()
        if len(centers) == 0:
            return 1.0
        mid = centers.mean(axis=0)
        radius = float(np.linalg.norm(centers - mid, axis=1).max()) * 1.1
        return radius if radius > 1e-6 else 1.0

    def split(self, test_every):
        """(train, test) camera lists; every Nth camera is held out."""
        if not test_every or test_every < 2 or len(self.cameras) < 3:
            return list(self.cameras), []
        test = [c for i, c in enumerate(self.cameras) if i % test_every == 0]
        train = [c for i, c in enumerate(self.cameras) if i % test_every != 0]
        return train, test

    def summary(self):
        widths = sorted({c.width for c in self.cameras})
        heights = sorted({c.height for c in self.cameras})
        lo = self.points_xyz.min(axis=0).tolist() if len(self.points_xyz) else None
        hi = self.points_xyz.max(axis=0).tolist() if len(self.points_xyz) else None
        return {
            "root": str(self.root),
            "format": self.format,
            "camera_count": len(self.cameras),
            "point_count": int(len(self.points_xyz)),
            "resolution": f"{widths[0]}x{heights[0]}" if len(widths) == 1 and len(heights) == 1
                          else f"{widths[0]}-{widths[-1]} x {heights[0]}-{heights[-1]}" if widths else "",
            "masks": sum(1 for c in self.cameras if c.mask_path is not None),
            "extent": self.extent(),
            "bounds": [lo, hi],
            "up_axis": self.up_axis.tolist(),
            "warnings": list(self.warnings),
            "extras": dict(self.extras),
        }
