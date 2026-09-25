"""Loader for SplatGen legacy builds and plain COLMAP datasets.

Accepted inputs (the first match wins):

* a SplatGen build folder ``<timestamp>/`` containing ``Dataset(Default)``
* ``Dataset(Default)`` itself (``images/``, ``masks/``, COLMAP text files)
* any COLMAP dataset: ``images/`` next to ``cameras/images/points3D`` in the
  folder, in ``sparse/0`` or in ``sparse`` (text or binary)
"""

from pathlib import Path

import numpy as np

from . import colmap
from .scene import Camera, Scene

LEGACY_FOLDER = "Dataset(Default)"
RAW_FOLDER = "Dataset(Raw)"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".bmp", ".webp")


def resolve(path):
    """(dataset folder, build folder or None) for any accepted input."""
    path = Path(path).expanduser()
    if (path / LEGACY_FOLDER).is_dir():
        return path / LEGACY_FOLDER, path
    if path.name == LEGACY_FOLDER:
        return path, path.parent
    if path.name.lower() in {"images", "sparse", "masks"}:
        path = path.parent
    return path, None


def can_load(path):
    folder, _build = resolve(path)
    model_folder, _ext = colmap.find_model(folder)
    return model_folder is not None


def _find_image(folder, name):
    for base in (folder / "images", folder):
        candidate = base / name
        if candidate.is_file():
            return candidate
        stem = candidate.with_suffix("")
        for ext in IMAGE_EXTS:
            if stem.with_suffix(ext).is_file():
                return stem.with_suffix(ext)
    return None


def _find_mask(folder, name):
    masks = folder / "masks"
    if not masks.is_dir():
        return None
    stem = Path(name).stem
    for ext in (".png", ".jpg", ".jpeg"):
        if (masks / f"{stem}{ext}").is_file():
            return masks / f"{stem}{ext}"
    return None


def _estimate_up(cameras):
    """Average camera 'up' (-Y in OpenCV axes), snapped to an axis if close."""
    if not cameras:
        return np.array([0.0, 0.0, 1.0])
    ups = np.array([-c.camera_to_world[:3, 1] for c in cameras])
    up = ups.mean(axis=0)
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    up /= norm
    axis = int(np.argmax(np.abs(up)))
    if abs(up[axis]) > 0.9:
        snapped = np.zeros(3)
        snapped[axis] = np.sign(up[axis])
        return snapped
    return up


def load(path):
    folder, build = resolve(path)
    model = colmap.read_model(folder)
    warnings = []
    cameras = []
    distorted = set()
    missing = []
    for image in sorted(model.images, key=lambda im: im.name):
        cam = model.cameras.get(image.camera_id)
        if cam is None:
            warnings.append(f"{image.name}: camera {image.camera_id} missing; skipped")
            continue
        if cam.model in colmap.DISTORTED:
            distorted.add(cam.model)
        image_path = _find_image(folder, image.name)
        if image_path is None:
            missing.append(image.name)
            continue
        fx, fy, cx, cy = cam.intrinsics()
        cameras.append(Camera(
            index=len(cameras), name=image.name, image_path=image_path,
            width=cam.width, height=cam.height, fx=fx, fy=fy, cx=cx, cy=cy,
            world_to_camera=image.world_to_camera(),
            mask_path=_find_mask(folder, image.name),
        ))
    if missing:
        warnings.append(f"{len(missing)} image(s) listed in images.txt were not found "
                        f"(first: {missing[0]})")
    if distorted:
        warnings.append(f"Lens distortion of {', '.join(sorted(distorted))} cameras is "
                        "ignored; undistort the images for best results")
    if not len(model.points_xyz):
        warnings.append("No points3D: splats will start from random points")
    splatgen = build is not None or folder.name == LEGACY_FOLDER
    extras = {}
    if build is not None:
        extras["build_folder"] = str(build)
        extras["raw_dataset"] = (build / RAW_FOLDER).is_dir()
    scene = Scene(
        root=folder,
        format="splatgen-legacy" if splatgen else "colmap",
        cameras=cameras,
        points_xyz=model.points_xyz.astype(np.float64),
        points_rgb=model.points_rgb.astype(np.uint8),
        warnings=warnings,
        extras=extras,
    )
    # SplatGen exports Blender world space: Z is up by construction.
    scene.up_axis = np.array([0.0, 0.0, 1.0]) if splatgen else _estimate_up(cameras)
    return scene
