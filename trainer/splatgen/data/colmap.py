"""COLMAP sparse model readers (text and binary).

Only what training needs: cameras (model + intrinsics), image poses and
names, and the sparse points with colours. Poses are COLMAP world-to-camera
in OpenCV axes (+X right, +Y down, +Z forward).
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# model id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
_MODEL_IDS = {name: (mid, n) for mid, (name, n) in CAMERA_MODELS.items()}
#: Models whose extra parameters are lens distortion the trainer ignores.
DISTORTED = {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV", "FOV",
             "OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",
             "THIN_PRISM_FISHEYE"}


@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def intrinsics(self):
        """(fx, fy, cx, cy); distortion terms are dropped."""
        p = self.params
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                          "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"}:
            return float(p[0]), float(p[0]), float(p[1]), float(p[2])
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])


@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray  # w, x, y, z
    tvec: np.ndarray
    camera_id: int
    name: str

    def world_to_camera(self):
        w, x, y, z = self.qvec
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.tvec
        return matrix


@dataclass
class ColmapModel:
    cameras: dict
    images: list
    points_xyz: np.ndarray   # (N, 3) float
    points_rgb: np.ndarray   # (N, 3) uint8


# -- text ---------------------------------------------------------------------

def _data_lines(path):
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() and not line.startswith("#"):
            yield line


def read_cameras_text(path):
    cameras = {}
    for line in _data_lines(path):
        fields = line.split()
        cameras[int(fields[0])] = ColmapCamera(
            int(fields[0]), fields[1], int(fields[2]), int(fields[3]),
            np.array([float(v) for v in fields[4:]]))
    return cameras


def read_images_text(path):
    images = []
    # Image lines alternate with (possibly empty) POINTS2D lines, so blank
    # lines matter here; comments do not.
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines()
             if not line.startswith("#")]
    index = 0
    while index < len(lines):
        fields = lines[index].split()
        if len(fields) < 10:
            index += 1
            continue
        images.append(ColmapImage(
            int(fields[0]), np.array([float(v) for v in fields[1:5]]),
            np.array([float(v) for v in fields[5:8]]), int(fields[8]),
            " ".join(fields[9:])))
        index += 2
    return images


def read_points_text(path):
    xyz, rgb = [], []
    for line in _data_lines(path):
        fields = line.split()
        xyz.append([float(v) for v in fields[1:4]])
        rgb.append([int(v) for v in fields[4:7]])
    return (np.array(xyz, dtype=np.float64).reshape(-1, 3),
            np.array(rgb, dtype=np.uint8).reshape(-1, 3))


# -- binary ---------------------------------------------------------------------

def _read(handle, fmt):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, handle.read(size))


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as handle:
        for _ in range(_read(handle, "Q")[0]):
            cam_id, model_id, width, height = _read(handle, "iiQQ")
            name, count = CAMERA_MODELS[model_id]
            params = np.array(_read(handle, "d" * count))
            cameras[cam_id] = ColmapCamera(cam_id, name, width, height, params)
    return cameras


def read_images_binary(path):
    images = []
    with open(path, "rb") as handle:
        for _ in range(_read(handle, "Q")[0]):
            values = _read(handle, "idddddddi")
            name = b""
            while True:
                char = handle.read(1)
                if char in (b"\x00", b""):
                    break
                name += char
            n_points = _read(handle, "Q")[0]
            handle.seek(24 * n_points, 1)
            images.append(ColmapImage(values[0], np.array(values[1:5]),
                                      np.array(values[5:8]), values[8],
                                      name.decode("utf-8")))
    return images


def read_points_binary(path):
    xyz, rgb = [], []
    with open(path, "rb") as handle:
        for _ in range(_read(handle, "Q")[0]):
            values = _read(handle, "QdddBBBd")
            xyz.append(values[1:4])
            rgb.append(values[4:7])
            track = _read(handle, "Q")[0]
            handle.seek(8 * track, 1)
    return (np.array(xyz, dtype=np.float64).reshape(-1, 3),
            np.array(rgb, dtype=np.uint8).reshape(-1, 3))


def find_model(folder):
    """The folder holding cameras/images files: itself, sparse/0 or sparse."""
    folder = Path(folder)
    for candidate in (folder, folder / "sparse" / "0", folder / "sparse"):
        for ext in (".txt", ".bin"):
            if (candidate / f"cameras{ext}").is_file() and (candidate / f"images{ext}").is_file():
                return candidate, ext
    return None, None


def read_model(folder):
    folder, ext = find_model(folder)
    if folder is None:
        raise FileNotFoundError("No COLMAP cameras/images files found")
    if ext == ".txt":
        cameras = read_cameras_text(folder / "cameras.txt")
        images = read_images_text(folder / "images.txt")
        points = folder / "points3D.txt"
        xyz, rgb = read_points_text(points) if points.is_file() else (np.zeros((0, 3)), np.zeros((0, 3), np.uint8))
    else:
        cameras = read_cameras_binary(folder / "cameras.bin")
        images = read_images_binary(folder / "images.bin")
        points = folder / "points3D.bin"
        xyz, rgb = read_points_binary(points) if points.is_file() else (np.zeros((0, 3)), np.zeros((0, 3), np.uint8))
    return ColmapModel(cameras, images, xyz, rgb)
