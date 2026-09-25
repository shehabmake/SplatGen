"""Training images, decoded once and kept in memory as uint8."""

import numpy as np
import torch
from PIL import Image


def load_image(path, size=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.array(image, dtype=np.uint8)


def load_mask(path, size=None):
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        return np.array(image, dtype=np.uint8) >= 128


class ImageCache:
    def __init__(self, cameras, downscale=1, masks=False, device="cpu"):
        self.downscale = max(1, int(downscale))
        self.device = torch.device(device)
        self.items = {}
        for cam in cameras:
            size = (max(1, round(cam.width / self.downscale)),
                    max(1, round(cam.height / self.downscale)))
            rgb = torch.from_numpy(load_image(cam.image_path, size))
            mask = None
            if masks and cam.mask_path is not None:
                mask = torch.from_numpy(load_mask(cam.mask_path, size))
            self.items[cam.index] = (rgb, mask, size)
        total = sum(rgb.numel() for rgb, _m, _s in self.items.values())
        # Keep images on the GPU when they comfortably fit.
        self.on_device = self.device.type != "cpu" and total < 1.5e9
        if self.on_device:
            self.items = {k: (rgb.to(self.device), None if m is None else m.to(self.device), s)
                          for k, (rgb, m, s) in self.items.items()}

    def size(self, cam):
        return self.items[cam.index][2]

    def get(self, cam):
        rgb, mask, size = self.items[cam.index]
        rgb = rgb.to(self.device, non_blocking=True).float() / 255.0
        if mask is not None:
            mask = mask.to(self.device, non_blocking=True)
        return rgb, mask, size
