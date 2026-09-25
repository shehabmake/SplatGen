"""Import and export of trained Gaussians."""

from .ply import read_ply, write_ply
from .splat import to_splat_bytes, write_splat

FORMATS = {
    "ply": {"label": "3DGS PLY", "extension": ".ply",
            "description": "Standard Gaussian Splatting PLY with full SH (SuperSplat, most tools)."},
    "splat": {"label": ".splat", "extension": ".splat",
              "description": "Compact 32 bytes per splat, colour only (web viewers)."},
}


def export_model(model, fmt, path):
    params = model.state_dict() if hasattr(model, "state_dict") else model
    if fmt == "ply":
        return write_ply(params, path)
    if fmt == "splat":
        return write_splat(params, path)
    raise ValueError(f"Unknown export format {fmt}")


__all__ = ["FORMATS", "export_model", "read_ply", "write_ply", "write_splat", "to_splat_bytes"]
