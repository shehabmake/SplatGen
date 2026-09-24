# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Canonical dataset exporter for SplatGen.

The exporter owns one standard layout and no aliases::

    Build timestamp/
      Dataset(Default)/
        images/               RGB only
        masks/                binary geometry-validity masks only
        depth/                float32 metric camera-Z for RGB-D reconstruction
        cameras.txt
        images.txt
        points3D.txt
      sg_metadata/            optional SplatGen numerical data

When requested, Blender-only numerical data is isolated below
``sg_metadata``.  Metric depth is part of the standard dataset because point
cloud generation back-projects it instead of intersecting the live scene.
Every numerical map needed by the dataset is written by its own typed
compositor output. No combined/archival render-pass output is generated.
"""

import hashlib
import json
import os
import shutil
import struct
import time
import zlib
from pathlib import Path

import bpy

from . import paths


SCHEMA = "splatgen-ground-truth-v5"
CAPTURE_PREFIX = "__SPLATGEN_DATASET_EXPORT__"
WORK_FOLDER = "_SceneRaySplat_Temporary"
CAPTURE_FOLDER = "dataset_export_capture"
CANONICAL_METADATA = ("cameras.txt", "images.txt", "points3D.txt")


class GroundTruthValidationError(RuntimeError):
    """Canonical output is valid, but enriched SplatGen data is incomplete."""

# One authoritative registry for typed pass behavior and manifest metadata.
PASS_DEFINITIONS = {
    "mask": {
        "label": "Validity Mask",
        "item_name": "Mask",
        "socket_type": "FLOAT",
        "folder": "masks",
        "channels": ("Mask.V",),
        "standard": True,
        "view_layer_flag": "use_pass_z",
    },
    "depth": {
        "label": "Metric Depth",
        "item_name": "Depth",
        "socket_type": "FLOAT",
        "folder": "depth",
        "channels": ("Depth.V",),
        "standard": True,
        "view_layer_flag": "use_pass_z",
    },
    "normal": {
        "label": "World Normal",
        "item_name": "Normal",
        "socket_type": "VECTOR",
        "folder": "normals",
        "channels": ("Normal.X", "Normal.Y", "Normal.Z"),
        "standard": False,
        "view_layer_flag": "use_pass_normal",
    },
    "object_index": {
        "label": "Object Index",
        "item_name": "ObjectIndex",
        "socket_type": "FLOAT",
        "folder": "object_index",
        "channels": ("ObjectIndex.V",),
        "standard": False,
        "view_layer_flag": "use_pass_object_index",
    },
    "material_index": {
        "label": "Material Index",
        "item_name": "MaterialIndex",
        "socket_type": "FLOAT",
        "folder": "material_index",
        "channels": ("MaterialIndex.V",),
        "standard": False,
        "view_layer_flag": "use_pass_material_index",
    },
}


def enabled(cfg):
    """Dataset export participates in every explicit build workflow stage."""
    return str(getattr(cfg, "build_workflow", "")) in {
        "RENDER", "POINTS", "FULL"
    }


def ground_truth_enabled(cfg):
    """Metadata is an invariant of every managed dataset build."""
    return enabled(cfg)


def requested_passes(cfg):
    if not enabled(cfg):
        return {key: False for key in PASS_DEFINITIONS}
    ground_truth = ground_truth_enabled(cfg)
    indices = ground_truth and bool(
        getattr(cfg, "export_ground_truth_indices", False)
    )
    return {
        "mask": True,
        # Metric depth is the authoritative point-cloud surface.  It is
        # required even when the optional normals/material training metadata
        # is disabled.
        "depth": True,
        "normal": ground_truth,
        "object_index": indices,
        "material_index": indices,
    }


def mask_format(cfg):
    value = str(getattr(cfg, "dataset_mask_format", "PNG")).upper()
    return ("JPEG", ".jpg") if value == "JPEG" else ("PNG", ".png")


def metadata_root(dataset_root):
    return paths.ground_truth_dir(dataset_root)


def manifest_path(dataset_root):
    return metadata_root(dataset_root) / "manifest.json"


def _capture_root(dataset_root):
    return Path(dataset_root) / WORK_FOLDER / CAPTURE_FOLDER


def _tree_for_scene(scene):
    state = {
        "scene": scene,
        "old_use_nodes": bool(getattr(scene, "use_nodes", False)),
        "old_group": getattr(scene, "compositing_node_group", None),
        "created_group": None,
    }
    tree = getattr(scene, "compositing_node_group", None)
    if hasattr(scene, "compositing_node_group"):
        if tree is None:
            tree = bpy.data.node_groups.new(
                f"{CAPTURE_PREFIX}{scene.name}", "CompositorNodeTree"
            )
            scene.compositing_node_group = tree
            state["created_group"] = tree
    else:
        scene.use_nodes = True
        tree = getattr(scene, "node_tree", None)
    if tree is None:
        raise RuntimeError("Blender did not provide a compositor node tree.")
    try:
        scene.use_nodes = True
    except (AttributeError, TypeError):
        pass
    return tree, state


def _new_output_node(tree, label, directory, socket_type, item_name):
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.name = f"{CAPTURE_PREFIX}{label}"
    node.label = f"SplatGen dataset: {label}"
    if not hasattr(node, "file_output_items") or not hasattr(node, "directory"):
        tree.nodes.remove(node)
        raise RuntimeError(
            "Dataset pass export requires Blender 5.0's typed File Output API."
        )
    node.directory = str(directory)
    node.file_name = ".pending_"
    node.file_output_items.new(socket_type, item_name)
    node.format.color_depth = "32"
    try:
        node.format.exr_codec = "ZIP"
    except (AttributeError, TypeError):
        pass
    return node


def _socket(node, *names):
    for name in names:
        found = node.outputs.get(name)
        if found is not None:
            return found
    return None


def _create_sources(tree, render_layers, passes, state):
    depth = _socket(render_layers, "Depth", "Z")
    if (passes["mask"] or passes["depth"]) and depth is None:
        raise RuntimeError(
            "The selected render engine does not expose the Depth pass required "
            "for geometry-derived masks."
        )
    sources = {}
    if passes["mask"]:
        try:
            threshold = tree.nodes.new("CompositorNodeMath")
        except RuntimeError:
            threshold = tree.nodes.new("ShaderNodeMath")
        threshold.name = f"{CAPTURE_PREFIX}GeometryVisibility"
        threshold.label = "Binary geometry visibility; independent of RGB"
        threshold.operation = "LESS_THAN"
        threshold.inputs[1].default_value = 1.0e9
        tree.links.new(depth, threshold.inputs[0])
        state["nodes"].append(threshold)
        state["mask_threshold_node"] = threshold
        sources["mask"] = threshold.outputs[0]
    if passes["depth"]:
        sources["depth"] = depth
    if passes["normal"]:
        normal = _socket(render_layers, "Normal")
        if normal is None:
            raise RuntimeError(
                "Export Blender Ground-Truth Metadata requested normals, but "
                "the selected render engine does not expose a Normal pass."
            )
        sources["normal"] = normal
    if passes["object_index"]:
        value = _socket(render_layers, "Object Index", "IndexOB")
        if value is None:
            raise RuntimeError(
                "Object Index was requested, but the selected render engine "
                "does not expose that pass. Use Cycles or disable index export."
            )
        sources["object_index"] = value
    if passes["material_index"]:
        value = _socket(render_layers, "Material Index", "IndexMA")
        if value is None:
            raise RuntimeError(
                "Material Index was requested, but the selected render engine "
                "does not expose that pass. Use Cycles or disable index export."
            )
        sources["material_index"] = value
    return sources


def begin_capture(scene, cfg, dataset_root, resolution):
    """Create one typed temporary compositor output per requested pass."""
    if not enabled(cfg):
        return None
    dataset_root = Path(dataset_root)
    passes = requested_passes(cfg)
    paths.masks_dir(dataset_root).mkdir(parents=True, exist_ok=True)
    paths.depth_dir(dataset_root).mkdir(parents=True, exist_ok=True)
    capture = _capture_root(dataset_root)
    for key, active in passes.items():
        if active:
            (capture / key).mkdir(parents=True, exist_ok=True)
    if ground_truth_enabled(cfg):
        gtroot = metadata_root(dataset_root)
        for key, active in passes.items():
            if active and not PASS_DEFINITIONS[key]["standard"]:
                (gtroot / PASS_DEFINITIONS[key]["folder"]).mkdir(
                    parents=True, exist_ok=True
                )

    tree, state = _tree_for_scene(scene)
    state.update({
        "tree": tree,
        "nodes": [],
        "dataset_root": dataset_root,
        "capture_root": capture,
        "passes": passes,
        "ground_truth": ground_truth_enabled(cfg),
        "resolution": tuple(int(value) for value in resolution),
        "mask_format": mask_format(cfg),
        "png_compression": int(getattr(cfg, "png_compression", 15)),
        "output_nodes": {},
    })
    layer = bpy.context.view_layer
    state["view_layer"] = layer
    needed_flags = {
        definition["view_layer_flag"]
        for key, definition in PASS_DEFINITIONS.items()
        if passes[key] and definition["view_layer_flag"]
    }
    state["old_pass_flags"] = {}
    for flag in needed_flags:
        if not hasattr(layer, flag):
            restore_capture(state)
            raise RuntimeError(
                f"The selected render engine cannot enable required pass flag {flag}."
            )
        state["old_pass_flags"][flag] = bool(getattr(layer, flag))
        setattr(layer, flag, True)
    try:
        layer.update_render_passes()
    except (AttributeError, RuntimeError):
        pass

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    render_layers.name = f"{CAPTURE_PREFIX}RenderLayers"
    render_layers.label = "SplatGen raw render-pass source"
    render_layers.scene = scene
    try:
        render_layers.layer = layer.name
    except (AttributeError, TypeError):
        pass
    state["nodes"].append(render_layers)

    try:
        sources = _create_sources(tree, render_layers, passes, state)
        for key, source in sources.items():
            definition = PASS_DEFINITIONS[key]
            node = _new_output_node(
                tree,
                definition["label"],
                capture / key,
                definition["socket_type"],
                definition["item_name"],
            )
            tree.links.new(source, node.inputs[definition["item_name"]])
            state["nodes"].append(node)
            state["output_nodes"][key] = node

    except Exception:
        restore_capture(state)
        raise
    return state


def prepare_view(state, frame_index, maximum_depth=None):
    if not state:
        return
    stem = f"frame_{int(frame_index):04d}"
    pending = f".{stem}.pending_"
    for node in state["output_nodes"].values():
        node.file_name = pending
    threshold = state.get("mask_threshold_node")
    if threshold is not None and maximum_depth is not None:
        maximum = max(1.0e-6, float(maximum_depth))
        threshold.inputs[1].default_value = maximum * (1.0 - 1.0e-6)
    state["active_stem"] = stem


def _pending_file(folder, stem, timeout=8.0):
    """Wait briefly for Blender's compositor File Output worker to flush.

    ``render_complete`` can be observed before every independent File Output
    node becomes visible on disk, especially for high-resolution multilayer
    EXRs.  Treating the first directory check as final caused one repeatable
    view to lose the later normal/archive outputs.  The renderer is done
    at this point, so a bounded filesystem poll is sufficient and avoids a
    second expensive render in the normal case.
    """
    deadline = time.monotonic() + max(0.0, float(timeout))
    pattern = f".{stem}.pending_*.exr"
    while True:
        candidates = sorted(
            (path for path in folder.glob(pattern) if path.is_file()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if candidates:
            return candidates[0], candidates
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    raise RuntimeError(
        f"Blender did not flush the requested output for {stem} within "
        f"{float(timeout):.1f} seconds: {folder}"
    )


def _oiio():
    try:
        import OpenImageIO as oiio
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Blender's OpenImageIO module is unavailable; typed dataset "
            "validation cannot continue. Repair Blender 5.0 or newer."
        ) from exc
    return oiio, np


def _read_exr_parts(path):
    oiio, np = _oiio()
    handle = oiio.ImageInput.open(str(path))
    if handle is None:
        raise RuntimeError(f"OpenEXR could not be opened: {path}")
    parts = []
    try:
        index = 0
        while handle.seek_subimage(index, 0):
            spec = handle.spec()
            decoded = handle.read_image(oiio.FLOAT)
            if decoded is None:
                raise RuntimeError(f"OpenEXR part {index} could not be decoded: {path}")
            # Own the pixel storage before ImageInput is closed.  Some OIIO
            # builds expose read_image() through a temporary native buffer;
            # retaining that view across close() is unsafe in a long Blender
            # dataset build.
            pixels = np.array(decoded, dtype=np.float32, copy=True, order="C")
            if pixels.size == 0:
                raise RuntimeError(f"OpenEXR part {index} is empty: {path}")
            parts.append({
                "name": str(spec.get_string_attribute("name") or ""),
                "width": int(spec.width),
                "height": int(spec.height),
                "channels": tuple(str(name) for name in spec.channelnames),
                "pixels": pixels,
            })
            index += 1
    finally:
        handle.close()
    if not parts:
        raise RuntimeError(f"OpenEXR contains no readable parts: {path}")
    return parts


def _validate_typed_pass(path, key, resolution):
    definition = PASS_DEFINITIONS[key]
    parts = _read_exr_parts(path)
    if len(parts) != 1:
        raise RuntimeError(
            f"{definition['label']} must contain exactly one EXR part; "
            f"found {len(parts)} in {path.name}."
        )
    part = parts[0]
    if (part["width"], part["height"]) != tuple(resolution):
        raise RuntimeError(
            f"{definition['label']} is {part['width']}x{part['height']}; "
            f"expected {resolution[0]}x{resolution[1]}: {path.name}"
        )
    if part["channels"] != definition["channels"]:
        raise RuntimeError(
            f"{definition['label']} channels are {part['channels']}; expected "
            f"{definition['channels']}: {path.name}"
        )
    _oiio_module, np = _oiio()
    pixels = np.asarray(part["pixels"], dtype=np.float32)
    if key == "normal":
        if not np.isfinite(pixels).all():
            raise RuntimeError(f"World Normal contains non-finite values: {path.name}")
        lengths = np.linalg.norm(pixels[..., :3], axis=-1)
        visible = lengths > 0.25
        if int(visible.sum()) == 0:
            raise RuntimeError(f"World Normal contains no rendered surface vectors: {path.name}")
        median = float(np.median(lengths[visible]))
        if not 0.75 <= median <= 1.25:
            raise RuntimeError(
                f"World Normal vectors are not approximately unit length "
                f"(median {median:.4f}): {path.name}"
            )
    elif key in {"object_index", "material_index"}:
        if not np.isfinite(pixels).all():
            raise RuntimeError(f"{definition['label']} contains non-finite values: {path.name}")
    return part


def _png_chunk(kind, payload):
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", checksum)
    )


def _encode_png_mask(pixels, target, width, height, compression=15):
    """Write an 8-bit grayscale PNG without an optional native dependency."""
    scanlines = bytearray((width + 1) * height)
    row_size = width + 1
    for row_index in range(height):
        offset = row_index * row_size
        # Filter method 0 makes this byte stream unambiguous and keeps the
        # encoder independent of NumPy/OIIO buffer-stride assumptions.
        scanlines[offset] = 0
        scanlines[offset + 1:offset + row_size] = pixels[row_index].tobytes()
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    # Blender exposes a percentage; deflate accepts levels 0 through 9.
    level = int(max(0, min(100, int(compression))) * 9 / 100)
    with open(target, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(_png_chunk(b"IHDR", header))
        handle.write(_png_chunk(b"IDAT", zlib.compress(scanlines, level=level)))
        handle.write(_png_chunk(b"IEND", b""))


def _encode_jpeg_mask(pixels, target, width, height, np):
    """Encode through a Blender-owned pixel buffer, never an OIIO NumPy view."""
    scene = bpy.context.scene
    settings = scene.render.image_settings
    previous = {
        "file_format": settings.file_format,
        "color_mode": settings.color_mode,
        "color_depth": settings.color_depth,
        "quality": settings.quality,
    }
    image = bpy.data.images.new(
        f"{CAPTURE_PREFIX}JPEGMask",
        width=width,
        height=height,
        alpha=False,
        float_buffer=False,
        is_data=True,
    )
    try:
        rgba = np.empty((height, width, 4), dtype=np.float32)
        # OIIO arrays are top-to-bottom; Blender's image pixel storage starts
        # at the lower-left. Reverse rows exactly once before save_render().
        normalized = pixels[::-1].astype(np.float32) / 255.0
        rgba[..., 0] = normalized
        rgba[..., 1] = normalized
        rgba[..., 2] = normalized
        rgba[..., 3] = 1.0
        image.pixels.foreach_set(rgba.reshape(-1))
        settings.file_format = "JPEG"
        settings.color_mode = "BW"
        settings.color_depth = "8"
        settings.quality = 100
        image.save_render(str(target), scene=scene)
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)
        bpy.data.images.remove(image)


def _write_mask(source, target, resolution, file_format, png_compression=15):
    part = _validate_typed_pass(source, "mask", resolution)
    _oiio_module, np = _oiio()
    values = np.asarray(part["pixels"])[..., 0]
    pixels = np.ascontiguousarray(
        (np.isfinite(values) & (values >= 0.5)).astype(np.uint8) * 255
    )
    width, height = int(resolution[0]), int(resolution[1])
    if pixels.shape != (height, width):
        raise RuntimeError(
            f"Mask raster has shape {pixels.shape}; expected {(height, width)}: "
            f"{source.name}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.stem}.encoding{target.suffix}")
    try:
        pending.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        # OIIO's Python ImageOutput.write_image() can access-violate Blender
        # while contiguizing NumPy data.  The standard PNG path below contains
        # no OIIO/Pillow dependency.  JPEG uses Blender-owned pixels, so no
        # foreign array pointer reaches Blender's native image writer.
        if file_format == "JPEG":
            _encode_jpeg_mask(pixels, pending, width, height, np)
        else:
            _encode_png_mask(pixels, pending, width, height, png_compression)
        if not pending.is_file() or pending.stat().st_size == 0:
            raise RuntimeError(f"Mask encoder produced no data: {target}")
        os.replace(pending, target)
    finally:
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"Mask output is missing after encoding: {target}")
    actual_width, actual_height, channels = _image_spec(target)
    if (actual_width, actual_height, channels) != (width, height, 1):
        raise RuntimeError(
            f"Encoded mask is {actual_width}x{actual_height} with {channels} "
            f"channels; expected {width}x{height} with 1 channel: {target.name}"
        )


def _remove_candidates(candidates, published=None):
    for path in candidates:
        if published is not None and path == published:
            continue
        try:
            path.unlink()
        except OSError:
            pass


def finish_view(state, frame_index):
    """Validate a whole view before publishing any of its outputs.

    Blender can occasionally omit one File Output node while still completing
    the RGB render.  Previously, outputs earlier in the loop were already
    moved into the dataset, leaving a misleading half-published view.  Stage
    and validate every pass first; only a complete view reaches the dataset.
    """
    if not state:
        return {}
    stem = f"frame_{int(frame_index):04d}"
    outputs = {}
    prepared = []
    staged_mask = None
    for key, active in state["passes"].items():
        if not active:
            continue
        source, candidates = _pending_file(state["capture_root"] / key, stem)
        if key == "mask":
            file_format, extension = state["mask_format"]
            target = paths.masks_dir(state["dataset_root"]) / f"{stem}{extension}"
            publish_dir = state["capture_root"] / "publish"
            publish_dir.mkdir(parents=True, exist_ok=True)
            staged_mask = publish_dir / f"{stem}{extension}"
            _write_mask(source, staged_mask, state["resolution"], file_format,
                        state.get("png_compression", 15))
            prepared.append((key, staged_mask, target, candidates))
        else:
            _validate_typed_pass(source, key, state["resolution"])
            target = _target_for(
                state["dataset_root"], key, stem, None
            )
            prepared.append((key, source, target, candidates))

    # No dataset file changes until every requested pass above has validated.
    for key, source, target, candidates in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        _remove_candidates(candidates, published=source)
        outputs[key] = target
        if key == "mask":
            # A format change must not leave a second mask with the same stem.
            for stale_extension in (".png", ".jpg", ".jpeg"):
                stale = target.with_suffix(stale_extension)
                if stale != target:
                    try:
                        stale.unlink(missing_ok=True)
                    except OSError:
                        pass
    return outputs


def _target_for(dataset_root, key, stem, cfg):
    if key == "mask":
        return paths.masks_dir(dataset_root) / f"{stem}{mask_format(cfg)[1]}"
    if key == "depth":
        return paths.depth_dir(dataset_root) / f"{stem}.exr"
    return (
        metadata_root(dataset_root)
        / PASS_DEFINITIONS[key]["folder"]
        / f"{stem}.exr"
    )


def read_typed_pass(dataset_root, key, frame_index, resolution):
    """Return a validated top-left raster for one dataset numerical pass.

    The returned array owns its storage, so the point sampler can keep it for
    the duration of a camera without retaining an OpenImageIO handle.
    Scalar passes are returned as ``H x W`` and vector passes as ``H x W x C``.
    """
    if key not in PASS_DEFINITIONS or key == "mask":
        raise ValueError(f"Unsupported typed dataset pass: {key}")
    stem = f"frame_{int(frame_index):04d}"
    path = _target_for(dataset_root, key, stem, None)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing {PASS_DEFINITIONS[key]['label']}: {path}")
    part = _validate_typed_pass(path, key, tuple(resolution))
    _oiio_module, np = _oiio()
    pixels = np.array(part["pixels"], dtype=np.float32, copy=True, order="C")
    if pixels.ndim == 3 and pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    return pixels


def view_complete(dataset_root, frame_index, cfg):
    if not enabled(cfg):
        return True
    stem = f"frame_{int(frame_index):04d}"
    checks = [
        _target_for(dataset_root, key, stem, cfg)
        for key, active in requested_passes(cfg).items() if active
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in checks)


def copy_view_outputs(previous, target, frame_index, cfg):
    """Copy only validated exporter-owned files for an unchanged reused view."""
    if not view_complete(previous, frame_index, cfg):
        return False
    stem = f"frame_{int(frame_index):04d}"
    for key, active in requested_passes(cfg).items():
        if not active:
            continue
        source = _target_for(previous, key, stem, cfg)
        destination = _target_for(target, key, stem, cfg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return True


def restore_capture(state):
    if not state:
        return
    tree = state.get("tree")
    if tree is not None:
        for node in reversed(state.get("nodes", ())):
            try:
                tree.nodes.remove(node)
            except (ReferenceError, RuntimeError):
                pass
    layer = state.get("view_layer")
    if layer is not None:
        for flag, value in state.get("old_pass_flags", {}).items():
            try:
                setattr(layer, flag, value)
            except (ReferenceError, AttributeError, TypeError):
                pass
    scene = state.get("scene")
    created = state.get("created_group")
    if scene is not None:
        try:
            if hasattr(scene, "compositing_node_group"):
                scene.compositing_node_group = state.get("old_group")
            else:
                scene.use_nodes = state.get("old_use_nodes", False)
        except (ReferenceError, AttributeError, TypeError):
            pass
    if created is not None:
        try:
            bpy.data.node_groups.remove(created)
        except (ReferenceError, RuntimeError):
            pass


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_spec(path):
    oiio, _np = _oiio()
    handle = oiio.ImageInput.open(str(path))
    if handle is None:
        raise RuntimeError(f"Image could not be decoded: {path}")
    try:
        if not handle.seek_subimage(0, 0):
            raise RuntimeError(f"Image has no readable raster: {path}")
        spec = handle.spec()
        return int(spec.width), int(spec.height), int(spec.nchannels)
    finally:
        handle.close()


def clean_legacy_outputs(dataset_root, keep_ground_truth):
    """Remove exporter artifacts from superseded layouts; never make aliases."""
    dataset_root = Path(dataset_root)
    standard_root = paths.standard_dataset_dir(dataset_root)
    image_dir = paths.data_dir(dataset_root)
    for name in CANONICAL_METADATA:
        try:
            (image_dir / name).unlink(missing_ok=True)
        except OSError:
            pass
    for name in ("camera.txt", "point.txt", "images_point.txt"):
        for parent in {dataset_root, standard_root}:
            try:
                (parent / name).unlink(missing_ok=True)
            except OSError:
                pass
    if standard_root != dataset_root:
        for name in CANONICAL_METADATA:
            try:
                (dataset_root / name).unlink(missing_ok=True)
            except OSError:
                pass
    for folder_name in ("metadata", "sgdata"):
        folder = dataset_root / folder_name
        if folder.is_dir():
            try:
                shutil.rmtree(folder)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not remove obsolete exporter folder {folder}: {exc}"
                ) from exc
    archive = metadata_root(dataset_root) / "render_passes"
    if archive.is_dir():
        try:
            shutil.rmtree(archive)
        except OSError as exc:
            raise RuntimeError(
                f"Could not remove obsolete render-pass archive {archive}: {exc}"
            ) from exc
    if not keep_ground_truth:
        folder = metadata_root(dataset_root)
        if folder.is_dir():
            try:
                shutil.rmtree(folder)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not remove disabled ground-truth output {folder}: {exc}"
                ) from exc


def _remove_empty_capture_tree(dataset_root):
    """Remove exporter temporaries while preserving other in-progress work."""
    capture = _capture_root(dataset_root)
    if capture.is_dir():
        for folder in sorted(capture.rglob("*"), reverse=True):
            if folder.is_dir():
                try:
                    folder.rmdir()
                except OSError:
                    pass
        try:
            capture.rmdir()
        except OSError:
            pass
    try:
        capture.parent.rmdir()
    except OSError:
        pass


def _validation_issues(dataset_root, views, resolution, passes, ground_truth,
                       require_enriched=True):
    issues = []
    dataset_root = paths.build_root(dataset_root)
    standard_root = paths.standard_dataset_dir(dataset_root)
    for name in CANONICAL_METADATA:
        path = paths.dataset_file(dataset_root, name)
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing canonical metadata: {name}")
        duplicate = paths.data_dir(dataset_root) / name
        if duplicate.exists():
            issues.append(f"metadata must not be inside images/: images/{name}")
    for forbidden in ("camera.txt", "point.txt", "images_point.txt", "metadata", "sgdata"):
        for parent in {dataset_root, standard_root}:
            if (parent / forbidden).exists():
                issues.append(f"obsolete duplicate output exists: {forbidden}")
                break
    if standard_root != dataset_root:
        for name in CANONICAL_METADATA:
            if (dataset_root / name).exists():
                issues.append(f"canonical metadata must be inside {paths.DEFAULT_DATASET_FOLDER}: {name}")

    for view in views:
        rgb = dataset_root / str(view.get("rgb", ""))
        mask = dataset_root / str(view.get("mask", ""))
        depth = dataset_root / str(view.get("depth", ""))
        if not rgb.is_file() or rgb.stat().st_size == 0:
            issues.append(f"missing RGB image: {view.get('rgb', '<unset>')}")
        else:
            try:
                width, height, _channels = _image_spec(rgb)
                if (width, height) != tuple(resolution):
                    issues.append(
                        f"RGB dimensions {width}x{height}, expected "
                        f"{resolution[0]}x{resolution[1]}: {rgb.name}"
                    )
            except Exception as exc:
                issues.append(str(exc))
        if require_enriched:
            if not mask.is_file() or mask.stat().st_size == 0:
                issues.append(
                    f"missing standard mask: {view.get('mask', '<unset>')}"
                )
            else:
                try:
                    width, height, channels = _image_spec(mask)
                    if (width, height) != tuple(resolution) or channels != 1:
                        issues.append(
                            f"mask must be one-channel "
                            f"{resolution[0]}x{resolution[1]}: {mask.name}"
                        )
                except Exception as exc:
                    issues.append(str(exc))

            if not depth.is_file() or depth.stat().st_size == 0:
                issues.append(
                    f"missing standard metric depth: "
                    f"{view.get('depth', '<unset>')}"
                )
            else:
                try:
                    _validate_typed_pass(depth, "depth", resolution)
                except Exception as exc:
                    issues.append(str(exc))

        if require_enriched and ground_truth:
            for key, active in passes.items():
                # A pass this version no longer writes (the 5.3.21 albedo)
                # is extra data, not a fault in the dataset.
                if not active or key in {"mask", "depth"} or key not in PASS_DEFINITIONS:
                    continue
                path = dataset_root / str(view.get(key, ""))
                if not path.is_file() or path.stat().st_size == 0:
                    issues.append(
                        f"missing {key} ground-truth pass for {rgb.name}: "
                        f"{view.get(key, '<unset>')}"
                    )
                    continue
                try:
                    _validate_typed_pass(path, key, resolution)
                except Exception as exc:
                    issues.append(str(exc))
    if require_enriched and ground_truth:
        seed = paths.scene_guidance_seed(dataset_root)
        guidance_manifest = paths.scene_guidance_manifest(dataset_root)
        if not seed.is_file() or seed.stat().st_size == 0:
            issues.append("missing compact scene-guidance surface seed")
        if not guidance_manifest.is_file():
            issues.append("missing compact scene-guidance manifest")
        else:
            try:
                guidance = json.loads(
                    guidance_manifest.read_text(encoding="utf-8")
                )
                if guidance.get("format") != "SplatGen Scene Guidance":
                    issues.append("unsupported compact scene-guidance format")
                if int(guidance.get("point_count", 0)) <= 0:
                    issues.append("scene-guidance point_count must be positive")
                if seed.is_file() and guidance.get("seed_sha256") != _sha256(seed):
                    issues.append("scene-guidance seed checksum mismatch")
            except (OSError, TypeError, ValueError) as exc:
                issues.append(f"invalid scene-guidance manifest: {exc}")
    return issues


def validate_complete_dataset(dataset_root):
    manifest = manifest_path(dataset_root)
    if not manifest.is_file():
        raise RuntimeError(f"Missing ground-truth manifest: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Ground-truth manifest is invalid: {exc}") from exc
    if payload.get("schema") != SCHEMA:
        raise RuntimeError(f"Unsupported ground-truth schema: {payload.get('schema')}")
    resolution = tuple(int(value) for value in payload.get("resolution", (0, 0)))
    passes = payload.get("enabled_passes") or {}
    views = payload.get("views") or []
    issues = _validation_issues(
        dataset_root, views, resolution, passes, True, require_enriched=True
    )
    if int(payload.get("view_count", -1)) != len(views):
        issues.append("manifest view_count does not match views")
    if issues:
        raise RuntimeError(
            "Dataset completion validation failed:\n - " + "\n - ".join(issues)
        )
    return {
        "view_count": len(views),
        "checked_passes": [key for key, active in passes.items() if active],
    }


def finalize_dataset(dataset_root, records, resolution, color_management, cfg):
    """Validate the single standard dataset and write optional GT manifest."""
    if not enabled(cfg):
        return None
    dataset_root = paths.build_root(dataset_root)
    ground_truth = ground_truth_enabled(cfg)
    clean_legacy_outputs(dataset_root, ground_truth)
    if not all(paths.dataset_file(dataset_root, name).is_file()
               for name in CANONICAL_METADATA):
        return None

    width, height = int(resolution[0]), int(resolution[1])
    passes = requested_passes(cfg)
    views = []
    for record in sorted(records, key=lambda item: int(item.get("frame_index", 0))):
        if str(record.get("status", "")).upper() != "RENDERED":
            continue
        relative = str(
            record.get("training_file_path", record.get("file_path", ""))
        ).lstrip("./\\").replace("\\", "/")
        rgb = dataset_root / relative
        stem = rgb.stem
        mask = _target_for(dataset_root, "mask", stem, cfg)
        view = {
            "index": int(record["frame_index"]),
            "camera_name": str(record.get("name", "")),
            "rgb": rgb.relative_to(dataset_root).as_posix(),
            "mask": mask.relative_to(dataset_root).as_posix(),
            "depth": _target_for(
                dataset_root, "depth", stem, cfg
            ).relative_to(dataset_root).as_posix(),
            "width": width,
            "height": height,
            "sha256": {},
        }
        if rgb.is_file():
            view["sha256"]["rgb"] = _sha256(rgb)
        if mask.is_file():
            view["sha256"]["mask"] = _sha256(mask)
        depth = _target_for(dataset_root, "depth", stem, cfg)
        if depth.is_file():
            view["sha256"]["depth"] = _sha256(depth)
        if ground_truth:
            for key, active in passes.items():
                if not active or key in {"mask", "depth"}:
                    continue
                path = _target_for(dataset_root, key, stem, cfg)
                view[key] = path.relative_to(dataset_root).as_posix()
                if path.is_file():
                    view["sha256"][key] = _sha256(path)
        views.append(view)

    canonical_issues = _validation_issues(
        dataset_root, views, (width, height), passes, False,
        require_enriched=False,
    )
    if canonical_issues:
        raise RuntimeError(
            "Canonical dataset validation failed:\n - "
            + "\n - ".join(canonical_issues)
        )
    enriched_issues = _validation_issues(
        dataset_root, views, (width, height), passes, ground_truth,
        require_enriched=True,
    )
    if enriched_issues:
        try:
            manifest_path(dataset_root).unlink(missing_ok=True)
        except OSError:
            pass
        raise GroundTruthValidationError(
            "Canonical images/cameras/points are complete, but SplatGen "
            "training metadata is incomplete:\n - "
            + "\n - ".join(enriched_issues)
        )
    if not ground_truth:
        _remove_empty_capture_tree(dataset_root)
        return dataset_root

    pass_contract = {}
    for key, active in passes.items():
        if not active:
            continue
        definition = PASS_DEFINITIONS[key]
        pass_contract[key] = {
            "directory": (
                f"{paths.DEFAULT_DATASET_FOLDER}/masks" if key == "mask"
                else f"{paths.DEFAULT_DATASET_FOLDER}/depth" if key == "depth"
                else f"sg_metadata/{definition['folder']}"
            ),
            "channels": ["Y"] if key == "mask" else list(definition["channels"]),
            "storage": (
                mask_format(cfg)[0]
                if key == "mask"
                else "OpenEXR 32-bit floating point"
            ),
        }
    payload = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "path_base": "build_root",
        "resolution": [width, height],
        "view_count": len(views),
        "enabled_passes": passes,
        "standard_dataset": {
            "root": paths.DEFAULT_DATASET_FOLDER,
            "images": f"{paths.DEFAULT_DATASET_FOLDER}/images",
            "masks": f"{paths.DEFAULT_DATASET_FOLDER}/masks",
            "depth": f"{paths.DEFAULT_DATASET_FOLDER}/depth",
            "cameras": f"{paths.DEFAULT_DATASET_FOLDER}/cameras.txt",
            "images_metadata": f"{paths.DEFAULT_DATASET_FOLDER}/images.txt",
            "points": f"{paths.DEFAULT_DATASET_FOLDER}/points3D.txt",
        },
        "ground_truth": {
            "root": "sg_metadata",
            "manifest": "sg_metadata/manifest.json",
            "scene_guidance": {
                "root": "sg_metadata/scene_guidance",
                "manifest": "sg_metadata/scene_guidance/manifest.json",
                "surface_seed": "sg_metadata/scene_guidance/surface_seed.npz",
            },
        },
        "pass_contract": pass_contract,
        "conventions": {
            "pixel_origin": "top-left; every pass matches the RGB raster without flipping",
            "mask": "one-channel binary raster; 255=valid rendered geometry and 0=ignored background; RGB color is never used to infer validity",
            "depth": "Depth.V float32 camera-space metric Blender Depth pass; used to back-project the point cloud and mask-invalid pixels are excluded by training",
            "normal_storage": "Normal.X/Y/Z float32 Blender world-space shading normals; approximately unit length where mask-valid",
            "normal_training_conversion": "n_camera_opencv = R_world_to_camera_opencv @ n_world; OpenCV axes are +X right, +Y down, +Z forward",
            "camera": "COLMAP PINHOLE with OpenCV world-to-camera poses",
        },
        "color_management": color_management or {},
        "views": views,
        "validation": {
            "status": "passed",
            "validated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checked_passes": [key for key, active in passes.items() if active],
        },
    }
    target = manifest_path(dataset_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(".manifest.pending.json")
    pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(pending, target)
    validate_complete_dataset(dataset_root)
    _remove_empty_capture_tree(dataset_root)
    return metadata_root(dataset_root)
