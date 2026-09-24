# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Cameras, scene description and the manifest of the raw dataset.

The manifest is written last and is the dataset's table of contents: every
per-view pass with its path pattern, channels, storage and completeness,
plus every scene-level file. A reader should never need to list folders.
"""

import math
import shutil
import time
from pathlib import Path

import bpy
from mathutils import Vector

from . import capture, layout, properties, sessions

CONVENTIONS = {
    "pixel_origin": "top-left; every per-view raster matches the legacy RGB image without flipping",
    "frame_naming": "frame_NNNN matches Dataset(Default)/images/frame_NNNN.* and images.txt names",
    "exr": "single-part OpenEXR; RGBA passes use channels R,G,B,A, vector passes X,Y,Z, scalar passes V",
    "color": "scene-linear in the Blender working space (see color_management); no view transform",
    "world_frame": "Blender world space: right-handed, Z up, scene units; identical to the legacy COLMAP export frame",
    "camera_axes_blender": "c2w_blender: camera looks down -Z, +Y up, +X right",
    "camera_axes_opencv": "c2w_opencv / w2c_opencv: +Z forward, +Y down, +X right (COLMAP)",
    "normals": "world space unless the pass says otherwise",
    "depth": "planar camera Z (not ray length); background 1e10",
    "ids": "integers stored as float32; 0 = background; names in ids/id_map.json",
    "equirect": "u = 0.5 - atan2(y, x) / 2pi (centre looks along +X, u = 0.25 along +Y), top row = +Z; matches Blender's Environment Texture",
}


def _matrix(m):
    return [[float(m[r][c]) for c in range(4)] for r in range(4)]


def rendered_frames(build):
    """(frames, resolution, color_management) of a build's rendered cameras."""
    from .. import sceneray_splat

    data = (sceneray_splat.read_render_manifest(build)
            or sceneray_splat.read_completed_camera_data(build) or {})
    frames = []
    for entry in data.get("cameras", ()):
        if str(entry.get("status", "")).upper() != "RENDERED":
            continue
        try:
            index = int(entry.get("frame_index"))
        except (TypeError, ValueError):
            continue
        frames.append({
            "index": index,
            "stem": layout.frame_stem(index),
            "camera_name": str(entry.get("name", "")),
            "legacy_image": str(entry.get("training_file_path",
                                          entry.get("file_path", ""))).lstrip("./\\"),
            "entry": entry,
        })
    frames.sort(key=lambda frame: frame["index"])
    resolution = tuple(int(v) for v in data.get("resolution", (0, 0)))
    return frames, resolution, data.get("color_management")


def camera_matches(frame, cfg):
    """True when the camera still renders what its legacy image shows."""
    from ..building_data import manifest as bd_manifest

    camera = bpy.data.objects.get(frame["camera_name"])
    if camera is None or camera.type != "CAMERA":
        return False
    entry = frame["entry"]
    if bd_manifest.signature_of(entry) is None:
        return True
    return bd_manifest.matches(entry, bd_manifest.camera_signature(camera, cfg))


# --------------------------------------------------------------------------
# cameras/
# --------------------------------------------------------------------------

def _colmap_ids(images_txt):
    ids = {}
    try:
        lines = Path(images_txt).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ids
    expect_data = True
    for line in lines:
        if line.startswith("#"):
            continue
        if expect_data:
            fields = line.split()
            if len(fields) >= 10:
                ids[fields[9]] = int(fields[0])
                expect_data = False
        else:
            expect_data = True
    return ids


def write_cameras(scene, build, root, frames, resolution):
    from .. import sceneray_splat
    from ..building_data import paths as bd_paths

    cfg = scene.SCENERAY_SPLAT
    colmap_dir = Path(root) / layout.COLMAP_FOLDER
    colmap_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in bd_paths.DATASET_FILES:
        source = bd_paths.dataset_file(build, name)
        if source.is_file():
            shutil.copy2(source, colmap_dir / name)
            copied[name] = f"{layout.COLMAP_FOLDER}/{name}"
    image_ids = _colmap_ids(colmap_dir / "images.txt")

    width, height = resolution

    class _Render:
        resolution_x, resolution_y = width, height
        resolution_percentage = 100
        pixel_aspect_x = scene.render.pixel_aspect_x
        pixel_aspect_y = scene.render.pixel_aspect_y

    records = []
    for frame in frames:
        camera = bpy.data.objects.get(frame["camera_name"])
        record = {
            "frame_index": frame["index"],
            "stem": frame["stem"],
            "camera_name": frame["camera_name"],
            "legacy_image": f"../{frame['legacy_image']}",
            "colmap_image_id": image_ids.get(Path(frame["legacy_image"]).name),
            "width": width,
            "height": height,
        }
        if camera is None or camera.type != "CAMERA":
            record["error"] = "camera no longer exists in the scene"
            records.append(record)
            continue
        data = camera.data
        c2w, had_scale = sceneray_splat.camera_pose_world(camera)
        fx, fy, cx, cy, angle = sceneray_splat.compute_intrinsics(data, _Render)
        c2w_cv = sceneray_splat.blender_to_opencv(c2w)
        w2c_cv = c2w_cv.inverted()
        q, t = sceneray_splat.colmap_pose_from_c2w(c2w)
        record.update({
            "pose_matches_render": camera_matches(frame, cfg),
            "model": "PINHOLE",
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "camera_angle_x": angle,
            "camera_angle_y": 2.0 * math.atan(height / (2.0 * fy)),
            "projection": str(data.type),
            "lens_mm": float(data.lens),
            "sensor_width_mm": float(data.sensor_width),
            "sensor_height_mm": float(data.sensor_height),
            "sensor_fit": str(data.sensor_fit),
            "shift_x": float(data.shift_x),
            "shift_y": float(data.shift_y),
            "clip_start": float(data.clip_start),
            "clip_end": float(data.clip_end),
            "object_scale_ignored": bool(had_scale),
            "c2w_blender": _matrix(c2w),
            "c2w_opencv": _matrix(c2w_cv),
            "w2c_opencv": _matrix(w2c_cv),
            "colmap_qvec": [q.w, q.x, q.y, q.z],
            "colmap_tvec": [t.x, t.y, t.z],
            "position": [float(v) for v in c2w.translation],
        })
        records.append(record)
    payload = {
        "format": "SplatGen raw cameras",
        "coordinate_frame": CONVENTIONS["world_frame"],
        "camera_axes": {
            "c2w_blender": CONVENTIONS["camera_axes_blender"],
            "c2w_opencv": CONVENTIONS["camera_axes_opencv"],
        },
        "intrinsics": "pixels; principal point measured from the top-left pixel corner",
        "resolution": [width, height],
        "colmap": copied,
        "frames": records,
    }
    sessions.write_json(Path(root) / layout.CAMERAS_JSON, payload)
    return payload


# --------------------------------------------------------------------------
# scene_description.json
# --------------------------------------------------------------------------

def _value(socket):
    try:
        value = socket.default_value
    except AttributeError:
        return None
    if hasattr(value, "__len__") and not isinstance(value, str):
        return [float(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _image_info(image):
    if image is None:
        return None
    return {
        "name": image.name_full,
        "filepath": bpy.path.abspath(image.filepath) if image.filepath else "",
        "source": str(image.source),
        "size": [int(v) for v in image.size],
        "colorspace": str(getattr(image.colorspace_settings, "name", "")),
        "packed": image.packed_file is not None,
    }


def _tree_images(tree, seen=None):
    """Images used anywhere in a node tree, including nested groups."""
    seen = seen if seen is not None else set()
    found = []
    if tree is None or tree.as_pointer() in seen:
        return found
    seen.add(tree.as_pointer())
    for node in tree.nodes:
        image = getattr(node, "image", None)
        if image is not None:
            found.append({"node": node.name, "type": node.bl_idname,
                          **_image_info(image)})
        if node.bl_idname in {"ShaderNodeGroup", "GeometryNodeGroup"}:
            found.extend(_tree_images(getattr(node, "node_tree", None), seen))
    return found


def _material_info(material, ids, report):
    info = {
        "id": ids.get(material.name_full),
        "name": material.name_full,
        "library": material.library.filepath if material.library else "",
        "blend_method": str(getattr(material, "surface_render_method",
                                    getattr(material, "blend_method", ""))),
        "use_backface_culling": bool(getattr(material, "use_backface_culling", False)),
        "viewport": {
            "diffuse_color": [float(v) for v in material.diffuse_color],
            "roughness": float(material.roughness),
            "metallic": float(material.metallic),
        },
    }
    info.update({k: v for k, v in (report or {}).items() if k != "name"})
    tree = getattr(material, "node_tree", None)
    if tree is not None:
        _output, bsdf = capture.find_bsdf(tree)
        if bsdf is not None:
            inputs = {}
            for socket in bsdf.inputs:
                if socket.type == "SHADER" or not socket.name:
                    continue
                inputs[socket.name] = {
                    "value": _value(socket),
                    "linked": bool(socket.is_linked),
                }
            info["bsdf"] = {"type": bsdf.bl_idname, "inputs": inputs}
        info["images"] = _tree_images(tree)
    return info


def _light_info(obj):
    data = obj.data
    info = {
        "name": obj.name_full,
        "type": str(data.type),
        "color": [float(v) for v in data.color],
        "energy": float(data.energy),
        "matrix_world": _matrix(obj.matrix_world),
        "use_shadow": bool(getattr(data, "use_shadow", True)),
        "hide_render": bool(obj.hide_render),
    }
    for attr in ("shadow_soft_size", "angle", "spot_size", "spot_blend",
                 "shape", "size", "size_y", "exposure", "normalize"):
        if hasattr(data, attr):
            value = getattr(data, attr)
            info[attr] = value if isinstance(value, (str, bool)) else float(value)
    return info


def _world_info(world, root=None):
    if world is None:
        return None
    info = {"name": world.name_full, "color": [float(v) for v in world.color]}
    tree = getattr(world, "node_tree", None)
    if tree is None:
        return info
    backgrounds = []
    for node in tree.nodes:
        if node.bl_idname == "ShaderNodeBackground":
            backgrounds.append({
                "color": _value(node.inputs.get("Color")),
                "color_linked": bool(node.inputs["Color"].is_linked),
                "strength": _value(node.inputs.get("Strength")),
            })
    info["backgrounds"] = backgrounds
    environments = []
    for node in tree.nodes:
        if node.bl_idname != "ShaderNodeTexEnvironment" or node.image is None:
            continue
        entry = {"node": node.name, "projection": str(node.projection),
                 "image": _image_info(node.image)}
        vector = node.inputs.get("Vector")
        link = vector.links[0] if vector is not None and vector.is_linked else None
        if link is not None and link.from_node.bl_idname == "ShaderNodeMapping":
            mapping = link.from_node
            entry["mapping"] = {
                name: _value(mapping.inputs.get(name))
                for name in ("Location", "Rotation", "Scale")
                if mapping.inputs.get(name) is not None
            }
        environments.append(entry)
    info["environment_textures"] = environments
    return info


def copy_environment_images(world, folder):
    """Copy (or unpack) every environment image the world uses."""
    copied = []
    if world is None or getattr(world, "node_tree", None) is None:
        return copied
    folder = Path(folder)
    for node in world.node_tree.nodes:
        image = getattr(node, "image", None)
        if node.bl_idname != "ShaderNodeTexEnvironment" or image is None:
            continue
        name = Path(image.filepath or image.name).name or "environment"
        target = folder / f"source_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if image.packed_file is not None:
                target.write_bytes(bytes(image.packed_file.data))
            else:
                source = Path(bpy.path.abspath(image.filepath))
                if not source.is_file():
                    continue
                shutil.copy2(source, target)
        except (OSError, AttributeError, TypeError):
            continue
        copied.append(target.name)
    return copied


def scene_description(scene, id_map, material_reports):
    from .. import sceneray_splat, version

    object_ids = {e["name"]: e["id"] for e in (id_map or {}).get("objects", ())}
    material_ids = {e["name"]: e["id"] for e in (id_map or {}).get("materials", ())}
    render = scene.render
    cycles = getattr(scene, "cycles", None)
    unit = scene.unit_settings

    objects, lights, cameras, probes = [], [], [], []
    for obj in sorted(scene.objects, key=lambda o: o.name_full):
        entry = {
            "name": obj.name_full,
            "id": object_ids.get(obj.name_full),
            "type": obj.type,
            "parent": obj.parent.name_full if obj.parent else "",
            "collections": [c.name_full for c in obj.users_collection],
            "matrix_world": _matrix(obj.matrix_world),
            "dimensions": [float(v) for v in obj.dimensions],
            "hide_render": bool(obj.hide_render),
            "visible_camera": bool(getattr(obj, "visible_camera", True)),
            "visible_glossy": bool(getattr(obj, "visible_glossy", True)),
            "visible_transmission": bool(getattr(obj, "visible_transmission", True)),
            "is_holdout": bool(getattr(obj, "is_holdout", False)),
            "pass_index": int(obj.pass_index),
            "materials": [slot.material.name_full if slot.material else ""
                          for slot in obj.material_slots],
            "modifiers": [{"name": m.name, "type": m.type,
                           "show_render": bool(m.show_render)}
                          for m in getattr(obj, "modifiers", ())],
            "instance_collection": (obj.instance_collection.name_full
                                    if obj.instance_type == "COLLECTION"
                                    and obj.instance_collection else ""),
            "library": obj.library.filepath if obj.library else "",
        }
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        entry["bounds_world"] = [
            [min(c[i] for c in corners) for i in range(3)],
            [max(c[i] for c in corners) for i in range(3)],
        ]
        if obj.type == "MESH":
            entry["mesh"] = {"vertices": len(obj.data.vertices),
                             "polygons": len(obj.data.polygons)}
        objects.append(entry)
        if obj.type == "LIGHT":
            lights.append(_light_info(obj))
        elif obj.type == "CAMERA":
            data = obj.data
            cameras.append({
                "name": obj.name_full, "type": str(data.type),
                "lens_mm": float(data.lens), "sensor_width_mm": float(data.sensor_width),
                "sensor_height_mm": float(data.sensor_height),
                "sensor_fit": str(data.sensor_fit),
                "clip_start": float(data.clip_start), "clip_end": float(data.clip_end),
                "matrix_world": _matrix(obj.matrix_world),
                "in_dataset_queue": any(item.camera == obj
                                        for item in scene.SCENERAY_SPLAT.camera_queue),
            })
        elif obj.type == "LIGHT_PROBE":
            data = obj.data
            probes.append({
                "name": obj.name_full, "type": str(data.type),
                "matrix_world": _matrix(obj.matrix_world),
                "clip_start": float(getattr(data, "clip_start", 0.0)),
                "clip_end": float(getattr(data, "clip_end", 0.0)),
                "influence_distance": float(getattr(data, "influence_distance", 0.0)),
            })

    materials = []
    seen = set()
    for obj in scene.objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or material.name_full in seen:
                continue
            seen.add(material.name_full)
            materials.append(_material_info(
                material, material_ids, (material_reports or {}).get(material.name_full)))
    materials.sort(key=lambda m: m["name"])

    def collection_tree(collection):
        return {"name": collection.name_full,
                "hide_render": bool(getattr(collection, "hide_render", False)),
                "objects": sorted(o.name_full for o in collection.objects),
                "children": [collection_tree(c) for c in collection.children]}

    return {
        "format": "SplatGen scene description",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "blend_file": bpy.data.filepath,
            "scene": scene.name_full,
            "blender_version": bpy.app.version_string,
            "addon_version": version.addon_stringversion,
        },
        "units": {"system": str(unit.system), "scale_length": float(unit.scale_length),
                  "length_unit": str(unit.length_unit)},
        "frame": {"current": int(scene.frame_current),
                  "fps": float(render.fps) / float(render.fps_base or 1.0)},
        "render": {
            "engine": str(render.engine),
            "resolution": [int(render.resolution_x), int(render.resolution_y)],
            "resolution_percentage": int(render.resolution_percentage),
            "pixel_aspect": [float(render.pixel_aspect_x), float(render.pixel_aspect_y)],
            "film_transparent": bool(render.film_transparent),
            "use_motion_blur": bool(render.use_motion_blur),
            "cycles": ({
                "samples": int(cycles.samples),
                "use_denoising": bool(getattr(cycles, "use_denoising", False)),
                "max_bounces": int(getattr(cycles, "max_bounces", 0)),
                "diffuse_bounces": int(getattr(cycles, "diffuse_bounces", 0)),
                "glossy_bounces": int(getattr(cycles, "glossy_bounces", 0)),
                "transmission_bounces": int(getattr(cycles, "transmission_bounces", 0)),
                "volume_bounces": int(getattr(cycles, "volume_bounces", 0)),
            } if cycles is not None else None),
            "color_management": sceneray_splat._sr_color_management_snapshot(
                scene, scene.SCENERAY_SPLAT),
        },
        "world": _world_info(scene.world),
        "objects": objects,
        "lights": lights,
        "cameras": cameras,
        "light_probes": probes,
        "materials": materials,
        "collections": collection_tree(scene.collection),
    }


# --------------------------------------------------------------------------
# manifest.json
# --------------------------------------------------------------------------

def _exr_header(path):
    """(channels, pixel type) straight from the file, if OIIO is present."""
    try:
        import OpenImageIO as oiio
    except ImportError:
        return None, None
    handle = oiio.ImageInput.open(str(path))
    if handle is None:
        return None, None
    try:
        spec = handle.spec()
        return list(spec.channelnames), str(spec.format)
    finally:
        handle.close()


_DEFAULT_CHANNELS = {"RGBA": ["R", "G", "B", "A"], "VECTOR": ["X", "Y", "Z"],
                     "FLOAT": ["V"]}


def write_manifest(scene, build, root, frames, resolution, color_management,
                   files, notes, issues):
    from .. import version

    raw = properties.settings(scene)
    root = Path(root)
    previous = sessions.read_json(root / layout.MANIFEST) or {}
    captures = {}
    for source in (layout.BEAUTY, layout.CLAY):
        info = sessions.read_json(layout.work_dir(root) / f"capture_{source}.json") or {}
        captures[source] = info
    unavailable = {}
    for key, entry in (previous.get("passes") or {}).items():
        if entry.get("status") == "unavailable":
            unavailable[key] = entry.get("reason", "")
    for info in captures.values():
        unavailable.update(info.get("unavailable", {}))

    appearance_type = "float" if raw.appearance_precision == "FLOAT" else "half"
    passes = {}
    for key, definition in layout.PASSES.items():
        present = [f for f in frames
                   if layout.pass_file(root, key, f["index"]).is_file()]
        missing = [f["stem"] for f in frames if f not in present]
        if not properties.group_enabled(raw, definition["group"]) and not present:
            status, reason = "disabled", "switched off in the raw export settings"
        elif not present and key in unavailable:
            status, reason = "unavailable", unavailable[key]
        elif not missing:
            status, reason = "complete", ""
        else:
            status, reason = ("partial" if present else "missing"), unavailable.get(key, "")
        channels, pixel_type = (None, None)
        if present:
            channels, pixel_type = _exr_header(
                layout.pass_file(root, key, present[0]["index"]))
        # Blender only honours half precision for RGBA File Output items.
        expected_type = "float"
        if definition["item"] == "RGBA":
            expected_type = {layout.POLICY_APPEARANCE: appearance_type,
                             layout.POLICY_HALF: "half"}.get(definition["policy"], "float")
        entry = {
            "label": definition["label"],
            "group": definition["group"],
            "source": definition["source"],
            "path_pattern": f"{definition['folder']}/{{stem}}.exr",
            "channels": channels or _DEFAULT_CHANNELS[definition["item"]],
            "pixel_type": pixel_type or expected_type,
            "compression": (raw.appearance_codec
                            if definition["policy"] == layout.POLICY_APPEARANCE
                            else "ZIP" if definition["exact"] else raw.data_codec),
            "space": definition["space"],
            "description": definition["description"],
            "status": status,
            "frames_present": len(present),
        }
        if missing and present:
            entry["missing_frames"] = missing
        if reason:
            entry["reason"] = reason
        passes[key] = entry
        if status in {"partial", "missing"} and properties.group_enabled(raw, definition["group"]):
            issues.append(f"{key}: {len(missing)} of {len(frames)} view(s) missing")

    material_report = (captures.get(layout.BEAUTY) or {}).get("materials") \
        or previous.get("materials") or {}
    for info in captures.values():
        for note in info.get("notes", []):
            if note not in notes:
                notes.append(note)

    payload = {
        "schema": layout.SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": {"addon": "SplatGen Prepare", "addon_version": version.addon_stringversion,
                      "blender_version": bpy.app.version_string},
        "source": {"blend_file": bpy.data.filepath, "scene": scene.name_full},
        "path_base": "paths are relative to this Dataset(Raw) folder",
        "resolution": list(resolution),
        "frame_count": len(frames),
        "frames": [{"index": f["index"], "stem": f["stem"], "camera_name": f["camera_name"],
                    "legacy_image": f"../{f['legacy_image']}"} for f in frames],
        "legacy": {
            "dataset": "../Dataset(Default)",
            "sg_metadata": "../sg_metadata",
            "note": "The legacy output is written exactly as before and kept for comparison.",
        },
        "passes": passes,
        "files": files,
        "render": {
            "beauty_engine": str(scene.render.engine),
            "auxiliary_engine": "CYCLES",
            "auxiliary_samples": int(raw.aux_samples) or "scene",
            "storage": {
                "appearance_precision": raw.appearance_precision,
                "appearance_codec": raw.appearance_codec,
                "data_codec": raw.data_codec,
                "exact_passes_codec": "ZIP",
            },
        },
        "color_management": color_management or {},
        "conventions": CONVENTIONS,
        "materials": material_report,
        "notes": notes,
        "status": "complete" if not issues else "incomplete",
        "issues": issues,
    }
    sessions.write_json(root / layout.MANIFEST, payload)
    return payload


README_TEXT = """SplatGen raw dataset
====================

Everything SplatGen could extract from the Blender scene for this build.
manifest.json is the table of contents: every pass, its path pattern
(appearance/combined/{stem}.exr ...), channels, storage, coordinate space and
completeness, and every scene-level file. scene_description.json describes the
objects, lights, materials, world and render settings.

  appearance/        combined beauty, lighting split, clay (lighting only)
  geometry/          position, depth, normal, true normal, uv, object coords,
                     pointiness, backfacing, ambient occlusion
  material/          base color, roughness, metallic, specular, ior,
                     anisotropic, coat, sheen, transmission, subsurface, alpha,
                     emission color/strength, material validity
  ids/               object id, material id, id_map.json
  motion_denoising/  motion vectors, noisy (pre-denoise) image, denoising
                     albedo and normal
  cameras/           cameras.json (intrinsics + poses) and colmap/ copies
  scene/             scene_mesh.ply (+ .json), collision_voxels.npz (+ .json)
  environment/       world/ equirect render + source HDRI, probes/ captures

Per-view files are frame_NNNN.exr and match Dataset(Default)/images/frame_NNNN.*
The legacy ../Dataset(Default) and ../sg_metadata folders are unchanged.
"""


def write_readme(root):
    (Path(root) / layout.README).write_text(README_TEXT, encoding="utf-8")
