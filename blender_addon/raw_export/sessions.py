# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Recipes for the four kinds of raw render.

    beauty  the user's own render, plus every appearance/data pass
    clay    the same view with every surface a neutral grey Lambert
    world   the world alone, as an equirectangular panorama
    probe   the whole scene as an equirectangular panorama from a point

Each builder returns a ready :class:`capture.CaptureSession`; the caller
renders, calls ``finish_view`` per view and ``restore`` at the end. A builder
that fails part-way restores what it already changed before re-raising.
"""

import json
import math
import os
from pathlib import Path

import bpy

from . import capture, layout, properties

#: Equirectangular camera rotation that matches Blender's Environment Texture
#: mapping: image centre looks along world +X, u = 0.25 along +Y, top = +Z.
EQUIRECT_ROTATION = (math.radians(90.0), 0.0, math.radians(-90.0))


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(pending, path)


def save_capture_info(root, session):
    """Remember unavailable passes and material handling for the manifest."""
    path = layout.work_dir(root) / f"capture_{session.source}.json"
    info = read_json(path) or {}
    unavailable = dict(info.get("unavailable", {}))
    unavailable.update(session.unavailable)
    notes = list(info.get("notes", []))
    for note in session.notes:
        if note not in notes:
            notes.append(note)
    info.update({
        "source": session.source,
        "unavailable": unavailable,
        "notes": notes,
        "outputs": sorted(set(info.get("outputs", [])) | set(session.outputs)),
    })
    if session.materials:
        info["materials"] = session.materials
    write_json(path, info)


def _scene_contents(scene):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objects = capture.scene_objects(scene, depsgraph)
    materials = capture.scene_materials(objects, depsgraph)
    return objects, materials


def _fix_resolution(session, resolution):
    render = session.scene.render
    width, height = (int(v) for v in resolution)
    session.set_attr(render, "resolution_x", width)
    session.set_attr(render, "resolution_y", height)
    session.set_attr(render, "resolution_percentage", 100)


def _aux_render_settings(session, raw):
    """Auxiliary renders always use Cycles (pointiness, panoramas)."""
    scene = session.scene
    session.set_attr(scene.render, "engine", "CYCLES")
    cycles = getattr(scene, "cycles", None)
    if cycles is not None and raw.aux_samples > 0:
        session.set_attr(cycles, "samples", int(raw.aux_samples))


# --------------------------------------------------------------------------
# beauty
# --------------------------------------------------------------------------

def begin_beauty(scene, view_layer, root, *, own_tree, resolution=None):
    """Every beauty-source pass, added to one render.

    ``own_tree`` is False inside the legacy render batch, where the legacy
    RGB image must still pass through the user's compositing; the raw nodes
    then join the scene's compositor. The raw passes always come straight
    from the Render Layers node either way.
    """
    raw = properties.settings(scene)
    definitions = capture.selected_passes(raw, layout.BEAUTY)
    capture.purge_leftovers()
    session = capture.CaptureSession(
        scene, view_layer, layout.work_dir(root) / "beauty", layout.BEAUTY
    )
    try:
        if resolution is not None:
            _fix_resolution(session, resolution)
        objects, materials = _scene_contents(scene)
        aov_passes = [d for d in definitions if d.get("aov")]
        if aov_passes:
            capture.inject_material_aovs(session, materials, aov_passes)
        existing = read_json(Path(root) / layout.ID_MAP)
        if raw.ids:
            session.id_map = capture.assign_ids(session, objects, materials, existing)
        else:
            session.id_map = existing or capture.plan_ids(objects, materials)
            session.id_map["mode"] = session.id_map.get("mode", "planned")
        write_json(Path(root) / layout.ID_MAP, session.id_map)
        if own_tree:
            session.use_own_tree()
        else:
            session.use_scene_tree()
        capture.build_outputs(session, raw, definitions, root)
        save_capture_info(root, session)
    except Exception:
        session.restore()
        raise
    return session


# --------------------------------------------------------------------------
# clay
# --------------------------------------------------------------------------

def begin_clay(scene, view_layer, root, resolution):
    raw = properties.settings(scene)
    definitions = capture.selected_passes(raw, layout.CLAY)
    capture.purge_leftovers()
    session = capture.CaptureSession(
        scene, view_layer, layout.work_dir(root) / "clay", layout.CLAY
    )
    try:
        _aux_render_settings(session, raw)
        _fix_resolution(session, resolution)
        objects, _materials = _scene_contents(scene)
        capture.apply_clay(session, objects)
        session.use_own_tree()
        capture.build_outputs(session, raw, definitions, root)
        save_capture_info(root, session)
    except Exception:
        session.restore()
        raise
    return session


# --------------------------------------------------------------------------
# panoramas: world and probes
# --------------------------------------------------------------------------

def _pano_camera(session, collection, name):
    data = bpy.data.cameras.new(f"{capture.PREFIX}{name}")
    session.on_restore(lambda: bpy.data.cameras.remove(data))
    data.type = "PANO"
    for owner in (data, getattr(data, "cycles", None)):
        if owner is not None and hasattr(owner, "panorama_type"):
            owner.panorama_type = "EQUIRECTANGULAR"
    obj = bpy.data.objects.new(f"{capture.PREFIX}{name}", data)
    session.on_restore(lambda: bpy.data.objects.remove(obj))
    collection.objects.link(obj)
    session.on_restore(lambda: collection.objects.unlink(obj))
    obj.rotation_euler = EQUIRECT_ROTATION
    return obj


def begin_world(scene, root):
    """A temporary scene sharing only the world, rendered as a panorama."""
    raw = properties.settings(scene)
    capture.purge_leftovers()
    temporary = bpy.data.scenes.new(f"{capture.PREFIX}World")
    session = capture.CaptureSession(
        temporary, temporary.view_layers[0],
        layout.work_dir(root) / "world", "world",
    )
    session.on_restore(lambda: bpy.data.scenes.remove(temporary))
    try:
        temporary.world = scene.world
        height = int(raw.world_resolution)
        _aux_render_settings(session, raw)
        _fix_resolution(session, (height * 2, height))
        session.set_attr(temporary.render, "film_transparent", False)
        camera = _pano_camera(session, temporary.collection, "WorldCamera")
        camera.location = (0.0, 0.0, 0.0)
        temporary.camera = camera
        session.use_own_tree()
        render_layers = session.render_layers_node()
        depth, codec = capture.exr_settings(raw, {"policy": layout.POLICY_APPEARANCE})
        session.add_output(
            "world", capture.find_socket(render_layers, ("Image",)), "RGBA",
            lambda stem: Path(root) / layout.WORLD_RENDER, "32", codec,
        )
    except Exception:
        session.restore()
        raise
    return session


def begin_probes(scene, view_layer, root):
    raw = properties.settings(scene)
    capture.purge_leftovers()
    session = capture.CaptureSession(
        scene, view_layer, layout.work_dir(root) / "probes", "probes",
    )
    try:
        height = int(raw.probe_resolution)
        _aux_render_settings(session, raw)
        _fix_resolution(session, (height * 2, height))
        render = scene.render
        session.set_attr(render, "pixel_aspect_x", 1.0)
        session.set_attr(render, "pixel_aspect_y", 1.0)
        session.set_attr(render, "use_border", False)
        session.set_attr(render, "film_transparent", False)
        camera = _pano_camera(session, scene.collection, "ProbeCamera")
        session.set_attr(scene, "camera", camera)
        session.probe_camera = camera
        capture.enable_pass_flags(session, [layout.PASSES["depth"]])
        capture.update_render_passes(session)
        session.use_own_tree()
        render_layers = session.render_layers_node()
        _depth, codec = capture.exr_settings(raw, {"policy": layout.POLICY_APPEARANCE})
        probes = Path(root) / layout.PROBES_FOLDER
        session.add_output(
            "radiance", capture.find_socket(render_layers, ("Image",)), "RGBA",
            lambda stem: probes / stem / "radiance.exr", "32", codec,
        )
        socket = capture.find_socket(render_layers, ("Depth", "Z"))
        if socket is not None:
            session.add_output(
                "distance", socket, "FLOAT",
                lambda stem: probes / stem / "distance.exr", "32", "ZIP",
            )
    except Exception:
        session.restore()
        raise
    return session
