# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Temporary scene changes that make Blender render the raw passes.

A :class:`CaptureSession` is the only thing that touches the user's scene
for the raw dataset. Every change it makes - a pass switch, an AOV, a node in
a material, an object's pass index, the material override - is recorded on
an undo stack as it is made, and :meth:`CaptureSession.restore` replays that
stack backwards. Nothing is restored by guessing what the value used to be.

Leftovers from a crash are recognisable by ``PREFIX`` and removed by
:func:`purge_leftovers` the next time the add-on loads or a capture starts.

Per-view files are written by one compositor File Output node per pass,
straight from the Render Layers node, so the user's own compositing never
alters raw data. Each node writes ``.<stem>.pending_<key>.exr`` into its own
work folder; :meth:`CaptureSession.finish_view` moves them into place.
"""

import os
import time
from pathlib import Path

import bpy

from . import layout

PREFIX = "__SPLATGEN_RAW__"
AOV_PREFIX = "sg_"

_ITEM_COLOR_MODE = {"RGBA": "RGBA", "VECTOR": "RGB", "FLOAT": "BW"}


# --------------------------------------------------------------------------
# Session: an undo stack around every scene change
# --------------------------------------------------------------------------

class CaptureSession:
    def __init__(self, scene, view_layer, work_root, source):
        self.scene = scene
        self.view_layer = view_layer
        self.work_root = Path(work_root)
        self.source = source
        self.undo = []
        self.tree = None
        self.outputs = {}
        self.unavailable = {}
        self.notes = []
        self.materials = {}
        self.id_map = None
        self.restored = False

    # -- recording ---------------------------------------------------------

    def set_attr(self, owner, attr, value):
        """Set ``owner.attr`` and remember how to put it back."""
        old = getattr(owner, attr)
        if old == value:
            return True
        try:
            setattr(owner, attr, value)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

        def undo(owner=owner, attr=attr, old=old):
            setattr(owner, attr, old)

        self.undo.append(undo)
        return True

    def on_restore(self, callback):
        self.undo.append(callback)

    def restore(self):
        if self.restored:
            return
        self.restored = True
        while self.undo:
            callback = self.undo.pop()
            try:
                callback()
            except (AttributeError, ReferenceError, RuntimeError, TypeError,
                    ValueError, KeyError):
                pass

    # -- compositor ----------------------------------------------------------

    def use_scene_tree(self):
        """Add nodes to the scene's compositor, as the legacy capture does.

        Used for the beauty render inside the legacy batch, where the legacy
        RGB image still has to go through the user's own compositing.
        """
        from ..building_data import dataset_export as bd_export

        tree, state = bd_export._tree_for_scene(self.scene)
        scene = self.scene

        def undo():
            if hasattr(scene, "compositing_node_group"):
                scene.compositing_node_group = state.get("old_group")
            created = state.get("created_group")
            if created is not None:
                bpy.data.node_groups.remove(created)

        self.on_restore(undo)
        self.tree = tree
        return tree

    def use_own_tree(self):
        """A private compositor, so auxiliary renders ignore user compositing."""
        tree = bpy.data.node_groups.new(f"{PREFIX}Compositor", "CompositorNodeTree")
        self.on_restore(lambda: bpy.data.node_groups.remove(tree))
        if hasattr(self.scene, "compositing_node_group"):
            self.set_attr(self.scene, "compositing_node_group", tree)
        self.set_attr(self.scene.render, "use_compositing", True)
        self.tree = tree
        return tree

    def _track_node(self, node):
        tree = self.tree

        def undo():
            tree.nodes.remove(node)

        self.on_restore(undo)

    def render_layers_node(self):
        node = self.tree.nodes.new("CompositorNodeRLayers")
        node.name = f"{PREFIX}RenderLayers"
        node.label = "SplatGen raw pass source"
        node.scene = self.scene
        try:
            node.layer = self.view_layer.name
        except (AttributeError, TypeError):
            pass
        self._track_node(node)
        return node

    def add_output(self, key, source_socket, item, target, depth, codec):
        """One File Output node writing one pass as a single-layer EXR."""
        folder = self.work_root / key
        folder.mkdir(parents=True, exist_ok=True)
        node = self.tree.nodes.new("CompositorNodeOutputFile")
        node.name = f"{PREFIX}{key}"
        node.label = f"SplatGen raw: {key}"
        self._track_node(node)
        node.directory = str(folder)
        node.file_name = ".pending_"
        fmt = node.format
        try:
            fmt.media_type = "IMAGE"
        except (AttributeError, TypeError):
            pass
        fmt.file_format = "OPEN_EXR"
        try:
            fmt.color_mode = _ITEM_COLOR_MODE[item]
        except (TypeError, KeyError):
            pass
        fmt.color_depth = depth
        try:
            fmt.exr_codec = codec
        except (AttributeError, TypeError):
            pass
        try:
            node.save_as_render = False
        except (AttributeError, TypeError):
            pass
        node.file_output_items.new(item, key)
        self.tree.links.new(source_socket, node.inputs[key])
        self.outputs[key] = {"node": node, "folder": folder, "target": target}
        return node

    # -- per view --------------------------------------------------------------

    def prepare_view(self, stem):
        pattern = f".{stem}.pending_*.exr"
        for output in self.outputs.values():
            for stale in output["folder"].glob(pattern):
                try:
                    stale.unlink()
                except OSError:
                    pass
            output["node"].file_name = f".{stem}.pending_"

    def finish_view(self, stem, timeout=8.0):
        """Move this view's files into place. Returns (published, missing)."""
        published, missing = {}, []
        deadline = time.monotonic() + float(timeout)
        for key, output in self.outputs.items():
            source = _wait_for(output["folder"], stem, deadline)
            if source is None:
                missing.append(key)
                continue
            target = Path(output["target"](stem))
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            published[key] = target
        return published, missing


def _wait_for(folder, stem, deadline):
    pattern = f".{stem}.pending_*.exr"
    while True:
        found = [path for path in folder.glob(pattern) if path.is_file()]
        if found:
            found.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
            for extra in found[1:]:
                try:
                    extra.unlink()
                except OSError:
                    pass
            return found[0]
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def find_socket(node, names):
    for name in names:
        socket = node.outputs.get(name)
        if socket is not None and getattr(socket, "enabled", True):
            return socket
    for socket in node.outputs:
        if socket.identifier in names and getattr(socket, "enabled", True):
            return socket
    return None


# --------------------------------------------------------------------------
# Pass switches
# --------------------------------------------------------------------------

def enable_pass_flags(session, definitions):
    layer = session.view_layer
    cycles = getattr(layer, "cycles", None)
    for definition in definitions:
        for owner_name, attr in definition["flags"]:
            owner = layer if owner_name == "layer" else cycles
            if owner is None or not hasattr(owner, attr):
                # Volume passes moved between releases; try the other owner.
                owner = cycles if owner is layer else layer
            if owner is None or not hasattr(owner, attr):
                session.unavailable[definition["key"]] = (
                    f"this Blender build has no {attr} switch"
                )
                continue
            session.set_attr(owner, attr, True)


def add_aovs(session, definitions):
    layer = session.view_layer
    existing = {aov.name for aov in layer.aovs}
    for definition in definitions:
        aov = definition.get("aov")
        if not aov or aov[0] in existing:
            continue
        item = layer.aovs.add()
        item.name = aov[0]
        item.type = aov[1]
        existing.add(aov[0])
        name = aov[0]

        def undo(layer=layer, name=name):
            for candidate in layer.aovs:
                if candidate.name == name:
                    layer.aovs.remove(candidate)
                    break

        session.on_restore(undo)


def update_render_passes(session):
    try:
        session.view_layer.update_render_passes()
    except (AttributeError, RuntimeError):
        pass


# --------------------------------------------------------------------------
# What the render can see
# --------------------------------------------------------------------------

def scene_objects(scene, depsgraph=None):
    """Every object that can reach the render, instances included."""
    found = {}
    for obj in scene.objects:
        found[obj.name_full] = obj
    if depsgraph is not None:
        try:
            for instance in depsgraph.object_instances:
                original = getattr(instance.object, "original", None)
                if original is not None:
                    found.setdefault(original.name_full, original)
        except (AttributeError, ReferenceError, RuntimeError):
            pass
    return [found[name] for name in sorted(found)]


def scene_materials(objects, depsgraph=None):
    found = {}
    for obj in objects:
        for slot in getattr(obj, "material_slots", ()):
            material = slot.material
            if material is not None:
                found.setdefault(material.name_full, material)
        data = getattr(obj, "data", None)
        for material in getattr(data, "materials", ()) or ():
            if material is not None:
                found.setdefault(material.name_full, material)
    if depsgraph is not None:
        # Geometry Nodes can set materials the object slots never mention.
        try:
            for instance in depsgraph.object_instances:
                data = getattr(instance.object, "data", None)
                for material in getattr(data, "materials", ()) or ():
                    original = getattr(material, "original", material)
                    if original is not None:
                        found.setdefault(original.name_full, original)
        except (AttributeError, ReferenceError, RuntimeError):
            pass
    return [found[name] for name in sorted(found)]


def _editable(idblock):
    if getattr(idblock, "library", None) is not None:
        return False
    return bool(getattr(idblock, "is_editable", True))


# --------------------------------------------------------------------------
# Shader AOVs: material properties and geometry read inside each material
# --------------------------------------------------------------------------

#: Non-Principled BSDFs: channel -> input name, or a constant.
_OTHER_BSDFS = {
    "ShaderNodeBsdfDiffuse": {"base_color": "Color", "roughness": "Roughness"},
    "ShaderNodeBsdfGlossy": {"base_color": "Color", "roughness": "Roughness",
                             "anisotropic": "Anisotropy", "metallic": 1.0},
    "ShaderNodeBsdfAnisotropic": {"base_color": "Color", "roughness": "Roughness",
                                  "anisotropic": "Anisotropy", "metallic": 1.0},
    "ShaderNodeBsdfMetallic": {"base_color": "Base Color", "roughness": "Roughness",
                               "anisotropic": "Anisotropy", "metallic": 1.0},
    "ShaderNodeBsdfGlass": {"base_color": "Color", "roughness": "Roughness",
                            "ior": "IOR", "transmission": 1.0},
    "ShaderNodeBsdfRefraction": {"base_color": "Color", "roughness": "Roughness",
                                 "ior": "IOR", "transmission": 1.0},
    "ShaderNodeEmission": {"emission_color": "Color",
                           "emission_strength": "Strength",
                           "base_color": (0.0, 0.0, 0.0, 1.0)},
    "ShaderNodeSubsurfaceScattering": {"base_color": "Color", "subsurface": 1.0},
    "ShaderNodeBsdfTransparent": {"base_color": "Color", "alpha": 0.0},
    "ShaderNodeBsdfSheen": {"base_color": "Color", "roughness": "Roughness",
                            "sheen": 1.0},
    "ShaderNodeBsdfToon": {"base_color": "Color"},
    "ShaderNodeBsdfTranslucent": {"base_color": "Color"},
}
_PASSTHROUGH = {"ShaderNodeMixShader": (1, 2), "ShaderNodeAddShader": (0, 1),
                "NodeReroute": (0,)}


def _defaults(material):
    diffuse = tuple(material.diffuse_color) if material is not None else (0.8, 0.8, 0.8, 1.0)
    return {
        "base_color": tuple(diffuse[:3]) + (1.0,),
        "roughness": float(getattr(material, "roughness", 0.5)),
        "metallic": float(getattr(material, "metallic", 0.0)),
        "specular": 0.5,
        "ior": 1.5,
        "anisotropic": 0.0,
        "coat": 0.0,
        "sheen": 0.0,
        "transmission": 0.0,
        "subsurface": 0.0,
        "alpha": float(diffuse[3]) if len(diffuse) > 3 else 1.0,
        "emission_color": (0.0, 0.0, 0.0, 1.0),
        "emission_strength": 0.0,
    }


def _active_output(tree):
    outputs = [node for node in tree.nodes
               if node.bl_idname == "ShaderNodeOutputMaterial"]
    for target in ("CYCLES", "ALL", "EEVEE"):
        for node in outputs:
            if getattr(node, "is_active_output", True) and getattr(node, "target", "ALL") == target:
                return node
    return outputs[0] if outputs else None


def _live_link(socket):
    for link in getattr(socket, "links", ()):
        if not getattr(link, "is_muted", False) and getattr(link, "is_valid", True):
            return link
    return None


def find_bsdf(tree):
    """The BSDF that best describes the surface: first Principled upstream
    of the active output, else the first other known BSDF."""
    output = _active_output(tree)
    if output is None:
        return None, None
    surface = output.inputs.get("Surface")
    link = _live_link(surface) if surface is not None else None
    if link is None:
        return output, None
    queue, seen, other = [link.from_node], set(), None
    while queue:
        node = queue.pop(0)
        if node is None or node.as_pointer() in seen:
            continue
        seen.add(node.as_pointer())
        if node.bl_idname == "ShaderNodeBsdfPrincipled":
            return output, node
        if other is None and node.bl_idname in _OTHER_BSDFS:
            other = node
        for index in _PASSTHROUGH.get(node.bl_idname, ()):
            if index < len(node.inputs):
                upstream = _live_link(node.inputs[index])
                if upstream is not None:
                    queue.append(upstream.from_node)
    return output, other


def _volume_only(material):
    tree = getattr(material, "node_tree", None)
    if tree is None:
        return False
    output = _active_output(tree)
    if output is None:
        return False
    surface = output.inputs.get("Surface")
    volume = output.inputs.get("Volume")
    return bool(volume is not None and _live_link(volume) is not None
                and (surface is None or _live_link(surface) is None))


def inject_material_aovs(session, materials, definitions):
    """Add AOV Output nodes to each material for the requested passes.

    Linked shader inputs are wired straight into the AOV, so textures and
    procedural networks are evaluated per pixel exactly as the BSDF sees
    them. Unlinked inputs become constants. The nodes are removed again by
    the session's restore.
    """
    wanted = [d for d in definitions if d.get("aov")]
    if not wanted:
        return {}
    channels = {key: inputs for key, _type, inputs, _label in layout.MATERIAL_CHANNELS}
    report = {}
    for material in materials:
        entry = {"name": material.name_full, "editable": _editable(material)}
        report[material.name_full] = entry
        tree = getattr(material, "node_tree", None)
        if not entry["editable"]:
            entry["shader"] = "not editable (linked library data)"
            continue
        if tree is None:
            entry["shader"] = "no node tree"
            continue
        output, bsdf = find_bsdf(tree)
        kind = bsdf.bl_idname if bsdf is not None else ""
        if kind == "ShaderNodeBsdfPrincipled":
            entry["shader"], validity = "principled", 1.0
        elif bsdf is not None:
            entry["shader"], validity = kind, 0.5
        else:
            entry["shader"], validity = "viewport fallback", 0.25
        entry["linked_inputs"] = []
        defaults = _defaults(material)
        created = []

        def new(idname):
            node = tree.nodes.new(idname)
            node.name = f"{PREFIX}{idname}"
            node.label = "SplatGen raw (temporary)"
            node.location = (-4000.0, -300.0 * len(created))
            created.append(node)
            return node

        geometry = texcoord = None
        for definition in wanted:
            key = definition["key"]
            aov_name, aov_type = definition["aov"]
            socket_name = "Color" if aov_type == "COLOR" else "Value"
            aov = new("ShaderNodeOutputAOV")
            aov.aov_name = aov_name
            target = aov.inputs[socket_name]
            source = None
            constant = None
            if key in channels:
                if kind == "ShaderNodeBsdfPrincipled":
                    for name in channels[key]:
                        found = bsdf.inputs.get(name)
                        if found is not None:
                            link = _live_link(found)
                            if link is not None:
                                source = link.from_socket
                            else:
                                constant = found.default_value
                            break
                elif bsdf is not None:
                    mapped = _OTHER_BSDFS[kind].get(key)
                    if isinstance(mapped, str):
                        found = bsdf.inputs.get(mapped)
                        if found is not None:
                            link = _live_link(found)
                            if link is not None:
                                source = link.from_socket
                            else:
                                constant = found.default_value
                    elif mapped is not None:
                        constant = mapped
                if source is None and constant is None:
                    constant = defaults[key]
            elif key == "material_valid":
                constant = validity
            elif key in {"true_normal", "pointiness", "backfacing"}:
                if geometry is None:
                    geometry = new("ShaderNodeNewGeometry")
                source = geometry.outputs[{
                    "true_normal": "True Normal",
                    "pointiness": "Pointiness",
                    "backfacing": "Backfacing",
                }[key]]
            elif key == "object_coords":
                if texcoord is None:
                    texcoord = new("ShaderNodeTexCoord")
                source = texcoord.outputs["Object"]
            if source is not None and getattr(source, "type", "") != "SHADER":
                tree.links.new(source, target)
                if key in channels:
                    entry["linked_inputs"].append(key)
            else:
                _set_default(target, constant, aov_type)

        def undo(tree=tree, created=created):
            for node in created:
                try:
                    tree.nodes.remove(node)
                except (ReferenceError, RuntimeError):
                    pass

        session.on_restore(undo)
    session.materials = report
    return report


def _set_default(socket, value, aov_type):
    if value is None:
        return
    if aov_type == "COLOR":
        if isinstance(value, (int, float)):
            value = (float(value),) * 3 + (1.0,)
        value = tuple(float(v) for v in value)
        if len(value) == 3:
            value = value + (1.0,)
        socket.default_value = value[:4]
    else:
        if not isinstance(value, (int, float)):
            components = [float(v) for v in value][:3]
            value = sum(components) / max(1, len(components))
        socket.default_value = float(value)


# --------------------------------------------------------------------------
# Object and material ids
# --------------------------------------------------------------------------

def _uses_index_nodes(materials):
    """A material reading Object/Material Index would change its look."""
    for material in materials:
        tree = getattr(material, "node_tree", None)
        if tree is None:
            continue
        for node in tree.nodes:
            if node.bl_idname != "ShaderNodeObjectInfo":
                continue
            for name in ("Object Index", "Material Index"):
                socket = node.outputs.get(name)
                if socket is not None and socket.is_linked:
                    return material.name_full
    return None


def plan_ids(objects, materials, existing=None):
    """Unique, stable ids by name, without touching the scene.

    Ids already recorded for a name (``existing``, from an earlier run into
    the same dataset) are kept, so views rendered at different times agree.
    New names are numbered after the highest id in use; 0 is background.
    """
    existing = existing or {}
    result = {"background_id": 0, "mode": "assigned",
              "objects": [], "materials": [], "not_assigned": []}
    for field, blocks in (("objects", objects), ("materials", materials)):
        known = {entry["name"]: int(entry["id"])
                 for entry in existing.get(field, ()) if "name" in entry}
        next_id = max(known.values(), default=0) + 1
        for block in blocks:
            name = block.name_full
            if name not in known:
                known[name] = next_id
                next_id += 1
            entry = {"id": known[name], "name": name}
            if field == "objects":
                entry["type"] = block.type
            result[field].append(entry)
    return result


def assign_ids(session, objects, materials, existing=None):
    """Write planned ids into ``pass_index`` for the duration of the render.

    If any material reads the pass index, the user's own indices are left
    alone and recorded instead - reassigning them would change the render.
    """
    blocker = _uses_index_nodes(materials)
    if blocker:
        return {
            "background_id": 0,
            "mode": "user_pass_index",
            "note": (f"Material '{blocker}' reads Object/Material Index, so ids "
                     "are the scene's own pass indices and may not be unique."),
            "objects": [{"id": int(o.pass_index), "name": o.name_full, "type": o.type}
                        for o in objects],
            "materials": [{"id": int(m.pass_index), "name": m.name_full}
                          for m in materials],
            "not_assigned": [],
        }
    result = plan_ids(objects, materials, existing)
    for field, blocks in (("objects", objects), ("materials", materials)):
        by_name = {entry["name"]: entry for entry in result[field]}
        for block in blocks:
            entry = by_name[block.name_full]
            if _editable(block) and session.set_attr(block, "pass_index", entry["id"]):
                continue
            if int(block.pass_index) != entry["id"]:
                # Linked library data cannot be changed; report what renders.
                result["not_assigned"].append(block.name_full)
                entry["id"] = int(block.pass_index)
    return result


# --------------------------------------------------------------------------
# Clay
# --------------------------------------------------------------------------

def apply_clay(session, objects):
    material = bpy.data.materials.new(f"{PREFIX}Clay")
    session.on_restore(lambda: bpy.data.materials.remove(material))
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    diffuse.inputs["Roughness"].default_value = 0.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
    session.set_attr(session.view_layer, "material_override", material)
    # The override would turn fog and smoke boxes into solid grey blocks.
    hidden = []
    for obj in objects:
        slots = [slot.material for slot in getattr(obj, "material_slots", ())
                 if slot.material is not None]
        if slots and all(_volume_only(m) for m in slots) and _editable(obj):
            if session.set_attr(obj, "hide_render", True):
                hidden.append(obj.name_full)
    if hidden:
        session.notes.append(
            "Volume-only objects hidden from the clay render: " + ", ".join(hidden)
        )
    return material


# --------------------------------------------------------------------------
# Building a capture for a set of passes
# --------------------------------------------------------------------------

def exr_settings(raw, definition):
    """(color_depth, codec) for one pass under the user's storage settings."""
    policy = definition["policy"]
    if policy == layout.POLICY_APPEARANCE:
        depth = "32" if raw.appearance_precision == "FLOAT" else "16"
        return depth, raw.appearance_codec
    return ("16" if policy == layout.POLICY_HALF else "32"), "ZIP"


def selected_passes(raw, source):
    from . import properties

    return [d for d in layout.passes_for(source)
            if properties.group_enabled(raw, d["group"])]


def build_outputs(session, raw, definitions, raw_root):
    """Enable, source and wire every pass; unavailable ones are recorded."""
    enable_pass_flags(session, definitions)
    add_aovs(session, definitions)
    update_render_passes(session)
    render_layers = session.render_layers_node()
    motion_blur = bool(getattr(session.scene.render, "use_motion_blur", False))
    for definition in definitions:
        key = definition["key"]
        if key in session.unavailable:
            continue
        if key == "motion_vector" and motion_blur:
            session.unavailable[key] = (
                "Cycles does not write motion vectors while motion blur is on"
            )
            continue
        socket = find_socket(render_layers, definition["sockets"])
        if socket is None:
            session.unavailable[key] = (
                f"render engine {session.scene.render.engine} does not provide "
                f"{definition['sockets'][0]}"
            )
            continue
        depth, codec = exr_settings(raw, definition)
        session.add_output(
            key, socket, definition["item"],
            lambda stem, key=key: Path(raw_root) / layout.PASSES[key]["folder"] / f"{stem}.exr",
            depth, codec,
        )


def purge_leftovers():
    """Remove anything a crashed capture may have left in the file."""
    try:
        materials = list(bpy.data.materials)
    except AttributeError:
        return
    for material in materials:
        if material.name.startswith(PREFIX):
            try:
                bpy.data.materials.remove(material)
            except (ReferenceError, RuntimeError):
                pass
            continue
        tree = getattr(material, "node_tree", None)
        if tree is None or not _editable(material):
            continue
        for node in [n for n in tree.nodes if n.name.startswith(PREFIX)]:
            try:
                tree.nodes.remove(node)
            except (ReferenceError, RuntimeError):
                pass
    for group in [g for g in bpy.data.node_groups if g.name.startswith(PREFIX)]:
        try:
            bpy.data.node_groups.remove(group)
        except (ReferenceError, RuntimeError):
            pass
    for obj in [o for o in bpy.data.objects if o.name.startswith(PREFIX)]:
        try:
            bpy.data.objects.remove(obj)
        except (ReferenceError, RuntimeError):
            pass
    for camera in [c for c in bpy.data.cameras if c.name.startswith(PREFIX)]:
        try:
            bpy.data.cameras.remove(camera)
        except (ReferenceError, RuntimeError):
            pass
    for scene in [s for s in bpy.data.scenes if s.name.startswith(PREFIX)]:
        try:
            bpy.data.scenes.remove(scene)
        except (ReferenceError, RuntimeError):
            pass
