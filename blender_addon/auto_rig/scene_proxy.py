# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Read the render-visible scene into a ``GeometrySet``. Main thread only.

The same rule as every other scene analysis in the add-on: if an object
would not appear in a render, it does not exist here. Evaluated geometry
comes from the dependency graph, so modifiers, geometry nodes, particles and
collection instances are all included exactly as they render.

Speed comes from reading each unique evaluated mesh once with
``foreach_get`` into numpy, then recording only a matrix per instance.
"""

import numpy as np

from .geometry import GeometrySet

#: Object types that become triangles when evaluated.
GEOMETRY_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}


def _active_objects(scene, view_layer):
    """Pointers of objects linked into a collection this View Layer renders.

    Built once by walking collections. ``Object.users_collection`` answers
    the same question per object, but scans every collection in the file on
    each call - with tens of thousands of instances that alone took seconds.
    """
    active = set()

    def walk(layer_collection, excluded=False):
        collection = layer_collection.collection
        excluded = (excluded or layer_collection.exclude
                    or bool(getattr(collection, "hide_render", False))
                    or bool(getattr(layer_collection, "holdout", False))
                    or bool(getattr(layer_collection, "indirect_only", False)))
        if not excluded:
            active.update(obj.as_pointer() for obj in collection.objects)
        for child in layer_collection.children:
            walk(child, excluded)

    if view_layer is None:
        return None
    walk(view_layer.layer_collection)
    return active


def _renders(obj, active):
    """Whether one original object can put pixels in the final image."""
    try:
        if obj.hide_render or not getattr(obj, "visible_camera", True):
            return False
        if getattr(obj, "is_holdout", False):
            return False
    except ReferenceError:
        return False
    return active is None or obj.as_pointer() in active


def _read_mesh(mesh):
    """Local vertices and triangle indices of an evaluated mesh."""
    count = len(mesh.vertices)
    if not count:
        return None, None
    verts = np.empty(count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts)
    triangles = mesh.loop_triangles
    tri_count = len(triangles)
    if not tri_count:
        return None, None
    tris = np.empty(tri_count * 3, dtype=np.int32)
    triangles.foreach_get("vertices", tris)
    return verts, tris


def extract(context, collection=None, exclude=()):
    """The scene's render-visible surfaces as a ``GeometrySet``.

    ``collection`` restricts the scan to one collection (recursively);
    ``exclude`` names objects never to include, such as the blobs.
    """
    from .. import sceneray_splat

    scene, view_layer = context.scene, context.view_layer
    active = _active_objects(scene, view_layer)
    mesh_ok = {o.as_pointer() for o in
               sceneray_splat.sr_renderable_mesh_objects(scene, view_layer)}
    boolean_operands = sceneray_splat._sr_boolean_operand_keys(scene)
    allowed = None
    if collection is not None:
        allowed = {o.as_pointer() for o in collection.all_objects}
    excluded = set(exclude)

    geometry = GeometrySet()
    mesh_cache = {}
    owner_index = {}
    visible_cache = {}

    def original_ok(original):
        key = original.as_pointer()
        if key not in visible_cache:
            if original.type == 'MESH':
                ok = key in mesh_ok
            else:
                ok = _renders(original, active) and key not in boolean_operands
            visible_cache[key] = ok
        return visible_cache[key]

    owner_cache = {}

    def instancer_ok(owner):
        key = owner.as_pointer()
        if key not in owner_cache:
            owner_cache[key] = _renders(owner, active)
        return owner_cache[key]

    depsgraph = context.evaluated_depsgraph_get()
    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type not in GEOMETRY_TYPES:
            continue
        source = obj.original
        if instance.is_instance:
            parent = instance.parent
            owner = parent.original if parent is not None else source
            # An instancer that does not render hides its instances too.
            if not instancer_ok(owner):
                continue
            if source.hide_render:
                continue
        else:
            owner = source
            if not original_ok(source):
                continue
            if source.is_instancer and not source.show_instancer_for_render:
                continue
        if owner.name in excluded:
            continue
        if allowed is not None and (owner.as_pointer() not in allowed
                                    and source.as_pointer() not in allowed):
            continue

        data = obj.data
        if obj.type == 'MESH' and data is not None:
            key = (data.as_pointer(), len(data.vertices), len(data.loop_triangles))
        else:
            key = ("other", source.as_pointer())
        if key not in mesh_cache:
            index = -1
            try:
                if obj.type == 'MESH':
                    verts, tris = _read_mesh(data)
                else:
                    mesh = obj.to_mesh()
                    try:
                        verts, tris = _read_mesh(mesh) if mesh is not None else (None, None)
                    finally:
                        obj.to_mesh_clear()
                if verts is not None:
                    index = geometry.add_mesh(verts, tris)
            except (RuntimeError, ReferenceError, AttributeError):
                index = -1
            mesh_cache[key] = index
        index = mesh_cache[key]
        if index < 0:
            continue
        owner_key = owner.as_pointer()
        if owner_key not in owner_index:
            owner_index[owner_key] = geometry.add_owner(owner.name)
        geometry.add_instance(index, np.array(instance.matrix_world, dtype=np.float64),
                              owner_index[owner_key])
    return geometry.finalize()
