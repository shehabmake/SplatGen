# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""One ray-castable copy of exactly what the render will contain.

Every scene analysis in the add-on - coverage validation, the point cloud,
scene bounds - has to agree with the final images. That means one rule: if an
object would not appear in a render, it does not exist here. Not as a hit, not
as an occluder, not as a contributor to bounds.

The old approach cast into the live scene and then *skipped* unwanted hits,
re-casting up to 64 times per ray to get past hidden geometry. That was both
slow and fragile: an object hidden from the render still blocked the ray until
it was skipped past, and every skip cost another full scene cast.

This builds a BVH from the evaluated meshes of the render-visible objects
only. Visibility is then enforced by construction - hidden geometry is simply
absent - and each ray is a single cast into a tree built once for the whole
operation rather than once per ray.

Evaluated meshes come from the dependency graph, so modifiers, geometry nodes
and instancing are all already applied.
"""

import time

import bpy
from mathutils.bvhtree import BVHTree

#: Rebuilt when the scene changes; a build is far cheaper than the casts it
#: serves, but not cheap enough to repeat per camera.
_cache = {"key": None, "scene": None, "built_at": 0.0}

CACHE_SECONDS = 30.0


class RenderVisibleScene:
    """A BVH over the geometry a render would actually show."""

    def __init__(self, tree, triangle_objects, triangle_materials, objects,
                 minimum, maximum, surface_samples=None):
        self.tree = tree
        self.triangle_objects = triangle_objects
        self.triangle_materials = triangle_materials
        self.objects = objects
        self.minimum = minimum
        self.maximum = maximum
        self.surface_samples = surface_samples

    @property
    def is_empty(self):
        return self.tree is None

    def ray_cast(self, origin, direction):
        """``(location, normal, index, distance)`` or all-None on a miss.

        A single cast: anything hidden from the render is not in the tree, so
        there is nothing to skip past.
        """
        if self.tree is None:
            return (None, None, None, None)
        return self.tree.ray_cast(origin, direction)

    def object_for(self, index):
        if index is None:
            return None
        if 0 <= index < len(self.triangle_objects):
            return self.triangle_objects[index]
        return None

    def material_for(self, index):
        """Material on the exact evaluated triangle hit by the shared BVH."""
        if index is None:
            return None
        if 0 <= index < len(self.triangle_materials):
            return self.triangle_materials[index]
        return None


def _visible_objects(context, cfg):
    from .. import sceneray_splat

    return sceneray_splat.sr_renderable_mesh_objects(
        context.scene, context.view_layer
    )


def _scene_key(context, objects):
    """Cheap identity for the geometry that would be built."""
    return (
        context.scene.name,
        getattr(context.view_layer, "name", ""),
        tuple(sorted(obj.name for obj in objects)),
    )


def _surface_samples(vertices, triangles, budget):
    """Bounded, deterministic area survey, independent of every camera.

    Evaluate triangle areas in chunks so large meshes do not require another
    full triangle-coordinate array. Only the small survey survives this build.
    """
    import numpy as np
    xyz = np.asarray(vertices, dtype=np.float64)
    areas = np.zeros(len(triangles), dtype=np.float64)
    for start in range(0, len(triangles), 32768):
        tri = xyz[np.asarray(triangles[start:start+32768], dtype=np.int64)]
        areas[start:start+len(tri)] = np.linalg.norm(
            np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) * .5
    total = float(areas.sum())
    if not total > 0 or not np.isfinite(total):
        return None
    face_ids = np.searchsorted(np.cumsum(areas),
        (np.arange(budget)+.5)*(total/budget), side='right')
    tri = xyz[np.asarray([triangles[int(i)] for i in face_ids], dtype=np.int64)]
    def radical(index, base):
        value, scale = 0., 1./base
        while index:
            index, remainder = divmod(index, base)
            value += remainder*scale
            scale /= base
        return value
    uv = np.array([(radical(i+1, 2), radical(i+1, 3)) for i in range(budget)])
    root = np.sqrt(uv[:, 0:1])
    positions = (1-root)*tri[:, 0] + root*(1-uv[:, 1:2])*tri[:, 1] + root*uv[:, 1:2]*tri[:, 2]
    normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    return positions, normals, face_ids, total


def build(context, cfg=None, force=False, surface_budget=0):
    """The render-visible scene, built or reused from the cache."""
    objects = _visible_objects(context, cfg)
    key = (_scene_key(context, objects), surface_budget)
    cached = _cache["scene"]
    if (
        not force
        and cached is not None
        and _cache["key"] == key
        and time.time() - _cache["built_at"] < CACHE_SECONDS
    ):
        return cached

    depsgraph = context.evaluated_depsgraph_get()
    vertices = []
    polygons = []
    triangle_objects = []
    triangle_materials = []
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3

    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        try:
            mesh = evaluated.to_mesh()
        except (RuntimeError, AttributeError):
            continue
        if mesh is None:
            continue
        try:
            mesh.calc_loop_triangles()
            matrix = evaluated.matrix_world
            offset = len(vertices)
            for vertex in mesh.vertices:
                world = matrix @ vertex.co
                vertices.append(world)
                for axis in range(3):
                    if world[axis] < minimum[axis]:
                        minimum[axis] = world[axis]
                    if world[axis] > maximum[axis]:
                        maximum[axis] = world[axis]
            for triangle in mesh.loop_triangles:
                polygons.append(
                    tuple(offset + index for index in triangle.vertices)
                )
                triangle_objects.append(obj)
                material = None
                try:
                    polygon = mesh.polygons[triangle.polygon_index]
                    slot_index = int(polygon.material_index)
                    slots = evaluated.material_slots
                    if 0 <= slot_index < len(slots):
                        material = slots[slot_index].material
                except (AttributeError, IndexError, ReferenceError, RuntimeError):
                    pass
                triangle_materials.append(material)
        finally:
            try:
                evaluated.to_mesh_clear()
            except (RuntimeError, AttributeError):
                pass

    tree = None
    if vertices and polygons:
        tree = BVHTree.FromPolygons(
            [tuple(v) for v in vertices], polygons, all_triangles=True
        )
    if not vertices:
        minimum = maximum = [0.0, 0.0, 0.0]

    scene = RenderVisibleScene(
        tree, triangle_objects, triangle_materials, objects,
        tuple(minimum), tuple(maximum),
        _surface_samples(vertices, polygons, surface_budget)
        if surface_budget and tree is not None else None,
    )
    _cache.update({"key": key, "scene": scene, "built_at": time.time()})
    return scene


def invalidate():
    _cache.update({"key": None, "scene": None, "built_at": 0.0})


@bpy.app.handlers.persistent
def _invalidate_on_load(*_args):
    invalidate()


def register():
    if _invalidate_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_invalidate_on_load)


def unregister():
    while _invalidate_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_invalidate_on_load)
    invalidate()
