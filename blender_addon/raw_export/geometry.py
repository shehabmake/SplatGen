# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Scene mesh and collision voxels, from the same geometry the render sees.

Both are in Blender world space, the frame every other SplatGen output uses
(the legacy export transform is the identity), so a triangle, a voxel, a
Position pixel and a COLMAP camera centre are directly comparable.

NumPy only; nothing here needs a package Blender does not ship.
"""

import json
import os
from pathlib import Path

import bpy

_MESHLIKE = {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD"}

#: Voxel labels.
FREE = 0
SURFACE = 1
ENCLOSED = 2


def _renderable_names(scene, view_layer):
    from .. import sceneray_splat

    names = {obj.name_full for obj in
             sceneray_splat.sr_renderable_mesh_objects(scene, view_layer)}
    layer_objects = {obj.name_full for obj in getattr(view_layer, "objects", ())}
    for obj in scene.objects:
        if obj.type in _MESHLIKE and obj.type != "MESH":
            if (obj.name_full in layer_objects and not obj.hide_render
                    and getattr(obj, "visible_camera", True)):
                names.add(obj.name_full)
    return names


def collect_scene_mesh(scene, view_layer, depsgraph, id_map=None):
    """World-space triangles of every render-visible object and instance.

    Returns a dict of NumPy arrays plus a per-object table. Geometry is
    evaluated through the dependency graph, so modifiers, Geometry Nodes and
    instancing are applied (at their viewport levels - Blender offers Python
    no render-level depsgraph).
    """
    import numpy as np

    object_ids = {entry["name"]: int(entry["id"])
                  for entry in (id_map or {}).get("objects", ())}
    material_ids = {entry["name"]: int(entry["id"])
                    for entry in (id_map or {}).get("materials", ())}
    renderable = _renderable_names(scene, view_layer)

    vertices, normals, triangles = [], [], []
    tri_object, tri_material = [], []
    table = []
    offset = 0
    tri_offset = 0
    for instance in depsgraph.object_instances:
        evaluated = instance.object
        if evaluated.type not in _MESHLIKE:
            continue
        original = getattr(evaluated, "original", evaluated)
        if instance.is_instance:
            parent = getattr(instance, "parent", None)
            parent = getattr(parent, "original", parent)
            if parent is None:
                continue
            # An empty (collection instance) is never "renderable" itself;
            # a mesh instancer must be.
            if parent.type in _MESHLIKE and parent.name_full not in renderable:
                continue
            if parent.hide_render or not getattr(parent, "visible_camera", True):
                continue
        elif original.name_full not in renderable:
            continue
        try:
            mesh = evaluated.to_mesh()
        except RuntimeError:
            continue
        if mesh is None:
            continue
        try:
            mesh.calc_loop_triangles()
            n_vert = len(mesh.vertices)
            n_tri = len(mesh.loop_triangles)
            if n_vert == 0 or n_tri == 0:
                continue
            co = np.empty(n_vert * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3).astype(np.float64)
            nrm = np.empty(n_vert * 3, dtype=np.float32)
            try:
                mesh.vertex_normals.foreach_get("vector", nrm)
            except AttributeError:
                mesh.vertices.foreach_get("normal", nrm)
            nrm = nrm.reshape(-1, 3).astype(np.float64)
            tri = np.empty(n_tri * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", tri)
            tri = tri.reshape(-1, 3)
            mat_index = np.empty(n_tri, dtype=np.int32)
            mesh.loop_triangles.foreach_get("material_index", mat_index)

            matrix = np.array(instance.matrix_world, dtype=np.float64)
            co = co @ matrix[:3, :3].T + matrix[:3, 3]
            normal_matrix = np.linalg.inv(matrix[:3, :3]).T
            nrm = nrm @ normal_matrix.T
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
            if np.linalg.det(matrix[:3, :3]) < 0:
                # A mirrored instance flips winding; keep faces outward.
                tri = tri[:, ::-1]

            slot_ids = []
            slot_names = []
            for material in mesh.materials:
                name = getattr(getattr(material, "original", material), "name_full", "")
                slot_names.append(name)
                slot_ids.append(material_ids.get(name, 0))
            lookup = np.array(slot_ids or [0], dtype=np.int32)
            mat_index = np.clip(mat_index, 0, len(lookup) - 1)

            vertices.append(co.astype(np.float32))
            normals.append(nrm.astype(np.float32))
            triangles.append(tri + offset)
            obj_id = object_ids.get(original.name_full, 0)
            tri_object.append(np.full(n_tri, obj_id, dtype=np.int32))
            tri_material.append(lookup[mat_index])
            table.append({
                "name": original.name_full,
                "object_id": obj_id,
                "instance": bool(instance.is_instance),
                "type": evaluated.type,
                "vertex_offset": int(offset),
                "vertex_count": int(n_vert),
                "triangle_offset": int(tri_offset),
                "triangle_count": int(n_tri),
                "materials": slot_names,
                "matrix_world": [[float(v) for v in row] for row in matrix],
            })
            offset += n_vert
            tri_offset += n_tri
        finally:
            evaluated.to_mesh_clear()

    if not triangles:
        return None
    return {
        "vertices": np.concatenate(vertices),
        "normals": np.concatenate(normals),
        "triangles": np.concatenate(triangles).astype(np.int32),
        "triangle_object": np.concatenate(tri_object),
        "triangle_material": np.concatenate(tri_material),
        "objects": table,
    }


def write_ply(mesh, path):
    """Binary little-endian PLY; readable by Open3D, trimesh and MeshLab."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = mesh["vertices"]
    faces = mesh["triangles"]
    vertex_data = np.empty(len(vertices), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ])
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = vertices.T
    vertex_data["nx"], vertex_data["ny"], vertex_data["nz"] = mesh["normals"].T
    face_data = np.empty(len(faces), dtype=[
        ("n", "u1"), ("v", "<i4", (3,)), ("object_id", "<i4"), ("material_id", "<i4"),
    ])
    face_data["n"] = 3
    face_data["v"] = faces
    face_data["object_id"] = mesh["triangle_object"]
    face_data["material_id"] = mesh["triangle_material"]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment SplatGen raw scene mesh, Blender world space (Z up, metres)\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "property int object_id\nproperty int material_id\n"
        "end_header\n"
    )
    pending = path.with_name(f".{path.name}.pending")
    with open(pending, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertex_data.tobytes())
        handle.write(face_data.tobytes())
    os.replace(pending, path)


def mesh_bounds(mesh):
    vertices = mesh["vertices"]
    return vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()


# --------------------------------------------------------------------------
# Voxels
# --------------------------------------------------------------------------

def _surface_points(vertices, triangles, spacing, chunk=200000):
    """Points covering every triangle at no more than ``spacing`` apart."""
    import numpy as np

    v = vertices.astype(np.float64)
    for start in range(0, len(triangles), chunk):
        tri = v[triangles[start:start + chunk]]
        edges = np.stack([
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ], axis=1).max(axis=1)
        steps = np.maximum(1, np.ceil(edges / spacing)).astype(np.int64)
        for k in np.unique(steps):
            group = tri[steps == k]
            i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
            keep = (i + j) <= k
            a = (i[keep] / k)[:, None]
            b = (j[keep] / k)[:, None]
            weights = np.concatenate([1.0 - a - b, a, b], axis=1)
            # Bound memory for very large triangles.
            per = max(1, 4000000 // max(1, len(weights)))
            for s in range(0, len(group), per):
                block = group[s:s + per]
                yield np.einsum("pk,tkc->tpc", weights, block).reshape(-1, 3)


def voxelize(mesh, camera_positions, resolution):
    """Label a grid FREE / SURFACE / ENCLOSED.

    Free space is everything reachable without crossing a surface voxel from
    the grid border or from any camera - cameras are necessarily in free
    space, which is what keeps the inside of a room free while the inside of
    a thick wall is enclosed.
    """
    import numpy as np

    vertices = mesh["vertices"].astype(np.float64)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    if len(camera_positions):
        cams = np.asarray(camera_positions, dtype=np.float64)
        lo = np.minimum(lo, cams.min(axis=0))
        hi = np.maximum(hi, cams.max(axis=0))
    extent = float(max((hi - lo).max(), 1e-6))
    size = extent / float(resolution)
    lo = lo - 2 * size
    dims = np.ceil((hi + 2 * size - lo) / size).astype(np.int64)
    dims = np.maximum(dims, 1)

    surface = np.zeros(tuple(dims), dtype=bool)
    for points in _surface_points(vertices, mesh["triangles"], size * 0.5):
        index = np.floor((points - lo) / size).astype(np.int64)
        np.clip(index, 0, dims - 1, out=index)
        surface[index[:, 0], index[:, 1], index[:, 2]] = True

    free = np.zeros_like(surface)
    free[0, :, :] = free[-1, :, :] = True
    free[:, 0, :] = free[:, -1, :] = True
    free[:, :, 0] = free[:, :, -1] = True
    if len(camera_positions):
        index = np.floor((np.asarray(camera_positions) - lo) / size).astype(np.int64)
        np.clip(index, 0, dims - 1, out=index)
        free[index[:, 0], index[:, 1], index[:, 2]] = True
    free &= ~surface
    open_space = ~surface
    count = int(free.sum())
    while True:
        grown = free.copy()
        grown[1:] |= free[:-1]
        grown[:-1] |= free[1:]
        grown[:, 1:] |= free[:, :-1]
        grown[:, :-1] |= free[:, 1:]
        grown[:, :, 1:] |= free[:, :, :-1]
        grown[:, :, :-1] |= free[:, :, 1:]
        grown &= open_space
        new_count = int(grown.sum())
        free = grown
        if new_count == count:
            break
        count = new_count

    labels = np.full(tuple(dims), ENCLOSED, dtype=np.uint8)
    labels[free] = FREE
    labels[surface] = SURFACE
    return {"labels": labels, "origin": lo, "voxel_size": size, "dims": dims}


def write_voxels(grid, npz_path, info_path, extra=None):
    import numpy as np

    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    pending = npz_path.with_name(f".{npz_path.stem}.pending.npz")
    np.savez_compressed(
        pending,
        labels=grid["labels"],
        origin=np.asarray(grid["origin"], dtype=np.float64),
        voxel_size=np.float64(grid["voxel_size"]),
        dims=np.asarray(grid["dims"], dtype=np.int64),
    )
    os.replace(pending, npz_path)
    labels = grid["labels"]
    info = {
        "format": "SplatGen collision voxels",
        "file": npz_path.name,
        "arrays": {
            "labels": "uint8 [X, Y, Z]; 0 = free, 1 = surface, 2 = enclosed",
            "origin": "float64 [3]; world position of voxel (0,0,0)'s min corner",
            "voxel_size": "float64; edge length in scene units",
            "dims": "int64 [3]",
        },
        "index_to_world": "center = origin + (index + 0.5) * voxel_size",
        "coordinate_frame": "Blender world space (Z up), same as COLMAP files",
        "dims": [int(v) for v in grid["dims"]],
        "origin": [float(v) for v in grid["origin"]],
        "voxel_size": float(grid["voxel_size"]),
        "counts": {
            "free": int((labels == FREE).sum()),
            "surface": int((labels == SURFACE).sum()),
            "enclosed": int((labels == ENCLOSED).sum()),
        },
        "free_space_rule": "reachable from the grid border or any dataset "
                           "camera without crossing a surface voxel",
    }
    if extra:
        info.update(extra)
    Path(info_path).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def probe_positions(grid, count, camera_positions=()):
    """Automatic probe centres in free space with clearance.

    The first is the free point nearest the centre of the geometry's bounds;
    the rest are spread by farthest-point sampling across free space inside
    the bounds of the geometry and cameras. Light Probe Sphere objects in
    the scene replace this entirely, for exact control.
    """
    import numpy as np

    if count <= 0:
        return []
    labels = grid["labels"]
    free = labels == FREE
    # Prefer voxels with a little clearance from surfaces.
    for _ in range(2):
        eroded = free.copy()
        eroded[1:] &= free[:-1]
        eroded[:-1] &= free[1:]
        eroded[:, 1:] &= free[:, :-1]
        eroded[:, :-1] &= free[:, 1:]
        eroded[:, :, 1:] &= free[:, :, :-1]
        eroded[:, :, :-1] &= free[:, :, 1:]
        if not eroded.any():
            break
        free = eroded
    # Keep away from the padded border.
    free[:2] = free[-2:] = False
    free[:, :2] = free[:, -2:] = False
    free[:, :, :2] = free[:, :, -2:] = False
    candidates = np.argwhere(free)
    if not len(candidates):
        return []
    origin = np.asarray(grid["origin"])
    size = float(grid["voxel_size"])
    centers = origin + (candidates + 0.5) * size
    occupied = np.argwhere(labels == SURFACE)
    if len(occupied):
        lo = origin + occupied.min(axis=0) * size
        hi = origin + (occupied.max(axis=0) + 1) * size
    else:
        lo, hi = centers.min(axis=0), centers.max(axis=0)
    # Aim at the middle of the geometry; stay inside the region spanned by
    # the geometry and the cameras, where the dataset actually looks.
    target = (lo + hi) * 0.5
    if len(camera_positions):
        cams = np.asarray(camera_positions, dtype=np.float64)
        lo = np.minimum(lo, cams.min(axis=0))
        hi = np.maximum(hi, cams.max(axis=0))
    inside = np.all((centers >= lo) & (centers <= hi), axis=1)
    if inside.any():
        centers = centers[inside]
    chosen = [int(np.argmin(np.linalg.norm(centers - target, axis=1)))]
    distance = np.linalg.norm(centers - centers[chosen[0]], axis=1)
    while len(chosen) < min(count, len(centers)):
        nxt = int(np.argmax(distance))
        if distance[nxt] <= 0:
            break
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(centers - centers[nxt], axis=1))
    return [centers[i].tolist() for i in chosen]
