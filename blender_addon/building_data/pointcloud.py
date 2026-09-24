# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""RGB-D point cloud: per-camera back-projection and multi-view fusion.

Pure numpy on purpose - nothing here touches ``bpy`` - so the maths can be
tested outside Blender and runs unchanged on the finalize worker thread.

Two stages:

``backproject_camera``
    One rendered view -> surface samples. Every sample is taken at an exact
    pixel centre, so its ray, depth and colour all describe the same pixel.
    Samples next to a depth discontinuity are flagged: their colour is an
    anti-aliased blend of foreground and background. Samples whose depth lies
    *between* two surfaces (a filtered depth value that belongs to neither)
    are dropped outright - those are the floaters a trainer cannot remove.

``fuse_samples``
    All views -> one seed cloud. Each sample asks for a merge cell sized to
    its own footprint (the surface area one sample covers), so a close-up
    camera keeps its detail and a distant one does not flood the scene with
    duplicates. Cells are processed finest first; a coarser sample is dropped
    where finer data already exists. Within a cell, samples are averaged with
    close, frontal views weighted highest, and only surfaces facing the same
    way are ever merged, so the two sides of a thin wall stay separate.

Nothing here limits the number of points. ``WARNING_POINTS`` is only the
size above which the add-on warns that the seed is very large.
"""

import math

import numpy as np

#: The add-on warns - but never caps - above this many seed points.
WARNING_POINTS = 2_000_000

#: Depth change between neighbouring pixels, in pixel footprints, above which
#: the pixels are treated as lying on different surfaces. tan(85 deg) ~= 11.4:
#: anything steeper than an 85 degree grazing surface counts as an edge.
EDGE_SLOPE = 12.0

#: Blender writes this (or the camera far clip) for pixels with no geometry;
#: the dataset mask uses the same threshold.
BACKGROUND_DEPTH = 1.0e9

_SQRT2 = math.sqrt(2.0)
_NEIGHBOURS = tuple(
    (dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy
)


def camera_offset(index):
    """Deterministic sub-stride offset in [0, 1)^2 for camera ``index``.

    An R2 low-discrepancy sequence: every camera samples a different phase of
    its pixel grid, so overlapping views fill each other's gaps instead of
    landing on the same aliasing pattern.
    """
    return (
        (0.5 + 0.7548776662466927 * (index + 1)) % 1.0,
        (0.5 + 0.5698402909980532 * (index + 1)) % 1.0,
    )


def grid_for_rays(rays, width, height):
    """Sample grid ``(columns, rows, count)`` with the image aspect ratio."""
    rays = max(64, int(rays))
    aspect = width / max(1, height)
    columns = max(8, min(int(width), int(round(math.sqrt(rays * aspect)))))
    rows = max(8, min(int(height), int(math.ceil(rays / columns))))
    return columns, rows, columns * rows


def _valid(depth, limit):
    return np.isfinite(depth) & (depth > 0.0) & (depth < limit)


def _unit(vectors):
    length = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(length, 1.0e-30), length[:, 0]


def backproject_camera(depth, rgb, rotation, origin, intrinsics, grid,
                       camera_index, max_depth=BACKGROUND_DEPTH,
                       normal_map=None):
    """Surface samples for one camera.

    ``depth`` is Blender's metric camera-space Z (H x W, top row first),
    ``rgb`` the matching H x W x 3 uint8 image, ``rotation``/``origin`` the
    camera-to-world rotation (3x3, Blender camera axes) and position,
    ``intrinsics`` ``(fx, fy, cx, cy)`` in pixels with pixel centres at +0.5,
    and ``grid`` ``(columns, rows)``. ``normal_map`` is the optional world
    normal pass (H x W x 3).

    Returns ``(samples, counts)``. ``samples`` holds world-space float64
    ``xyz``, uint8 ``rgb``, float32 ``normal`` (camera-facing), float32
    ``spacing`` (area-equivalent distance between this camera's samples on
    that surface), bool ``edge`` and int32 ``camera``. ``counts`` reports what was rejected.
    """
    depth = np.asarray(depth)
    height, width = depth.shape[:2]
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    columns, rows = int(grid[0]), int(grid[1])
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    limit = min(float(max_depth) * (1.0 - 1.0e-6), BACKGROUND_DEPTH)

    offset_x, offset_y = camera_offset(int(camera_index))
    xs = np.minimum(width - 1, np.floor(
        (np.arange(columns) + offset_x) * width / columns)).astype(np.int64)
    ys = np.minimum(height - 1, np.floor(
        (np.arange(rows) + offset_y) * height / rows)).astype(np.int64)
    px, py = (axis.ravel() for axis in np.meshgrid(xs, ys))
    requested = px.size

    d0 = depth[py, px].astype(np.float64)
    keep = _valid(d0, limit)
    background = int(requested - np.count_nonzero(keep))
    px, py, d0 = px[keep], py[keep], d0[keep]

    # One pixel's footprint at this depth; a surface tilted by t degrees
    # changes depth by footprint * tan(t) per pixel.
    footprint = d0 / max(1.0e-9, min(fx, fy))
    threshold = EDGE_SLOPE * footprint
    farther = np.zeros(px.size, dtype=bool)
    nearer = np.zeros(px.size, dtype=bool)
    silhouette = np.zeros(px.size, dtype=bool)
    neighbour_depth = {}
    for dx, dy in _NEIGHBOURS:
        qx = np.clip(px + dx, 0, width - 1)
        qy = np.clip(py + dy, 0, height - 1)
        dn = depth[qy, qx].astype(np.float64)
        valid = _valid(dn, limit)
        silhouette |= ~valid
        step = threshold * (_SQRT2 if dx and dy else 1.0)
        difference = np.where(valid, dn - d0, 0.0)
        farther |= difference > step
        nearer |= difference < -step
        if (dx == 0) != (dy == 0):
            # Only the four direct neighbours are needed for normals.
            neighbour_depth[(dx, dy)] = (
                dn, valid & (np.abs(difference) <= step))
    # A depth between a nearer and a farther surface belongs to neither: it
    # is a filtered edge value floating in mid-air. Never keep it.
    floater = farther & nearer
    edge = (silhouette | farther | nearer) & ~floater

    def rays(x, y):
        return np.stack((
            (x + 0.5 - cx) / fx,
            -(y + 0.5 - cy) / fy,
            -np.ones(x.shape, dtype=np.float64),
        ), axis=-1) @ rotation.T

    direction = rays(px.astype(np.float64), py.astype(np.float64))
    xyz = origin + direction * d0[:, None]
    view, _length = _unit(direction)

    # Geometric normal from the depth map itself: exact for rendered depth,
    # independent of bump or normal maps, and available without a normal pass.
    def neighbour_point(dx, dy):
        dn, usable = neighbour_depth[(dx, dy)]
        qx = np.clip(px + dx, 0, width - 1).astype(np.float64)
        qy = np.clip(py + dy, 0, height - 1).astype(np.float64)
        return origin + rays(qx, qy) * dn[:, None], usable

    def tangent(positive, negative):
        (p_point, p_ok), (n_point, n_ok) = positive, negative
        central = p_point - n_point
        forward = p_point - xyz
        backward = xyz - n_point
        result = np.where((p_ok & n_ok)[:, None], central,
                          np.where(p_ok[:, None], forward, backward))
        return result, p_ok | n_ok

    tangent_x, ok_x = tangent(neighbour_point(1, 0), neighbour_point(-1, 0))
    tangent_y, ok_y = tangent(neighbour_point(0, 1), neighbour_point(0, -1))
    geometric, area = _unit(np.cross(tangent_x, tangent_y))
    has_geometric = ok_x & ok_y & (area > 0.0) & np.isfinite(area)
    facing = np.einsum("ij,ij->i", geometric, view)
    geometric = np.where((facing > 0.0)[:, None], -geometric, geometric)
    cosine = np.where(has_geometric, np.abs(facing), 1.0)

    normal = np.where(has_geometric[:, None], geometric, -view)
    if normal_map is not None:
        shading = np.asarray(normal_map)[py, px, :3].astype(np.float64)
        shading, length = _unit(np.nan_to_num(shading, nan=0.0))
        usable = np.isfinite(length) & (length > 0.25)
        towards = np.einsum("ij,ij->i", shading, view)
        shading = np.where((towards > 0.0)[:, None], -shading, shading)
        normal = np.where(usable[:, None], shading, normal)
        # Tilt comes from the depth geometry when it can; the shading normal
        # only stands in where the depth neighbourhood could not give one.
        cosine = np.where(has_geometric | ~usable, cosine, np.abs(towards))

    # Area-equivalent sample spacing: this camera's samples are stride pixels
    # apart, stretched by 1/cos on a tilted surface along one axis only.
    stride = math.sqrt((width / columns) * (height / rows) / (fx * fy))
    spacing = d0 * stride / np.sqrt(np.clip(cosine, 0.05, 1.0))

    edge = edge[~floater]
    samples = {
        "xyz": xyz[~floater],
        "rgb": np.asarray(rgb)[py, px, :3][~floater].astype(np.uint8),
        "normal": normal[~floater].astype(np.float32),
        "spacing": spacing[~floater].astype(np.float32),
        "edge": edge,
        "camera": np.full(edge.size, int(camera_index), dtype=np.int32),
    }
    counts = {
        "requested": int(requested),
        "background": background,
        "floaters": int(np.count_nonzero(floater)),
        "edges": int(np.count_nonzero(edge)),
        "kept": int(edge.size),
    }
    return samples, counts


def normal_bins(normal):
    """One of six direction classes; opposite-facing surfaces never merge."""
    normal = np.asarray(normal, dtype=np.float32)
    if normal.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    axis = np.argmax(np.abs(normal), axis=1)
    negative = normal[np.arange(normal.shape[0]), axis] < 0.0
    return axis.astype(np.int64) * 2 + negative


def _cell_keys(cells):
    """Exact int64 identity per row of an ``N x 4`` integer cell array."""
    if cells.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    relative = cells - cells.min(axis=0)
    spans = relative.max(axis=0)
    if int(spans[:3].max()) < (1 << 20) and int(spans[3]) < 8:
        return ((relative[:, 0] << 43) | (relative[:, 1] << 23)
                | (relative[:, 2] << 3) | relative[:, 3])
    # Enormous scene-to-detail ratio: exact but slower row identity.
    _unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64)


def fuse_samples(xyz, rgb, normal, spacing, edge, camera=None,
                 merge_strength=1.0, cancelled=None, report=None):
    """Footprint-adaptive, finest-first fusion of every camera's samples.

    ``merge_strength`` scales every merge cell: 1.0 keeps about one point
    per sample footprint, 2.0 about a quarter of that. ``cancelled`` is an
    optional callable polled between levels; when it returns True the
    function returns None. ``report(fraction)`` receives progress.

    Returns a dict of fused ``xyz`` (float64), ``rgb`` (uint8), ``normal``
    (float32), ``radius`` (float32, half the merge cell), ``support``
    (int32 sample count) and ``edge_only`` (bool: built only from edge
    samples, typically thin structures), plus ``levels``.

    A cell built only from edge samples is kept only when at least two
    cameras agree on it (``camera`` gives each sample's view): a real thin
    structure is seen from several views, a filtered-depth floater is not.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    count = xyz.shape[0]
    empty = {
        "xyz": np.zeros((0, 3), np.float64), "rgb": np.zeros((0, 3), np.uint8),
        "normal": np.zeros((0, 3), np.float32),
        "radius": np.zeros(0, np.float32), "support": np.zeros(0, np.int32),
        "edge_only": np.zeros(0, bool), "levels": 0,
    }
    if count == 0:
        return empty
    rgb = np.asarray(rgb, dtype=np.uint8)
    normal = np.asarray(normal, dtype=np.float32)
    edge = np.asarray(edge, dtype=bool)
    camera = (np.zeros(count, np.int64) if camera is None
              else np.asarray(camera, dtype=np.int64))
    cell_wanted = (np.maximum(np.asarray(spacing, dtype=np.float64), 1.0e-12)
                   * max(1.0e-3, float(merge_strength)))
    finest = max(float(np.percentile(cell_wanted, 0.5)), 1.0e-12)
    level = np.rint(np.log2(np.maximum(cell_wanted, finest) / finest))
    level = np.clip(level, 0, 60).astype(np.int64)
    # Closer, more frontal samples cover less area each: trust them most.
    # Relative to the finest cell so the weights stay well inside float range.
    weight = (finest / cell_wanted) ** 2
    bins = normal_bins(normal)
    origin = xyz.min(axis=0)

    levels = np.unique(level)
    parts = {key: [] for key in
             ("xyz", "rgb", "normal", "radius", "support", "edge_only")}
    occupied_xyz = np.zeros((0, 3), np.float64)
    occupied_bin = np.zeros(0, np.int64)
    for number, current in enumerate(levels.tolist()):
        if cancelled is not None and cancelled():
            return None
        index = np.flatnonzero(level == current)
        cell = finest * (2.0 ** current)
        cells = np.empty((index.size, 4), dtype=np.int64)
        cells[:, :3] = np.floor((xyz[index] - origin) / cell)
        cells[:, 3] = bins[index]
        if occupied_xyz.shape[0]:
            # A coarser sample is redundant wherever a finer point already
            # describes the same surface orientation in this cell.
            occupied = np.empty((occupied_xyz.shape[0], 4), dtype=np.int64)
            occupied[:, :3] = np.floor((occupied_xyz - origin) / cell)
            occupied[:, 3] = occupied_bin
            keys = _cell_keys(np.concatenate((occupied, cells)))
            taken, keys = keys[:occupied.shape[0]], keys[occupied.shape[0]:]
            fresh = ~np.isin(keys, taken)
            index, keys = index[fresh], keys[fresh]
        else:
            keys = _cell_keys(cells)
        if index.size:
            groups, inverse = np.unique(keys, return_inverse=True)
            inverse = inverse.reshape(-1)
            total = groups.size
            sample_edge = edge[index]
            clean = np.bincount(inverse, weights=(~sample_edge).astype(
                np.float64), minlength=total) > 0.0
            # Edge colours are blends: use them only where nothing clean
            # exists, which is what keeps thin wires and twigs in the seed.
            used = ~sample_edge | ~clean[inverse]
            w = np.where(used, weight[index], 0.0)
            total_weight = np.maximum(
                np.bincount(inverse, weights=w, minlength=total), 1.0e-300)

            def mean(values):
                return np.stack([
                    np.bincount(inverse, weights=w * values[:, axis],
                                minlength=total) / total_weight
                    for axis in range(values.shape[1])
                ], axis=1)

            # Edge-only cells need a second view to vouch for them.
            views = np.zeros(total, dtype=np.int64)
            pairs = np.unique(inverse * (int(camera.max()) + 1)
                              + camera[index])
            np.add.at(views, pairs // (int(camera.max()) + 1), 1)
            accepted = clean | (views >= 2)
            fused_xyz = mean(xyz[index])
            fused_rgb = np.clip(np.rint(mean(rgb[index].astype(np.float64))),
                                0, 255).astype(np.uint8)
            fused_normal = mean(normal[index].astype(np.float64))
            fused_normal, length = _unit(fused_normal)
            if np.any(length <= 1.0e-12):
                first = np.zeros(total, dtype=np.int64)
                first[inverse[::-1]] = index[::-1]
                fallback = normal[first].astype(np.float64)
                fused_normal = np.where(
                    (length <= 1.0e-12)[:, None], fallback, fused_normal)
            support = np.bincount(inverse, weights=used.astype(np.float64),
                                  minlength=total).astype(np.int32)
            fused_xyz = fused_xyz[accepted]
            fused_normal = fused_normal[accepted]
            parts["xyz"].append(fused_xyz)
            parts["rgb"].append(fused_rgb[accepted])
            parts["normal"].append(fused_normal.astype(np.float32))
            parts["radius"].append(
                np.full(fused_xyz.shape[0], 0.5 * cell, np.float32))
            parts["support"].append(support[accepted])
            parts["edge_only"].append(~clean[accepted])
            occupied_xyz = np.concatenate((occupied_xyz, fused_xyz))
            occupied_bin = np.concatenate(
                (occupied_bin, normal_bins(fused_normal)))
        if report is not None:
            report((number + 1) / len(levels))

    if not parts["xyz"]:
        return empty
    fused = {key: np.concatenate(values) for key, values in parts.items()}
    # A fixed shuffle: trainers that keep an evenly strided subset of the
    # file (as Safe Mode does) then keep an evenly spread subset of the scene.
    order = np.random.default_rng(0x5EED).permutation(fused["xyz"].shape[0])
    fused = {key: value[order] for key, value in fused.items()}
    fused["levels"] = int(len(levels))
    return fused


def write_points3d(handle, xyz, rgb, header_lines=(), cancelled=None,
                   chunk=200_000):
    """COLMAP ``points3D.txt`` rows. Returns False if cancelled mid-write."""
    count = int(xyz.shape[0])
    handle.write("# 3D point list with one line of data per point:\n")
    handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                 "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    handle.write(f"# Number of points: {count}, mean track length: 0\n")
    for line in header_lines:
        handle.write(f"# {line}\n")
    for start in range(0, count, chunk):
        if cancelled is not None and cancelled():
            return False
        end = min(count, start + chunk)
        block = np.empty((end - start, 8), dtype=np.float64)
        block[:, 0] = np.arange(start + 1, end + 1)
        block[:, 1:4] = xyz[start:end]
        block[:, 4:7] = rgb[start:end]
        block[:, 7] = 0.0
        np.savetxt(handle, block, fmt="%d %.6f %.6f %.6f %d %d %d %.1f")
    return True
