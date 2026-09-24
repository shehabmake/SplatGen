# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Camera coverage analysis: how well the queued cameras capture the scene.

Pure numpy, sharing the Smart camera rig's scene analysis. Every queued
camera is evaluated exactly as it will render - its own lens, resolution,
clipping and pose, with occlusion from the voxelised scene - and every
surface point gets three measurements:

    views     how many cameras see it at a usable angle
    spread    how different those viewing directions are: two cameras side
              by side see a surface "twice" but add almost nothing
    detail    metres per pixel from the closest good view

and one verdict, from the same principles the Smart rig plans with:

    UNSEEN    no camera sees it
    WEAK      one view, or several from nearly the same direction
    FAIR      fewer views than the target, or directions too close together
    GOOD      at least the target views, spread across at least 25 degrees

Which surfaces are judged - the part that decides whether a check is useful:
every surface bordering the space the cameras can reach, found the way the
Smart rig finds rooms. Door-sized openings are sealed to tell rooms from the
outside. Cameras in rooms make *every* room reachable from them count, so a
room or tunnel that no camera went into shows up red instead of being left
out; the outside counts when a camera stands outside. Narrow passages always
count. Unseen surfaces are never dropped for being unseen - only outside
surfaces far beyond the cameras' own viewing distance are out of reach.

Weak and unseen points are then grouped into numbered problem areas, so the
result reads as "these four places need cameras" rather than as a cloud, and
cameras whose removal would lower no surface's grade are listed as redundant.
"""

import math
import time

import numpy as np

from . import planner
from .planner import _Context, _cell_pick, _march, _thin, _dilate, _dilate_cube

UNSEEN, WEAK, FAIR, GOOD = 0, 1, 2, 3
CLASS_NAMES = {UNSEEN: "Unseen", WEAK: "Weak", FAIR: "Fair", GOOD: "Good"}

BUDGETS = {
    'DRAFT': dict(voxels=1_500_000, samples=700_000, targets=25_000, image=72),
    'STANDARD': dict(voxels=4_000_000, samples=1_600_000, targets=60_000, image=112),
    'HIGH': dict(voxels=9_000_000, samples=3_000_000, targets=120_000, image=160),
}
GOOD_SPREAD = 25.0       # degrees between viewing directions for a "good" surface
WEAK_SPREAD = 10.0
GRAZING = math.cos(math.radians(80.0))

STAGE_SCENE = "Scene"
STAGE_CAMERAS = "Cameras"
STAGE_SUMMARY = "Summary"


class Camera:
    """One queued camera, as it will render."""

    def __init__(self, name, matrix, fx, fy, cx, cy, width, height, clip_start, clip_end):
        self.name = name
        self.matrix = np.asarray(matrix, dtype=np.float64)      # camera to world, no scale
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.width, self.height = int(width), int(height)
        self.clip_start, self.clip_end = float(clip_start), float(clip_end)

    @property
    def position(self):
        return self.matrix[:3, 3]


class Result:
    def __init__(self):
        self.points = np.zeros((0, 3), np.float32)
        self.normals = np.zeros((0, 3), np.float32)
        self.weights = np.zeros(0)
        self.cell = 0.1                             # spacing between points, for drawing
        self.views = np.zeros(0, np.int32)
        self.directions = np.zeros((0, 3))          # sum of surface-to-camera directions
        self.spread = np.zeros(0, np.float32)
        self.detail = np.zeros(0, np.float32)      # metres per pixel, inf when unseen
        self.grade = np.zeros(0, np.int8)
        self.fractions = {}
        self.areas = []                             # problem areas, largest first
        self.redundant = []                         # camera names adding nothing needed
        self.owners = np.zeros(0, np.int32)          # object index of each point
        self.owner_names = []
        self.camera_names = []
        self.camera_positions = np.zeros((0, 3))
        self.seen = []                              # per camera: point indices it sees
        self.rays = []                              # per camera: surface-to-camera directions
        self.duplicates = []                        # queued cameras repeating another's pose
        self.stats = {}
        self.warnings = []
        self.voxel = 1.0
        self.field = None
        self.grid = None
        self.region = None
        self.view_distance = 1.0
        self.target_views = 3
        self.scope = ""                             # what was judged, in words
        self.extent = 1.0                           # size of the analysed box
        self.cells = None                           # per-point tile size, when areas differ
        self.raw_weight = 1.0


# ---------------------------------------------------------------------------
# What to judge
# ---------------------------------------------------------------------------

def _analysis_box(geometry, positions, backdrop, ground, budget, seal):
    """The structures and the cameras; in a huge scene, the cameras' surroundings.

    A ground plane stretching far past everything else is kept to the
    structures' footprint - its distant corners are nobody's subject.
    """
    geo_lo, geo_hi = geometry.bounds(~backdrop)
    focus_lo, focus_hi = geometry.bounds(~backdrop & ~ground)
    if focus_lo is None:
        focus_lo, focus_hi = geo_lo, geo_hi
    cam_lo, cam_hi = positions.min(axis=0), positions.max(axis=0)
    size = float(np.max(focus_hi - focus_lo)) or 1.0
    # Wide enough for the outside to survive sealing beside the walls, so
    # it can claim its side of every door.
    pad = max(0.08 * size, 1.2 * seal)
    lo = focus_lo - np.array([pad, pad, 0.02 * size])
    hi = focus_hi + np.array([pad, pad, pad])
    # Cameras nearby widen the box to the space they stand in; cameras far
    # off - an aerial view, say - must not coarsen the grid for everyone.
    # Rays from outside the box enter it where they cross it.
    lo = np.minimum(lo, np.maximum(cam_lo, focus_lo - 0.3 * size))
    hi = np.maximum(hi, np.minimum(cam_hi, focus_hi + 0.3 * size))
    lo = np.maximum(lo, geo_lo - 0.02 * size)

    def voxel(lo_, hi_):
        return (float(np.prod(np.maximum(hi_ - lo_, 1e-6))) / budget['voxels']) ** (1.0 / 3.0)

    # Doors must stay resolvable, or rooms cannot be told apart.
    limit = seal / 3.0 if seal > 0 else np.inf
    if voxel(lo, hi) <= limit:
        return lo, hi, False
    span = float(np.linalg.norm(np.minimum(cam_hi, hi) - np.maximum(cam_lo, lo)))
    reach = max(0.45 * span, 0.2 * size, 0.5)
    while True:
        lo_c = np.maximum(cam_lo - reach, lo)
        hi_c = np.minimum(cam_hi + reach, hi)
        if voxel(lo_c, hi_c) <= limit or reach <= 0.3 * span + 0.5:
            return lo_c, hi_c, True
        reach *= 0.8


def _judged_space(ctx, positions, seal, report):
    """The free space whose surfaces are judged, and the outside part of it.

    Returns (region, outside, scope, beyond) - ``beyond`` is the outside when
    it is not judged - or Nones when every camera sits inside solid geometry.
    """
    base = ctx.level(1)
    seeds = np.array([ctx.seed(base, p)[0] for p in positions], dtype=np.int64)
    if not np.any(seeds >= 0):
        return None, None, "", None
    reach = base.to_grid(base.flood(seeds, ctx.tick))
    level = ctx.steps_for(seal)
    if level <= 1:
        return reach, None, "everything the cameras can reach", None
    report(STAGE_SCENE, 0.7, "Telling rooms from the outside")
    space = ctx.level(level)
    outside = (space.to_grid(space.flood(space.open, ctx.tick)) if len(space.open)
               else np.zeros(ctx.grid.shape, dtype=bool))
    rooms = space.free & ~outside & reach
    # Sealing keeps only the middle of each space. Grow the outside and the
    # rooms back together, a cube step at a time through free space, so every
    # camera - one beside a wall, one just inside a door - belongs to the
    # space that reaches it first: its own, not the one behind the door.
    label = np.zeros(ctx.grid.shape, dtype=np.int8)
    label[outside] = 1
    label[rooms] = 2
    for _step in range(level + 1):
        open_ = (label == 0) & ctx.free1
        room_side = _dilate_cube(label == 2, 1) & open_
        out_side = _dilate_cube(label == 1, 1) & open_ & ~room_side
        if not room_side.any() and not out_side.any():
            break
        label[room_side] = 2
        label[out_side] = 1
    near_outside = label == 1
    out_cams = in_cams = 0
    narrow = []
    for seed in seeds:
        if seed < 0:
            continue
        flat = int(base.index[seed])
        side = label.ravel()[flat]
        if side == 1:
            out_cams += 1
        else:
            in_cams += 1
            if side == 0:
                narrow.append(flat)     # a passage too narrow to survive sealing
    core = np.zeros(ctx.grid.shape, dtype=bool)
    parts = []
    # A room rig often has a camera or two standing in a doorway; the whole
    # outside is judged only when a real share of the cameras stand there.
    judge_outside = bool(out_cams) and (out_cams >= 0.05 * (out_cams + in_cams) or not in_cams)
    if judge_outside:
        core |= outside
        parts.append("the outside")
    if in_cams:
        # Every room the cameras can walk to counts, visited or not.
        core |= space.free & ~outside & reach
        parts.append("every room the cameras can reach")
    region = ctx.complete(core, level, extra=narrow) & reach
    if judge_outside:
        return region, near_outside & region, " and ".join(parts), None
    return region, None, " and ".join(parts), near_outside


def _cell_keys(points, cell):
    """One integer per cell of the world lattice the points fall in."""
    keys = np.floor(points / cell).astype(np.int64)
    return keys @ np.array([73856093, 19349663, 83492791], dtype=np.int64)


def _near_structures(points, grounded, distance):
    """Which points lie within ``distance`` (horizontally) of a non-ground point.

    A coarse 2D map of where things stand, grown by ``distance`` - ground
    is close enough to flat for that, and it costs nothing next to a 3D
    distance search.
    """
    cell = max(distance / 4.0, 1e-6)
    keys = np.floor(points[:, :2] / cell).astype(np.int64)
    lo = keys.min(axis=0) - 5
    keys -= lo
    shape = tuple(keys.max(axis=0) + 6)
    if shape[0] * shape[1] > 16_000_000:
        return np.ones(len(points), dtype=bool)
    taken = np.zeros(shape, dtype=bool)
    stand = keys[~grounded]
    taken[stand[:, 0], stand[:, 1]] = True
    for _ in range(4):
        grown = taken.copy()
        grown[1:] |= taken[:-1]
        grown[:-1] |= taken[1:]
        grown[:, 1:] |= taken[:, :-1]
        grown[:, :-1] |= taken[:, 1:]
        taken = grown
    return taken[keys[:, 0], keys[:, 1]]


def _march_from(field, grid, origin, dirs, limits):
    """``_march`` for a camera that may stand outside the grid.

    Each ray starts where it enters the grid; one that never does sees
    nothing in it to block the view.
    """
    lo = grid.origin + 1e-6
    hi = grid.origin + np.asarray(grid.shape) * grid.voxel - 1e-6
    safe = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    t0 = (lo - origin) / safe
    t1 = (hi - origin) / safe
    enter = np.maximum(np.max(np.minimum(t0, t1), axis=1), 0.0)
    leave = np.min(np.maximum(t0, t1), axis=1)
    depth = np.full(len(dirs), np.inf)
    idx = np.flatnonzero((leave >= enter) & (enter < limits))
    if len(idx):
        start = origin + dirs[idx] * (enter[idx, None] + 1e-5)
        depth[idx] = _march(field, grid, start, dirs[idx], limits[idx] - enter[idx]) + enter[idx]
    return depth


def _faces_space(region, grid, points, normals, voxel):
    """Whether each surface's front looks into the judged space."""
    out = np.zeros(len(points), dtype=bool)
    for k in (1.5, 2.5, 3.5):
        cell, inside = grid.ijk(points + normals * (k * voxel))
        out[inside] |= region.ravel()[grid.flat(cell[inside])]
    return out


def _covered(ctx, points, normals, owners, voxel, chunk=2048):
    """Surfaces lying face to face with another surface, which no camera can see.

    A surface is covered when, right in front of it, another surface lies
    parallel to it: within ``0.6`` voxel along the normal and ``0.6`` voxel
    sideways - a picture over a wall, a rug over a floor - or when a
    surface of another object touches it face to face, as the back of a
    picture touches the wall or a sofa's underside the floor. Surfaces
    meeting at an angle, as a wall meets the floor, never count.
    """
    reach = 0.6 * voxel
    samples, sample_normals, sample_owners = ctx.points, ctx.normals, ctx.owner_ids
    order = np.argsort(ctx.sample_flat, kind="stable")
    keys = ctx.sample_flat[order]
    shape = np.asarray(ctx.grid.shape)
    offsets = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    cell, inside = ctx.grid.ijk(points)
    out = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk):
        c, ok = cell[start:start + chunk], inside[start:start + chunk]
        m = len(c)
        near = c[:, None, :] + offsets[None]
        valid = np.all((near >= 0) & (near < shape), axis=2) & ok[:, None]
        flat = np.ravel_multi_index(np.clip(near, 0, shape - 1).reshape(-1, 3).T, shape)
        lo = np.searchsorted(keys, flat, "left")
        count = np.where(valid.ravel(), np.searchsorted(keys, flat, "right") - lo, 0)
        total = int(count.sum())
        if not total:
            continue
        target = np.repeat(np.repeat(np.arange(m), len(offsets)), count)
        first = np.repeat(lo, count) + (np.arange(total) - np.repeat(np.cumsum(count) - count, count))
        other = order[first]
        p = points[start:start + chunk][target]
        n = normals[start:start + chunk][target]
        d = samples[other] - p
        along = np.einsum("ij,ij->i", d, n)
        side = np.einsum("ij,ij->i", d, d) - along * along
        facing = np.einsum("ij,ij->i", sample_normals[other], n)
        stacked = (along > 5e-4) & (along <= reach) & (np.abs(facing) > 0.7)
        touching = ((np.abs(along) <= 2e-3) & (facing < -0.7)
                    & (sample_owners[other] != owners[start:start + chunk][target]))
        hit = (side <= reach * reach) & (stacked | touching)
        out[start + np.unique(target[hit])] = True
    return out


def _open_side(region, grid, points, normals, voxel, sheet):
    """Turn the normals of open sheets toward the judged space.

    A solid's faces point outwards already. A sheet - a plane used as a
    wall, a card - may face either way; probing the judged space on both
    sides tells which side is photographed. Solids are left alone: a probe
    through a thin wall would reach the room behind it.
    """
    facing = np.zeros(len(points), dtype=bool)
    behind = np.zeros(len(points), dtype=bool)
    for k in (2.5, 4.0):
        for sign, out in ((1.0, facing), (-1.0, behind)):
            cell, inside = grid.ijk(points + sign * normals * (k * voxel))
            out[inside] |= region.ravel()[grid.flat(cell[inside])]
    normals[sheet & ~facing & behind] *= -1.0


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

#: Voxel size above which a check is split into areas with grids of their own.
FINE_VOXEL = 0.12
MAX_PARTS = 8


def analyse(geometry, cameras, *, quality='STANDARD', target_views=3, seal_size=1.5,
            duplicates=(), seed=0, progress=None, cancelled=None):
    """Grade every relevant surface for the given cameras.

    One grid covers the scene when it can be fine. In a large scene - a city
    block around a house - one grid would be coarse, so the check is split
    into areas around groups of cameras, each with a grid of its own, and
    the areas' results are joined.
    """
    report = planner._Reporter(progress, cancelled)
    rng = np.random.default_rng(seed)
    budget = BUDGETS.get(quality, BUDGETS['STANDARD'])
    started = time.perf_counter()
    if geometry.is_empty:
        raise planner.PlanError("No render-visible geometry was found.")
    if not cameras:
        raise planner.PlanError("Queue cameras first: coverage is checked for the camera queue.")

    report(STAGE_SCENE, 0.0, "Reading the scene")
    ground, backdrop = geometry.owner_roles()
    positions = np.array([camera.position for camera in cameras])
    lo, hi, cropped = _analysis_box(geometry, positions, backdrop, ground, budget, seal_size)
    parts = _split(lo, hi, positions, budget, seal_size)
    if parts is None:
        result = _analyse_box(geometry, cameras, lo, hi, cropped, None, budget, seal_size,
                              rng, report, ground, backdrop)
    else:
        found = []
        # The point budget is shared, so a split check costs about what one does.
        share = dict(budget, targets=max(15_000, int(1.5 * budget['targets'] / len(parts))))
        for number, (core_lo, core_hi, box_lo, box_hi) in enumerate(parts):
            near = np.all((positions >= box_lo - 0.5 * (core_hi - core_lo))
                          & (positions <= box_hi + 0.5 * (core_hi - core_lo)), axis=1)
            group = [camera for camera, keep in zip(cameras, near) if keep]
            if not group:
                continue
            report(STAGE_SCENE, number / len(parts), f"Area {number + 1} of {len(parts)}")
            try:
                found.append((_analyse_box(geometry, group, box_lo, box_hi, False,
                                           (core_lo, core_hi), share, seal_size, rng, report,
                                           ground, backdrop), group))
            except planner.PlanError:
                continue
        if not found:
            raise planner.PlanError("No surfaces around the cameras to check.")
        result = _join(found, cameras, lo, hi)
    result.duplicates = list(duplicates)
    report(STAGE_SUMMARY, 0.5, "Checking each camera's contribution")
    grade(result, target_views)
    result.stats["seconds"] = time.perf_counter() - started
    return result


def _split(lo, hi, positions, budget, seal):
    """Areas for a large scene: (core lo, core hi, box lo, box hi) each, or None.

    Cameras are grouped on a ground-plane grid whose cells, with padding for
    the doorways, fit the voxel budget at ``FINE_VOXEL``. Each cell with
    cameras becomes an area; its points are judged there and nowhere else.
    """
    height = float(hi[2] - lo[2])
    if (float(np.prod(hi - lo)) / budget['voxels']) ** (1.0 / 3.0) <= FINE_VOXEL or len(positions) < 2:
        return None
    pad = max(1.2 * seal, 1.0)
    side = max(4.0, (budget['voxels'] * FINE_VOXEL ** 3 / max(height, 1.0)) ** 0.5 - 2.0 * pad)
    # Cameras standing outside the box - far off, aerial - join the nearest area.
    inside = np.clip(positions[:, :2], lo[:2], hi[:2] - 1e-6)
    for _attempt in range(12):
        keys = np.floor((inside - lo[:2]) / side).astype(np.int64)
        cells = np.unique(keys, axis=0)
        if len(cells) <= MAX_PARTS:
            break
        side *= 1.4
    if len(cells) < 2:
        return None
    whole = (float(np.prod(hi - lo)) / budget['voxels']) ** (1.0 / 3.0)
    area = ((side + 2 * pad) ** 2 * height / budget['voxels']) ** (1.0 / 3.0)
    if area > 0.75 * whole:
        return None         # no meaningfully finer grid to gain
    parts = []
    for cell in cells:
        core_lo = np.array([lo[0] + cell[0] * side, lo[1] + cell[1] * side, lo[2]])
        core_hi = np.array([core_lo[0] + side, core_lo[1] + side, hi[2]])
        box_lo = np.maximum(core_lo - np.array([pad, pad, 0.0]), lo)
        box_hi = np.minimum(core_hi + np.array([pad, pad, 0.0]), hi)
        parts.append((core_lo, core_hi, box_lo, box_hi))
    return parts


def _join(found, cameras, lo, hi):
    """One result from several areas' results."""
    result = Result()
    index = {camera.name: i for i, camera in enumerate(cameras)}
    offsets, total = [], 0
    for part, _group in found:
        offsets.append(total)
        total += len(part.points)
    cat = lambda name: np.concatenate([getattr(part, name) for part, _group in found])
    result.points, result.normals, result.owners = cat("points"), cat("normals"), cat("owners")
    raw = np.concatenate([part.weights * part.raw_weight for part, _group in found])
    result.weights = raw / max(raw.sum(), 1e-12)
    result.cells = np.concatenate([np.full(len(part.points), part.cell, np.float32)
                                   for part, _group in found])
    result.cell = float(np.min(result.cells)) if len(result.cells) else 0.1
    result.views, result.directions = cat("views"), cat("directions")
    result.spread, result.detail = cat("spread"), cat("detail")
    result.owner_names = found[0][0].owner_names
    result.camera_names = [camera.name for camera in cameras]
    result.camera_positions = np.array([camera.position for camera in cameras])
    seen = [[] for _ in cameras]
    rays = [[] for _ in cameras]
    for (part, group), offset in zip(found, offsets):
        for local, camera in enumerate(group):
            seen[index[camera.name]].append(part.seen[local].astype(np.int64) + offset)
            rays[index[camera.name]].append(part.rays[local])
    result.seen = [np.concatenate(s).astype(np.int32) if s else np.zeros(0, np.int32) for s in seen]
    result.rays = [np.concatenate(r) if r else np.zeros((0, 3), np.float32) for r in rays]
    biggest = max(found, key=lambda item: len(item[0].points))[0]
    result.voxel = min(part.voxel for part, _group in found)
    result.field, result.grid, result.region = biggest.field, biggest.grid, biggest.region
    result.view_distance = float(np.median([part.view_distance for part, _group in found]))
    scopes = []
    for part, _group in found:
        if part.scope not in scopes:
            scopes.append(part.scope)
    result.scope = "; ".join(scopes) + f" (in {len(found)} areas)"
    result.extent = float(np.linalg.norm(hi - lo))
    for part, _group in found:
        result.warnings.extend(w for w in part.warnings if w not in result.warnings)
    views, detail = result.views, result.detail
    seen_detail = detail[np.isfinite(detail)]
    result.stats = dict(
        cameras=len(cameras), points=len(result.points), seconds=0.0,
        out_of_reach=sum(part.stats.get("out_of_reach", 0) for part, _group in found),
        median_views=float(np.median(views)) if len(views) else 0.0,
        median_spread=float(np.median(result.spread[views >= 2])) if np.any(views >= 2) else 0.0,
        median_detail=float(np.median(seen_detail)) if len(seen_detail) else 0.0,
        areas=len(found))
    return result


def _analyse_box(geometry, cameras, lo, hi, cropped, core, budget, seal_size, rng, report,
                 ground, backdrop):
    """Measure every relevant surface in one box; ``core`` limits what is kept."""
    result = Result()
    positions = np.array([camera.position for camera in cameras])
    report(STAGE_SCENE, 0.3, "Voxelising the scene")
    ctx = _Context(geometry, lo, hi, voxels=budget['voxels'], samples=budget['samples'],
                   rng=rng, exclude=backdrop,
                   open_faces=lambda lo_, hi_, voxel: [[True, True]] * 3,
                   seal_size=seal_size, tick=lambda: report(STAGE_SCENE, None))
    voxel = ctx.grid.voxel
    report(STAGE_SCENE, 0.6, "Finding the space the cameras can reach")
    region, outside, scope, beyond = _judged_space(ctx, positions, seal_size, report)
    if region is None or not region.any():
        # Every camera is inside geometry: judge everything in the box.
        region, outside, scope, beyond = ~ctx.barriers[1], None, "everything around the cameras", None
        result.warnings.append("The cameras are inside solid geometry; every surface "
                               "around them is judged.")
    floor_z = ctx.grid.origin[2] + ctx.z_floor * voxel
    if positions[:, 2].min() > floor_z:
        region[:, :, :ctx.z_floor] = False
    if cropped:
        scope += ", near the cameras"

    near = _dilate(region, 2).ravel()[ctx.sample_flat]
    candidates = np.flatnonzero(near)
    if not len(candidates):
        raise planner.PlanError("No surfaces around the cameras to check.")
    picked, weights, cell = _thin(ctx.points[candidates], budget['targets'], rng, return_cell=True)
    if cell is None or cell < voxel:
        # Finer than the scene is sampled, the tiles would have holes - and
        # the visibility test resolves no finer than a voxel anyway.
        cell = voxel
        picked, weights = _cell_pick(ctx.points[candidates], cell, rng)
        weights = weights.astype(np.float64)
    chosen = candidates[picked]
    # Surfaces off the judged space - in a passage too narrow for this grid
    # to hold open voxels - still count wherever a camera photographs them.
    # They are tested like the rest and kept only if seen.
    others = np.flatnonzero(~near)
    optional = np.zeros(len(chosen), dtype=bool)
    if len(others):
        extra, extra_weights = _cell_pick(ctx.points[others], cell, rng)
        taken = _cell_keys(ctx.points[chosen], cell)
        extra_keys = _cell_keys(ctx.points[others[extra]], cell)
        fresh = ~np.isin(extra_keys, taken)
        chosen = np.concatenate([chosen, others[extra[fresh]]])
        weights = np.concatenate([weights, extra_weights[fresh].astype(np.float64)])
        optional = np.concatenate([optional, np.ones(int(fresh.sum()), dtype=bool)])
    points = ctx.points[chosen].astype(np.float64)
    normals = ctx.normals[chosen].astype(np.float64)
    owners = ctx.owner_ids[chosen].astype(np.int32)
    # Solids are seen from the front only - the grid alone cannot tell the
    # two faces of a thin wall apart, their facing direction can. Open
    # sheets are seen from either side.
    closed = geometry.owner_closed()
    two_sided = ~closed[np.minimum(owners, len(closed) - 1)]
    _open_side(region, ctx.grid, points, normals, voxel, two_sided)
    # Only what a camera could ever photograph is judged. A face pressed
    # against another - the back of a picture, the wall behind it, the
    # floor under a sofa - is nobody's to capture. A solid's face that turns
    # away from the judged space - the outside of a thin wall or ceiling,
    # seen from a room - counts only if a camera does photograph it.
    covered = _covered(ctx, points, normals, owners, voxel)
    if covered.any():
        # A cell whose pick was covered - the wall behind a picture - takes a
        # visible sample of the same cell instead, so the picture's face is
        # not left full of holes.
        lost = _cell_keys(points[covered], cell)
        lost_weight = dict(zip(lost.tolist(), weights[covered].tolist()))
        pool = candidates[np.isin(_cell_keys(ctx.points[candidates], cell), lost)]
        pool = pool[~np.isin(pool, chosen)]
        keep = ~covered
        chosen, weights, optional, two_sided = chosen[keep], weights[keep], optional[keep], two_sided[keep]
        points, normals, owners = points[keep], normals[keep], owners[keep]
        if len(pool):
            p_points = ctx.points[pool].astype(np.float64)
            p_normals = ctx.normals[pool].astype(np.float64)
            p_owners = ctx.owner_ids[pool].astype(np.int32)
            p_two = ~closed[np.minimum(p_owners, len(closed) - 1)]
            _open_side(region, ctx.grid, p_points, p_normals, voxel, p_two)
            visible = ~_covered(ctx, p_points, p_normals, p_owners, voxel)
            keys = _cell_keys(p_points, cell)[visible]
            _unique, first = np.unique(keys, return_index=True)
            take = np.flatnonzero(visible)[first]
            chosen = np.concatenate([chosen, pool[take]])
            weights = np.concatenate([weights, [lost_weight[k] for k in keys[first].tolist()]])
            optional = np.concatenate([optional, np.zeros(len(take), dtype=bool)])
            two_sided = np.concatenate([two_sided, p_two[take]])
            points = np.concatenate([points, p_points[take]])
            normals = np.concatenate([normals, p_normals[take]])
            owners = np.concatenate([owners, p_owners[take]])
    optional |= ~two_sided & ~_faces_space(region, ctx.grid, points, normals, voxel)
    if beyond is not None:
        # A room rig glimpses the outside through doors and windows; what it
        # glimpses there is not its subject and makes no problem areas.
        glimpse = optional & _dilate(beyond, 2).ravel()[ctx.sample_flat[chosen]]
        if glimpse.any():
            keep = ~glimpse
            chosen, weights, optional, two_sided = chosen[keep], weights[keep], optional[keep], two_sided[keep]
            points, normals, owners = points[keep], normals[keep], owners[keep]
    grounded = np.zeros(len(points), dtype=bool)
    if ground.any():
        # Ground counts, but - as when planning - it must not outvote the
        # buildings and objects just because it has the most square metres.
        on_ground = np.zeros(len(ground) + 1, bool)
        on_ground[:len(ground)] = ground
        grounded = on_ground[np.minimum(ctx.owner_ids[chosen], len(ground))]
        weights = weights * np.where(grounded, 0.35, 1.0)
    outer = np.zeros(len(points), dtype=bool)
    if outside is not None:
        outer = _dilate(outside, 2).ravel()[ctx.sample_flat[chosen]]
    T = len(points)

    # ---- every camera, as it renders ----------------------------------------
    field = ctx.field()
    views = np.zeros(T, dtype=np.int32)
    directions = np.zeros((T, 3))
    detail = np.full(T, np.inf)
    seen, rays = [], []
    distances = []
    tolerance = 1.5 * voxel
    for index, camera in enumerate(cameras):
        report(STAGE_CAMERAS, index / len(cameras), f"{camera.name} ({index + 1} of {len(cameras)})")
        rotation = camera.matrix[:3, :3]
        origin = camera.position
        local = (points - origin) @ rotation            # camera space: -Z forward, +Y up
        depth_z = -local[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = camera.fx * local[:, 0] / depth_z + camera.cx
            v = camera.cy - camera.fy * local[:, 1] / depth_z
        distance = np.linalg.norm(points - origin, axis=1)
        frame = ((depth_z > camera.clip_start) & (distance < camera.clip_end)
                 & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height))
        idx = np.flatnonzero(frame)
        if not len(idx):
            seen.append(np.zeros(0, np.int32))
            rays.append(np.zeros((0, 3), np.float32))
            continue
        ray = (points[idx] - origin) / distance[idx, None]
        facing = -np.einsum('ij,ij->i', ray, normals[idx])
        facing = np.where(two_sided[idx], np.abs(facing), facing)
        keep = facing > GRAZING
        idx, ray, facing = idx[keep], ray[keep], facing[keep]
        # A coarse depth image first; exact rays only for the uncertain band.
        cols = budget['image']
        rows = max(8, int(round(cols * camera.height / max(camera.width, 1))))
        px = (np.arange(cols) + 0.5) / cols * camera.width
        py = (np.arange(rows) + 0.5) / rows * camera.height
        gx, gy = np.meshgrid(px, py)
        local_dirs = np.stack([(gx - camera.cx) / camera.fx, (camera.cy - gy) / camera.fy,
                               -np.ones_like(gx)], axis=-1).reshape(-1, 3)
        local_dirs /= np.linalg.norm(local_dirs, axis=1, keepdims=True)
        world_dirs = local_dirs @ rotation.T
        limit = min(camera.clip_end,
                    float(np.linalg.norm(ctx.hi - ctx.lo) + np.linalg.norm(origin - 0.5 * (ctx.lo + ctx.hi))))
        depth = _march_from(field, ctx.grid, origin, world_dirs, np.full(len(world_dirs), limit))
        pix_x = np.clip((u[idx] / camera.width * cols).astype(np.int64), 0, cols - 1)
        pix_y = np.clip((v[idx] / camera.height * rows).astype(np.int64), 0, rows - 1)
        first = depth[pix_y * cols + pix_x]
        d = distance[idx]
        pixel_angle = max(camera.width / cols / camera.fx, camera.height / rows / camera.fy)
        sure = d <= first + tolerance
        maybe = np.flatnonzero(~sure & (d <= first + tolerance + d * pixel_angle * 1.5))
        if len(maybe):
            hits = _march_from(field, ctx.grid, origin, ray[maybe], d[maybe] - 1.8 * voxel)
            sure[maybe] = ~np.isfinite(hits)
        idx, ray, facing, d = idx[sure], ray[sure], facing[sure], d[sure]
        views[idx] += 1
        directions[idx] -= ray                     # surface-to-camera directions
        footprint = d / (camera.fx * np.maximum(facing, 1e-3))
        np.minimum.at(detail, idx, footprint)
        seen.append(idx.astype(np.int32))
        rays.append((-ray).astype(np.float32))
        if len(d):
            distances.append(np.median(d))

    # ---- only what the rig can be expected to capture ----------------------
    # Outside, surfaces far beyond every camera's own viewing distance - the
    # far end of a street, say - are out of this rig's reach, not failures
    # of it. Rooms are judged whole: an unvisited room is exactly what the
    # check must show.
    out_of_reach = 0
    view_distance = float(np.median(distances)) if distances else 4.0 * voxel
    within = np.ones(T, dtype=bool)
    if outer.any() or optional.any():
        span = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        reach = max(4.0 * view_distance, 0.5 * span)
        nearest = (planner._nearest_distance(points, positions) if outer.any()
                   else np.zeros(T))
        within = (~outer | (views > 0) | (nearest <= reach)) & (~optional | (views > 0))
        # Open ground counts around the things standing on it; the empty
        # stretches between them are nobody's subject unless a camera looks.
        bare = outer & grounded & (views == 0)
        if bare.any() and (~grounded).any():
            within &= ~bare | _near_structures(points, grounded, 1.5 * view_distance)
    if core is not None:
        # One area of a split check keeps only its own points.
        within &= np.all((points[:, :2] >= core[0][:2]) & (points[:, :2] < core[1][:2]), axis=1)
    if not within.all():
        keep = np.flatnonzero(within)
        remap = np.full(T, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        points, normals, weights = points[keep], normals[keep], weights[keep]
        views, directions, detail = views[keep], directions[keep], detail[keep]
        owners = owners[keep]
        for c, idx in enumerate(seen):
            mapped = remap[idx]
            seen[c] = mapped[mapped >= 0].astype(np.int32)
            rays[c] = rays[c][mapped >= 0]
        out_of_reach = int(T - len(keep))
        T = len(keep)

    # Point every normal toward the cameras that saw it, so the overlay sits
    # on the photographed side of the surface.
    seen_any = views > 0
    flip = seen_any & (np.einsum('ij,ij->i', normals, directions) < 0)
    normals[flip] *= -1.0

    report(STAGE_SUMMARY, 0.2, "Grading surfaces")
    result.points = points.astype(np.float32)
    result.normals = normals.astype(np.float32)
    result.owners = owners
    result.owner_names = list(geometry.owner_names)
    result.raw_weight = float(weights.sum())
    result.weights = weights / max(weights.sum(), 1e-12)
    # Tiles for display: one per lattice cell the points were thinned to.
    result.cell = float(cell) if cell else 2.0 * voxel
    result.views = views
    result.directions = directions
    with np.errstate(invalid='ignore', divide='ignore'):
        resultant = np.linalg.norm(directions, axis=1) / np.maximum(views, 1)
    result.spread = np.where(views >= 2, 2.0 * np.degrees(np.arccos(np.clip(resultant, -1, 1))),
                             0.0).astype(np.float32)
    result.detail = detail.astype(np.float32)
    result.camera_names = [camera.name for camera in cameras]
    result.camera_positions = positions
    result.seen, result.rays = seen, rays
    result.voxel = voxel
    result.field = field
    result.grid = ctx.grid
    result.region = region
    result.view_distance = view_distance
    result.scope = scope
    result.extent = float(np.linalg.norm(ctx.hi - ctx.lo))
    seen_detail = detail[np.isfinite(detail)]
    result.stats = dict(
        cameras=len(cameras), points=T, seconds=0.0, out_of_reach=out_of_reach,
        median_views=float(np.median(views)) if T else 0.0,
        median_spread=float(np.median(result.spread[views >= 2])) if np.any(views >= 2) else 0.0,
        median_detail=float(np.median(seen_detail)) if len(seen_detail) else 0.0,
    )
    return result


def grade(result, target_views):
    """Verdicts, problem areas and redundant cameras for a target view count.

    Cheap - no rendering - so changing the target after a check re-grades
    at once.
    """
    K = max(1, int(target_views))
    result.target_views = K
    views, spread, detail = result.views, result.spread, result.detail
    verdict = np.full(len(views), GOOD, dtype=np.int8)
    verdict[(views < K) | (spread < GOOD_SPREAD)] = FAIR
    verdict[(views == 1) | ((views >= 2) & (spread < WEAK_SPREAD))] = WEAK
    verdict[views == 0] = UNSEEN
    median = result.stats.get("median_detail", 0.0)
    if median > 0:
        # Far lower resolution than the rest of the capture is not "good".
        verdict[(verdict == GOOD) & (detail > 3.0 * median)] = FAIR
    result.grade = verdict
    result.fractions = {name: float(result.weights[verdict == code].sum())
                        for code, name in CLASS_NAMES.items()}
    result.areas = _problem_areas(result.points, result.normals, result.weights, verdict,
                                  views, result.extent, result.voxel)
    result.redundant = result.duplicates + _redundant(result, K)
    return result


def _redundant(result, K):
    """Cameras that can all go together without any surface getting worse.

    Greedy, least useful first: a camera goes when every surface it sees
    keeps at least K views and no less angular spread than it needs. Each
    removal updates the counts, so the whole list is safe to delete at once -
    two cameras covering for each other are never both listed.
    """
    counts = result.views.astype(np.int64).copy()
    sums = result.directions.copy()
    weights = result.weights
    usefulness = []
    for idx in result.seen:
        usefulness.append(float(np.sum(weights[idx] / np.maximum(counts[idx], 1))) if len(idx) else 0.0)
    removed = []
    for c in np.argsort(usefulness, kind="stable"):
        idx, ray = result.seen[c], result.rays[c].astype(np.float64)
        if len(idx):
            after = counts[idx] - 1
            if np.any(after < K):
                continue
            new_sum = sums[idx] - ray
            with np.errstate(invalid='ignore', divide='ignore'):
                before = 2 * np.degrees(np.arccos(np.clip(
                    np.linalg.norm(sums[idx], axis=1) / np.maximum(counts[idx], 1), -1, 1)))
                later = 2 * np.degrees(np.arccos(np.clip(
                    np.linalg.norm(new_sum, axis=1) / np.maximum(after, 1), -1, 1)))
            if np.any(later < np.minimum(before, GOOD_SPREAD) - 1e-6):
                continue
            counts[idx] = after
            sums[idx] = new_sum
        removed.append(result.camera_names[c])
    return removed


def _problem_areas(points, normals, weights, grade, views, extent, voxel, limit=12):
    """Group unseen and weak points into a few numbered places to fix.

    Points join an area only when they are close *and* face the same way, so
    a floor, a wall and a ceiling are separate areas; an area larger than a
    camera can take in at once is cut into tiles. Each area is then one place
    to point a camera at.
    """
    bad = np.flatnonzero(grade <= WEAK)
    if not len(bad):
        return []
    cell = max(3.0 * voxel, 0.02 * extent)
    tile = max(8.0 * cell, 0.12 * extent)
    # Six orientation bins: the dominant axis of the normal and its sign.
    n = normals[bad]
    axis = np.argmax(np.abs(n), axis=1)
    facing = axis * 2 + (n[np.arange(len(n)), axis] < 0)
    keys = np.floor(points[bad] / cell).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 3
    flat = ((keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]) * 6 + facing
    cells, member = np.unique(flat, return_inverse=True)
    label = np.arange(len(cells))
    offsets = [((dx * span[1] + dy) * span[2] + dz) * 6
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) > (0, 0, 0)]
    pairs_a, pairs_b = [], []
    for offset in offsets:
        target = cells + offset                # same orientation bin
        pos = np.minimum(np.searchsorted(cells, target), len(cells) - 1)
        hit = cells[pos] == target
        pairs_a.append(np.flatnonzero(hit))
        pairs_b.append(pos[hit])
    a, b = np.concatenate(pairs_a), np.concatenate(pairs_b)
    for _ in range(200):
        if not len(a):
            break
        before = label.copy()
        low = np.minimum(label[a], label[b])
        np.minimum.at(label, a, low)
        np.minimum.at(label, b, low)
        label = label[label]
        if np.array_equal(before, label):
            break
    group = label[member].astype(np.int64)
    # Cut oversized groups into tiles.
    tiles = np.floor(points[bad] / tile).astype(np.int64)
    tiles -= tiles.min(axis=0)
    tspan = tiles.max(axis=0) + 1
    tile_id = (tiles[:, 0] * tspan[1] + tiles[:, 1]) * tspan[2] + tiles[:, 2]
    group = group * int(tspan.prod()) + tile_id
    total = float(weights.sum()) or 1.0
    areas = []
    for g in np.unique(group):
        idx = bad[group == g]
        share = float(weights[idx].sum()) / total
        if share < 0.001:
            continue
        w = weights[idx] / max(weights[idx].sum(), 1e-12)
        centre = (points[idx] * w[:, None]).sum(axis=0)
        normal = (normals[idx] * w[:, None]).sum(axis=0)
        consistent = float(np.linalg.norm(normal))
        normal = normal / consistent if consistent > 1e-6 else np.zeros(3)
        size = float(np.linalg.norm(np.ptp(points[idx], axis=0)))
        unseen = float(np.sum(w[grade[idx] == UNSEEN]))
        if unseen > 0.5:
            issue = "never seen"
        elif np.mean(views[idx]) <= 1.2:
            issue = "seen by one camera"
        else:
            issue = "views too similar"
        areas.append(dict(centre=centre, normal=normal, normal_confidence=consistent,
                          share=share, size=max(size, voxel), issue=issue,
                          views=float(np.mean(views[idx])), count=int(len(idx))))
    areas.sort(key=lambda area: -area["share"])
    return areas[:limit]


def fix_poses(result, area, count, rng_seed=0):
    """Camera poses that look at a problem area from clear, varied directions.

    Tries directions around the area's surface normal (both sides when the
    normal is ambiguous), keeps positions in the judged free space with a
    clear line of sight, and spreads the picks at least 30 degrees apart.
    """
    grid, field = result.grid, result.field
    centre = np.asarray(area["centre"], dtype=np.float64)
    base = float(np.clip(max(result.view_distance * 0.8, area["size"] * 1.2),
                         4.0 * result.voxel, 50.0 * result.voxel + result.view_distance))
    # Fibonacci sphere of candidate directions.
    n = 160
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    normal = np.asarray(area["normal"], dtype=np.float64)
    if area["normal_confidence"] > 0.3:
        score = np.abs(dirs @ normal)          # face the surface, either side
    else:
        score = np.ones(n)
    score = score - 0.15 * np.abs(dirs[:, 2])  # mild preference for level views
    picks = []
    # A tight place - a tunnel, a corner - may have no room at the usual
    # distance: come closer before giving up.
    for distance in (base, 0.6 * base, 0.35 * base):
        distance = max(distance, 3.0 * result.voxel)
        positions = centre + dirs * distance
        cell, inside = grid.ijk(positions)
        ok = inside.copy()
        flat = np.zeros(n, dtype=np.int64)
        flat[inside] = grid.flat(cell[inside])
        ok[inside] &= result.region.ravel()[flat[inside]]
        clear = field[flat[inside]]
        ok[inside] &= (clear >= 1) & (clear != 255)     # free, and not hugging a surface
        idx = np.flatnonzero(ok)
        if len(idx):
            hits = _march(field, grid, positions[idx], -dirs[idx],
                          np.full(len(idx), distance - 2.0 * result.voxel))
            idx = idx[~np.isfinite(hits)]
        order = idx[np.argsort(-score[idx])]
        for k in order:
            if all(np.dot(dirs[k], dirs[j]) < math.cos(math.radians(30.0)) for j, _p in picks):
                picks.append((k, positions[k]))
            if len(picks) >= count:
                break
        if len(picks) >= count:
            break
    poses = []
    for k, position in picks:
        forward = -dirs[k]
        f, _right, up = planner._basis(forward)
        poses.append((position, f, up))
    return poses
