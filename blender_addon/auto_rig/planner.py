# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Automatic camera-rig planning. Pure numpy: no ``bpy`` anywhere in here.

The problem is classic *view planning* - choose a small set of viewpoints
that together observe every surface well - specialised to what trains a good
Gaussian Splat from a synthetic scene whose poses are exact:

    1. Every visible surface is seen by several cameras        (redundancy)
    2. ...from genuinely different directions                   (angular spread)
    3. ...at a useful distance and not at a grazing angle       (resolution)
    4. ...with the camera positions spread through the space
       someone will later look at the Splat from               (viewpoint spread)

SfM feature overlap is irrelevant here - the add-on writes the true poses -
so it is deliberately not an objective.

The pipeline follows the sampling-based planners from the literature (Scott,
Roth & Rivest 2003; Roberts et al. 2017; Smith et al. 2018):

    free space     Voxelise the surface samples. Flood free space from each
                   scan blob with vectorised run sweeps. Progressive
                   sealing (closing door-sized openings) decides whether the
                   blob is enclosed - interior - or open to the world -
                   exterior - and keeps an exterior scan out of rooms that
                   are only reachable through a door.
    targets        Surface samples bordering the scanned space, thinned to
                   an even, area-weighted set.
    candidates     Evenly spread viewpoints inside that space, clear of
                   surfaces. For each: a depth cube map ray-marched through
                   the voxel grid gives true occlusion, and a yaw/pitch
                   histogram of what it can see proposes its best headings.
    selection      Lazy-greedy maximisation of a saturating, submodular
                   coverage score (diminishing returns once a surface has K
                   good views), with an angular-novelty term so repeated
                   directions earn little, and a farthest-view spread factor.

Everything is vectorised; cost scales with the voxel and target budgets of
the chosen quality, never with how many objects the scene contains.
"""

import heapq
import math
import time

import numpy as np


# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------

#: Every budget the planner uses, per quality. Nothing here depends on the
#: scene's object count - that is what keeps a huge scene as fast as a small
#: one at the same quality.
PRESETS = {
    'DRAFT': dict(
        voxels=1_500_000, samples=700_000, targets=3000, candidates=240,
        views_per_candidate=8, diversity_deg=35.0, cube=16),
    'STANDARD': dict(
        voxels=4_000_000, samples=1_600_000, targets=7000, candidates=520,
        views_per_candidate=10, diversity_deg=28.0, cube=24),
    'HIGH': dict(
        voxels=9_000_000, samples=3_000_000, targets=14000, candidates=1000,
        views_per_candidate=12, diversity_deg=22.0, cube=32),
}

INTERIOR = 'INTERIOR'
#: Deepest sealing, in voxels. Fine grids need many levels for a door-sized
#: opening: at 3 cm voxels a 0.9 m door takes 15. With 12 a small, finely
#: gridded room could not close its door and merged with the outside.
MAX_SEAL_STEPS = 24
#: Finest voxel a room's own grid gets. Finer adds nothing to where cameras
#: go - furniture and doors are resolved - but makes every visibility ray and
#: every sealing level slower (High quality on a small room reached 4 cm).
ROOM_MIN_VOXEL = 0.05
EXTERIOR = 'EXTERIOR'
OBJECT = 'OBJECT'


class Cancelled(Exception):
    """Raised inside the planner when the user stops it."""


class PlanError(RuntimeError):
    """A plan that cannot be made, with a message the user can act on."""


class Blob:
    """An interior or exterior system: a point in the space to capture."""

    def __init__(self, name, center, radius, mode='AUTO', bounded=False,
                 views=3, reach='NEARBY', max_cameras=0, clearance=None,
                 seal=None, limit_height=None, height_min=None, height_max=None,
                 allow_below=None, stop_when_covered=None):
        self.name = str(name)
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = max(1e-4, float(radius))
        self.mode = mode
        self.bounded = bool(bounded)
        self.views = max(1, int(views))
        self.reach = reach                  # exterior: 'NEARBY' or 'SCENE'
        # Per-system placement; None means "use the request's default".
        self.max_cameras = max(0, int(max_cameras))
        self.clearance = clearance
        self.seal = seal
        self.limit_height = limit_height
        self.height_min = height_min
        self.height_max = height_max
        self.allow_below = allow_below
        self.stop_when_covered = stop_when_covered


class ObjectGroup:
    """An object/collection system: surfaces to orbit, given as an owner mask."""

    def __init__(self, name, owners, views=3, max_cameras=0, clearance=None,
                 stop_when_covered=None):
        self.name = str(name)
        self.owners = np.asarray(owners, dtype=bool)
        self.views = max(1, int(views))
        self.max_cameras = max(0, int(max_cameras))
        self.clearance = clearance
        self.stop_when_covered = stop_when_covered


class Request:
    """Everything a plan needs, gathered on Blender's main thread.

    Systems combine freely: interior and exterior blobs and object groups are
    planned together, and every surface they bring is covered by one joint
    selection of cameras.
    """

    def __init__(self, geometry, *, blobs=(), objects=(),
                 tan_h=0.36, tan_v=0.24, quality='STANDARD',
                 max_cameras=500, stop_when_covered=True,
                 clearance=0.3, seal_size=1.5, limit_height=False,
                 height_min=0.4, height_max=2.0, allow_below=False, seed=0):
        self.geometry = geometry
        self.blobs = list(blobs)
        self.objects = list(objects)
        self.tan_h = float(tan_h)
        self.tan_v = float(tan_v)
        self.preset = dict(PRESETS.get(quality, PRESETS['STANDARD']))
        self.max_cameras = max(1, int(max_cameras))
        self.stop_when_covered = bool(stop_when_covered)
        self.clearance = max(0.0, float(clearance))
        self.seal_size = max(0.0, float(seal_size))
        self.limit_height = bool(limit_height)
        self.height_min = float(height_min)
        self.height_max = float(height_max)
        self.allow_below = bool(allow_below)
        self.seed = int(seed)
        # Fill every system's unset settings from these defaults once, so the
        # planner reads one place.
        for blob in self.blobs:
            blob.clearance = self.clearance if blob.clearance is None else max(0.0, float(blob.clearance))
            blob.seal = self.seal_size if blob.seal is None else max(0.0, float(blob.seal))
            blob.limit_height = self.limit_height if blob.limit_height is None else bool(blob.limit_height)
            blob.height_min = self.height_min if blob.height_min is None else float(blob.height_min)
            blob.height_max = self.height_max if blob.height_max is None else float(blob.height_max)
            blob.allow_below = self.allow_below if blob.allow_below is None else bool(blob.allow_below)
        for group in self.objects:
            group.clearance = self.clearance if group.clearance is None else max(0.0, float(group.clearance))
        for system in self.blobs + self.objects:
            system.stop_when_covered = (self.stop_when_covered if system.stop_when_covered is None
                                        else bool(system.stop_when_covered))


class Result:
    def __init__(self):
        self.positions = np.zeros((0, 3))
        self.forwards = np.zeros((0, 3))
        self.ups = np.zeros((0, 3))
        self.regions = []       # [{"name", "kind", "voxels"}]
        self.warnings = []
        self.stats = {}
        self.target_points = np.zeros((0, 3), np.float32)
        self.target_coverage = np.zeros(0, np.float32)


class _Reporter:
    """Thread-safe progress: the worker only writes plain values here."""

    def __init__(self, callback=None, cancelled=None):
        self.callback = callback
        self.cancelled = cancelled

    def __call__(self, stage, fraction, message=""):
        if self.cancelled is not None and self.cancelled():
            raise Cancelled()
        if self.callback is not None:
            self.callback(stage, fraction, message)


STAGE_SPACE = "Free space"
STAGE_VISIBILITY = "Visibility"
STAGE_SELECTION = "Camera selection"


# ---------------------------------------------------------------------------
# Voxel grid primitives
# ---------------------------------------------------------------------------

class _Grid:
    def __init__(self, lo, hi, voxel):
        self.voxel = float(voxel)
        self.origin = np.asarray(lo, dtype=np.float64)
        size = np.asarray(hi, dtype=np.float64) - self.origin
        self.shape = tuple(int(v) for v in np.maximum(4, np.ceil(size / self.voxel)))
        self.size = int(np.prod(self.shape))

    def ijk(self, points):
        """Integer voxel coordinates, and which points fall inside the grid."""
        cell = np.floor((np.asarray(points, np.float64) - self.origin)
                        / self.voxel).astype(np.int64)
        inside = np.all((cell >= 0) & (cell < np.asarray(self.shape)), axis=1)
        return cell, inside

    def flat(self, cell):
        return np.ravel_multi_index(cell.T, self.shape)

    def centres(self, cell):
        return self.origin + (np.asarray(cell, np.float64) + 0.5) * self.voxel


def _dilate(mask, steps=1):
    """6-connected (Manhattan-ball) dilation, in place of scipy."""
    out = mask.copy()
    for _ in range(int(steps)):
        src = out.copy()
        out[1:] |= src[:-1]
        out[:-1] |= src[1:]
        out[:, 1:] |= src[:, :-1]
        out[:, :-1] |= src[:, 1:]
        out[:, :, 1:] |= src[:, :, :-1]
        out[:, :, :-1] |= src[:, :, 1:]
    return out


class _FreeSpace:
    """Free voxels at one sealing level, with run ids for fast flooding.

    A flood fill is normally a queue of voxels - hopeless in Python for
    millions of them. Instead every free voxel knows which contiguous run it
    belongs to along each axis. One sweep marks a whole run filled as soon as
    any voxel in it is, so a flood converges in a handful of sweeps: one per
    turn the free space takes, not one per voxel.
    """

    def __init__(self, free):
        self.free = free
        self.shape = free.shape
        flat_free = free.ravel()
        self.index = np.flatnonzero(flat_free)
        # rank[v] is voxel v's position among the free voxels: a gather
        # instead of a binary search everywhere a voxel id is looked up.
        self.rank = (np.cumsum(flat_free, dtype=np.int64) - 1).astype(np.int32)
        self.runs = []
        count = len(self.index)
        if not count:
            return
        ids = np.arange(free.size, dtype=np.int32).reshape(self.shape)
        for axis in range(3):
            # A run starts at a free voxel whose predecessor along the axis
            # is not free. Flat ids grow along every axis, so a running
            # maximum of start ids hands each voxel its own run's start -
            # no transposed copies of the grid needed.
            start = free.copy()
            lead = [slice(None)] * 3
            trail = [slice(None)] * 3
            lead[axis], trail[axis] = slice(1, None), slice(None, -1)
            start[tuple(lead)] &= ~free[tuple(trail)]
            label = np.where(start, ids, -1)
            np.maximum.accumulate(label, axis=axis, out=label)
            compact = np.cumsum(start.ravel(), dtype=np.int32) - 1
            run = compact[label.ravel()[self.index]]
            self.runs.append((run, int(compact[-1]) + 1))

    def position(self, flat):
        """Index into ``self.index`` of each flat voxel id, -1 when not free."""
        flat = np.atleast_1d(np.asarray(flat, np.int64))
        return np.where(self.free.ravel()[flat], self.rank[flat], -1)

    def flood(self, seeds, report=None, stop_at=None):
        """Boolean over ``self.index``: every free voxel connected to seeds.

        With ``stop_at`` (positions of open-world voxels) the flood gives up
        as soon as it reaches one and returns ``None``: proving that a space
        leaks never needs the whole outside filled.
        """
        filled = np.zeros(len(self.index), dtype=bool)
        seeds = np.asarray(seeds, np.int64)
        seeds = seeds[seeds >= 0]
        if not len(seeds) or not self.runs:
            return filled
        filled[seeds] = True
        previous = -1
        for sweep in range(4096):
            for run, count in self.runs:
                hit = np.zeros(count, dtype=bool)
                hit[run[filled]] = True
                filled = hit[run]
                if stop_at is not None and len(stop_at) and filled[stop_at].any():
                    return None
            current = int(np.count_nonzero(filled))
            if current == previous:
                break
            previous = current
            if report is not None and sweep % 4 == 3:
                report()
        return filled

    def open_positions(self, open_faces):
        """Positions of the free voxels lying on open faces of the grid."""
        mask = np.zeros(self.shape, dtype=bool)
        for axis in range(3):
            for side, index in ((0, 0), (1, -1)):
                if open_faces[axis][side]:
                    sl = [slice(None)] * 3
                    sl[axis] = index
                    mask[tuple(sl)] = True
        return self.position(np.flatnonzero(mask & self.free))

    def to_grid(self, filled):
        grid = np.zeros(self.shape, dtype=bool)
        grid.ravel()[self.index[filled]] = True
        return grid


def _touches(grid_mask, open_faces):
    """Whether a region reaches any open face of the domain (it leaks)."""
    checks = (
        (0, 0, grid_mask[0]), (0, 1, grid_mask[-1]),
        (1, 0, grid_mask[:, 0]), (1, 1, grid_mask[:, -1]),
        (2, 0, grid_mask[:, :, 0]), (2, 1, grid_mask[:, :, -1]),
    )
    return any(open_faces[axis][side] and face.any()
               for axis, side, face in checks)


def _dilate_cube(mask, steps=1):
    """Dilation by a full 3x3x3 cube per step (Chebyshev ball).

    Unlike the 6-connected ``_dilate`` it reaches into the corners of a room
    - where floor meets wall - in as many steps as it reaches a flat wall.
    """
    out = mask.copy()
    for _ in range(int(steps)):
        for axis in range(3):
            lead = [slice(None)] * 3
            trail = [slice(None)] * 3
            lead[axis], trail[axis] = slice(1, None), slice(None, -1)
            src = out.copy()
            out[tuple(lead)] |= src[tuple(trail)]
            out[tuple(trail)] |= src[tuple(lead)]
    return out


def _grow_back(region, steps, free):
    """Undo the erosion that sealing caused, without re-crossing a seal far."""
    for _ in range(int(steps)):
        region = _dilate(region, 1) & free
    return region


def _sphere_mask(grid, centre, radius):
    axes = [grid.origin[a] + (np.arange(grid.shape[a]) + 0.5) * grid.voxel
            - centre[a] for a in range(3)]
    x, y, z = np.ix_(*axes)
    return (x * x + y * y + z * z) <= radius * radius


# ---------------------------------------------------------------------------
# Even thinning (stratified by hashed cells)
# ---------------------------------------------------------------------------

def _cell_pick(points, cell, rng):
    """One random representative per occupied cell, with the cell's count."""
    keys = np.floor(points / cell).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    order = rng.permutation(len(points))
    _unique, first, counts = np.unique(flat[order], return_index=True,
                                       return_counts=True)
    return order[first], counts


def _thin(points, count, rng, dimension=2.0, return_cell=False):
    """About ``count`` evenly spread picks from ``points``, plus weights.

    With ``return_cell`` also the lattice cell size: every pick is then the
    only one in its cell of a world-aligned grid, which a display can tile.
    """
    n = len(points)
    if n <= count:
        if return_cell:
            return np.arange(n), np.ones(n), None
        return np.arange(n), np.ones(n)
    extent = float(np.max(np.ptp(points, axis=0))) or 1.0
    probe = points
    if n > 250_000:
        probe = points[rng.choice(n, 250_000, replace=False)]
    # A random probe keeps nearly every occupied cell while cells hold many
    # points, which is always the case when thinning to a small budget.
    lo, hi = extent * 1e-5, extent
    cell = extent / count ** (1.0 / dimension)
    for _ in range(14):
        picked, _counts = _cell_pick(probe, cell, rng)
        found = len(picked)
        if abs(found - count) <= 0.08 * count:
            break
        if found > count:
            lo = cell
        else:
            hi = cell
        cell = math.sqrt(lo * hi) if lo > 0 else (lo + hi) * 0.5
    picked, counts = _cell_pick(points, cell, rng)
    if return_cell:
        return picked, counts.astype(np.float64), cell
    if len(picked) > count * 1.25:
        keep = rng.choice(len(picked), int(count * 1.25), replace=False)
        picked, counts = picked[keep], counts[keep]
    return picked, counts.astype(np.float64)


# ---------------------------------------------------------------------------
# Visibility: depth cube maps ray-marched through the occupancy grid
# ---------------------------------------------------------------------------

def _cube_directions(res):
    """Unit direction per cube-map pixel, laid out as face*res*res + i*res + j."""
    t = (np.arange(res) + 0.5) / res * 2.0 - 1.0
    uu, vv = np.meshgrid(t, t, indexing='ij')
    uu, vv = uu.ravel(), vv.ravel()
    dirs = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for sign in (1.0, -1.0):
            d = np.zeros((res * res, 3))
            d[:, axis] = sign
            d[:, others[0]] = uu
            d[:, others[1]] = vv
            dirs.append(d)
    dirs = np.concatenate(dirs)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def _cube_pixel(u, res):
    """Cube-map pixel of each unit direction (matches ``_cube_directions``)."""
    a = np.abs(u)
    axis = np.argmax(a, axis=1)
    rows = np.arange(len(u))
    major = a[rows, axis]
    sign_face = (u[rows, axis] < 0).astype(np.int64)
    other0 = np.where(axis == 0, 1, 0)
    other1 = np.where(axis == 2, 1, 2)
    uc = u[rows, other0] / major
    vc = u[rows, other1] / major
    i = np.clip(((uc + 1.0) * 0.5 * res).astype(np.int64), 0, res - 1)
    j = np.clip(((vc + 1.0) * 0.5 * res).astype(np.int64), 0, res - 1)
    return (axis * 2 + sign_face) * res * res + i * res + j


def _clearance_field(occ, levels=10):
    """Voxels of guaranteed-empty space around each voxel (Chebyshev).

    ``field[v] = k`` means no occupied voxel lies within k voxels of v's
    boundary, so a ray inside v may advance k voxels without testing. Rays
    through open air then move in long strides instead of half-voxel steps.
    """
    field = np.zeros(occ.shape, dtype=np.uint8)
    reach = occ.copy()
    for _ in range(levels):
        for axis in range(3):
            src = reach.copy()
            lead = [slice(None)] * 3
            trail = [slice(None)] * 3
            lead[axis], trail[axis] = slice(1, None), slice(None, -1)
            reach[tuple(lead)] |= src[tuple(trail)]
            reach[tuple(trail)] |= src[tuple(lead)]
        field += ~reach
    return field


def _depth_maps(field_flat, grid, origins, dirs, max_dist):
    """Depth cube maps: first-hit distance per (origin, cube pixel)."""
    b, p = len(origins), len(dirs)
    depth = _march(field_flat, grid, np.repeat(origins, p, axis=0),
                   np.tile(dirs, (b, 1)), np.full(b * p, max_dist))
    return depth.reshape(b, p)


def _march(field_flat, grid, origins, dirs, max_dist):
    """Distance to the first occupied voxel along each ray, inf when none.

    Each ray stops at its own ``max_dist``. Strides come from the clearance
    field, so open space costs a handful of steps per ray.
    """
    n = len(origins)
    shape = np.asarray(grid.shape)
    strides = np.array([grid.shape[1] * grid.shape[2], grid.shape[2], 1])
    pos = ((origins - grid.origin) / grid.voxel).astype(np.float32)
    direction = dirs.astype(np.float32)
    t = np.zeros(n, dtype=np.float32)
    ids = np.arange(n)
    depth = np.full(n, np.inf, dtype=np.float32)
    limit = (np.asarray(max_dist, np.float64) / grid.voxel).astype(np.float32)
    for _ in range(4096):
        cell = pos.astype(np.int64)   # positions are >= 0 while in bounds
        inside = np.all((pos >= 0) & (cell < shape), axis=1) & (t <= limit)
        if not inside.all():
            pos, direction, t, ids, cell, limit = (
                pos[inside], direction[inside], t[inside], ids[inside],
                cell[inside], limit[inside])
        if not len(ids):
            break
        clear = field_flat[cell @ strides]
        hit = clear == 255
        if hit.any():
            depth[ids[hit]] = t[hit] * grid.voxel
            keep = ~hit
            pos, direction, t, ids, clear, limit = (
                pos[keep], direction[keep], t[keep], ids[keep], clear[keep], limit[keep])
            if not len(ids):
                break
        stride = np.maximum(clear.astype(np.float32), 0.5)
        pos += direction * stride[:, None]
        t += stride
    return depth


def _basis(forward):
    """Camera right/up for a forward vector, keeping world Z up (no roll)."""
    fx, fy, fz = (float(v) for v in forward)
    length = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / length, fy / length, fz / length
    rx, ry = fy, -fx                      # forward x world Z
    horizontal = math.sqrt(rx * rx + ry * ry)
    if horizontal < 1e-4:
        rx, ry, horizontal = 1.0, 0.0, 1.0
    rx, ry = rx / horizontal, ry / horizontal
    up = (ry * fz, -rx * fz, rx * fy - ry * fx)   # right x forward
    return (np.array((fx, fy, fz)), np.array((rx, ry, 0.0)), np.array(up))


# ---------------------------------------------------------------------------
# Analysis contexts: one voxel grid each, at the resolution its space needs
# ---------------------------------------------------------------------------

class _Context:
    """Samples, occupancy and free space for one box of the scene.

    A house in a city is analysed twice over: the city at a voxel size that
    fits its budget, the house's rooms in a box of their own at a voxel size
    that resolves doors and furniture. Every context has the same budget, so
    detail follows the space being scanned rather than the whole scene.
    """

    def __init__(self, geometry, lo, hi, *, voxels, samples, rng, exclude,
                 open_faces, seal_size, tick, min_voxel=0.0):
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        volume = float(np.prod(np.maximum(hi - lo, 1e-6)))
        voxel = max((volume / voxels) ** (1.0 / 3.0), min_voxel)
        # Keep flat boxes from producing absurd grid dimensions.
        voxel = max(voxel, float(np.max(hi - lo)) / 1024.0)
        lo, hi = lo - 2.5 * voxel, hi + 2.5 * voxel
        self.grid = grid = _Grid(lo, hi, voxel)
        self.lo, self.hi = lo, grid.origin + np.asarray(grid.shape) * voxel
        self.open_faces = open_faces(lo, hi, voxel)
        self.tick = tick
        points, normals, owners = geometry.sample(
            voxel=voxel, lo=lo, hi=hi, cap=samples, rng=rng, exclude=exclude)
        cell, inside = grid.ijk(points)
        self.points, self.normals = points[inside], normals[inside]
        self.owner_ids, cell = owners[inside], cell[inside]
        self.sample_flat = grid.flat(cell) if len(cell) else np.zeros(0, np.int64)
        self.occ = np.zeros(grid.shape, dtype=bool)
        self.occ.ravel()[self.sample_flat] = True
        self.z_floor = int(np.clip(np.min(cell[:, 2]) + 1, 0, grid.shape[2] - 1)) if len(cell) else 0
        self.seal_steps = 1
        if seal_size > 0:
            self.seal_steps = int(np.clip(math.ceil(seal_size / (2 * voxel)), 1, MAX_SEAL_STEPS))
        self.barriers = {1: _dilate(self.occ, 1)}
        self.free1 = ~self.barriers[1]
        self._levels = {}
        self._probes = {}
        self._field = None

    # ---- free space ------------------------------------------------------
    def level(self, r):
        if r not in self._levels:
            if r not in self.barriers:
                previous = max(k for k in self.barriers if k < r)
                self.barriers[r] = _dilate(self.barriers[previous], r - previous)
            space = _FreeSpace(~self.barriers[r])
            space.open = space.open_positions(self.open_faces)
            self._levels[r] = space
        return self._levels[r]

    def seed(self, space, centre):
        """Position of the free voxel nearest ``centre``, or -1."""
        c, _inside = self.grid.ijk(np.asarray(centre)[None])
        c = np.clip(c[0], 0, np.asarray(self.grid.shape) - 1)
        for reach in range(0, 9):
            lo_c = np.maximum(c - reach, 0)
            hi_c = np.minimum(c + reach + 1, self.grid.shape)
            box = space.free[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]]
            if box.any():
                local = np.argwhere(box) + lo_c
                best = local[np.argmin(np.sum((local - c) ** 2, axis=1))]
                return space.position(self.grid.flat(best[None]))
        return np.array([-1])

    def probe(self, centre, r):
        """'SOLID', 'LEAK', or the enclosed region at sealing level ``r``."""
        key = (tuple(np.round(centre, 9)), r)
        if key not in self._probes:
            space = self.level(r)
            seed = self.seed(space, centre)
            if seed[0] < 0:
                self._probes[key] = 'SOLID'
            else:
                filled = space.flood(seed, self.tick, stop_at=space.open)
                self._probes[key] = 'LEAK' if filled is None else space.to_grid(filled)
        return self._probes[key]

    def steps_for(self, seal_size):
        """Sealing levels that close openings up to ``seal_size`` in this grid."""
        if seal_size <= 0:
            return 1
        return int(np.clip(math.ceil(seal_size / (2 * self.grid.voxel)), 1, MAX_SEAL_STEPS))

    def widest_steps(self, seal_size):
        """Sealing for a room whose openings exceed the doorway width: twice
        that width, at least 2.5 m - a garage door, a glass front."""
        return self.steps_for(max(2.0 * seal_size, 2.5))

    def enclose(self, centre, steps=None):
        """(sealing level, region, seeded): the smallest sealing that encloses.

        Seal progressively larger openings until the space around ``centre``
        no longer reaches the open world. Sealing only ever closes more, so
        bisection finds it; and a leak is proven the moment a flood touches
        the open world, so testing a door-open room never fills the outside.
        """
        def encloses(r):
            return not isinstance(self.probe(centre, r), str)

        def solid(r):
            state = self.probe(centre, r)
            return isinstance(state, str) and state == 'SOLID'

        enclosed_at = None
        if encloses(1):
            enclosed_at = 1
        elif (steps or self.seal_steps) > 1:
            # A blob in a corridor narrower than the sealing size ends up in
            # solid voxels at high levels; seal only as far as it stays free.
            top = steps or self.seal_steps
            if solid(top):
                low, high = 1, top
                while high - low > 1:
                    mid = (low + high) // 2
                    if solid(mid):
                        high = mid
                    else:
                        low = mid
                top = low
            if top > 1 and encloses(top):
                low, high = 1, top
                while high - low > 1:
                    mid = (low + high) // 2
                    if encloses(mid):
                        high = mid
                    else:
                        low = mid
                enclosed_at = high
        seeded = any(state != 'SOLID' for (c, _r), state in self._probes.items()
                     if c == tuple(np.round(centre, 9)) and isinstance(state, str)) \
            or enclosed_at is not None
        if enclosed_at is None:
            return None, None, seeded
        region = self.complete(self.probe(centre, enclosed_at), enclosed_at)
        return enclosed_at, region, True

    def flood_from(self, centre, order):
        """Complete flood (leaks and all) at the first level with a seed."""
        for r in order:
            space = self.level(r)
            seed = self.seed(space, centre)
            if seed[0] >= 0:
                return self.complete(space.to_grid(space.flood(seed, self.tick)), r)
        return None

    def complete(self, core, level, extra=()):
        """A space found with openings sealed, plus the narrow places off it.

        Sealing at ``level`` closes doors and windows - and with them every
        passage narrower than a door: a tunnel, a hallway, a gap between
        buildings. Growing the space back ``level - 1`` voxels only undoes
        the sealing next to walls, so such a passage used to end at its
        mouth. Instead the unsealed free space is flooded from the space,
        stopping only where a *different* sealed space begins - the outside,
        another room. Tunnels, closets and doorways come back; the next room
        and the outside stay out. ``extra`` adds flat voxel ids to seed from.

        The flood itself runs slightly sealed - passages narrower than about
        0.3 m are not places to put or aim cameras - which also keeps it out
        of the hollow inside of solids whose sampled surfaces have pinholes.
        """
        free = self.free1
        if level <= 1:
            return core & free
        # Grow this space and all the others back to the walls together, a
        # cube step at a time: each free voxel goes to whichever space reaches
        # it first. A doorway's far side belongs to the space beyond it - a
        # room's cameras stop at its door - and the corners where floor meets
        # wall belong to their own room, so no channel along them leaks out.
        mine = core & free
        theirs = self.level(level).free & ~core
        for _step in range(level - 1):
            left = free & ~mine & ~theirs
            grown = _dilate_cube(mine, 1) & left
            mine |= grown
            theirs |= _dilate_cube(theirs, 1) & left & ~grown
        narrow = min(level, max(2, int(math.ceil(0.15 / self.grid.voxel))))
        if level <= narrow and not len(extra):
            return mine
        space_n = self.level(narrow)
        space = _FreeSpace(space_n.free & ~theirs)
        seeds = np.concatenate([np.flatnonzero((mine & space_n.free).ravel()),
                                np.asarray(extra, dtype=np.int64)])
        filled = space.to_grid(space.flood(space.position(seeds), self.tick))
        return _grow_back(filled, narrow - 1, free & ~theirs) | mine

    def around(self, owner_mask, steps=1):
        """The space the given objects stand in.

        A statue in a room is orbited from inside that room, a lone object
        from all around it. Every separate space beside the objects' surface
        - found with door-sized openings sealed (``steps``) - gets a vote per
        surface voxel it borders, and only the clear winners count: a sofa
        that sinks an inch through the floor, or a cabinet that pokes through
        a wall, touches the space under the floor or behind the wall with a
        sliver of its surface and must not be photographed from there.
        """
        wanted = np.zeros(max(len(owner_mask), int(self.owner_ids.max(initial=0)) + 1), bool)
        wanted[:len(owner_mask)] = owner_mask
        surface = np.zeros(self.grid.shape, dtype=bool)
        surface.ravel()[self.sample_flat[wanted[self.owner_ids]]] = True
        # One vote per surface sample, cast for the free voxel just in front
        # of its face: votes follow surface area, a face buried in a wall
        # casts none, and no vote reaches through a floor slab.
        own = wanted[np.minimum(self.owner_ids, len(wanted) - 1)]
        cell, inside = self.grid.ijk(self.points[own] + self.normals[own] * (2.0 * self.grid.voxel))
        front = self.grid.flat(cell[inside]) if inside.any() else np.zeros(0, np.int64)
        beside = front[self.free1.ravel()[front]]
        for level in sorted({max(1, int(steps)), 1}, reverse=True):
            space = self.level(level)
            seeds = space.position(np.flatnonzero(_dilate(surface, level + 1) & space.free))
            seeds = seeds[seeds >= 0]
            if not len(seeds):
                continue
            core = self._winners(space, seeds, beside, level)
            if core is not None:
                return self.complete(core, level)
        return np.zeros(self.grid.shape, dtype=bool)

    def _winners(self, space, seeds, beside, level, floods=12):
        """The separate space(s) facing most of an object's surface.

        Each space found from ``seeds`` is grown back to the surface through
        free space only and scores the votes (flat voxel ids in ``beside``)
        it reaches. The best wins; another counts only when it is as good -
        an object standing in a doorway belongs to both rooms.
        """
        remaining = np.zeros(len(space.index), dtype=bool)
        remaining[seeds] = True
        parts = []
        for _ in range(floods):
            left = np.flatnonzero(remaining)
            if not len(left):
                break
            filled = space.flood(left[:1], self.tick)
            remaining &= ~filled
            grid = space.to_grid(filled)
            # Grown by cubes through free space: back to every surface the
            # sealed space faces, and never through a wall or slab.
            zone = grid
            for _step in range(level + 1):
                zone = _dilate_cube(zone, 1) & self.free1
            parts.append((int(np.count_nonzero(zone.ravel()[beside])), grid))
        if not parts:
            return None
        best = max(count for count, _grid in parts)
        if best == 0:
            return None
        chosen = np.zeros(self.grid.shape, dtype=bool)
        for count, grid in parts:
            if count >= 0.9 * best:
                chosen |= grid
        return chosen

    def mask_bounds(self, mask):
        cells = np.argwhere(mask[::2, ::2, ::2]) * 2
        if not len(cells):
            return self.lo, self.hi
        return self.grid.centres(cells.min(axis=0) - 1), self.grid.centres(cells.max(axis=0) + 2)

    def field(self):
        """Clearance field for ray marching, built once per context."""
        if self._field is None:
            field = _clearance_field(self.occ)
            field[self.occ] = 255
            self._field = field.ravel()
        return self._field


class _Region:
    """A space to capture: the flooded free voxels of one context."""

    def __init__(self, name, kind, ctx, mask, views=3, owners=None, frame=None):
        self.name, self.kind, self.ctx, self.mask = name, kind, ctx, mask
        self.views = views          # views per surface asked of this system
        self.owners = owners        # object systems: whose surfaces count
        self.frame = frame          # object systems: (centre, radius, fit)
        self.cap = 0                # this system's own camera limit, 0 = none
        self.stop = True            # may finish early once its surfaces are covered
        self.clearance = 0.0
        self.limit_height = False
        self.height_min, self.height_max = 0.0, 1e9

    def adopt(self, system):
        """Take a system's own placement settings."""
        self.views = system.views
        self.cap = system.max_cameras
        # An automatic system always stops when covered: that is what
        # automatic means. A counted one does only when asked to.
        self.stop = bool(system.stop_when_covered) or not system.max_cameras
        self.clearance = system.clearance
        if hasattr(system, "limit_height"):
            self.limit_height = system.limit_height
            self.height_min, self.height_max = system.height_min, system.height_max
        return self

    def contains(self, point):
        cell, inside = self.ctx.grid.ijk(np.asarray(point)[None])
        if not inside[0]:
            return False
        lo = np.maximum(cell[0] - 1, 0)
        hi = np.minimum(cell[0] + 2, self.ctx.grid.shape)
        return bool(self.mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any())

    def holds(self, points):
        """Which points lie in this space or on the surfaces bounding it."""
        if not hasattr(self, "_near"):
            self._near = _dilate(self.mask, 2)
        cell, inside = self.ctx.grid.ijk(points)
        out = np.zeros(len(points), dtype=bool)
        if inside.any():
            c = cell[inside]
            out[inside] = self._near[c[:, 0], c[:, 1], c[:, 2]]
        return out


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

def _few(array, count, rng):
    """A random subset of at most ``count`` rows, for the live preview."""
    if len(array) <= count:
        return np.asarray(array, dtype=np.float32)
    return np.asarray(array[rng.choice(len(array), count, replace=False)], dtype=np.float32)


def plan(request, progress=None, cancelled=None, visual=None):
    """Plan a rig. Returns a ``Result``; raises ``PlanError`` / ``Cancelled``.

    ``visual(kind, **data)`` optionally receives small snapshots of the work -
    surfaces, flooded space, candidates, each chosen camera - for a live
    viewport preview. They are subsampled copies, so watching costs nothing
    measurable, and the plan is identical with or without it.
    """
    report = _Reporter(progress, cancelled)
    rng = np.random.default_rng(request.seed)
    shown = np.random.default_rng(12345)    # never disturbs the planning stream
    show = visual or (lambda kind, **data: None)
    timings = {}
    result = Result()
    started = time.perf_counter()
    geometry = request.geometry
    preset = request.preset

    report(STAGE_SPACE, 0.0, "Measuring the scene")
    if geometry.is_empty:
        raise PlanError("No render-visible geometry was found to scan.")
    if not request.blobs and not request.objects:
        raise PlanError("Add a system: Interior, Exterior or Object/Collection.")

    # ---- the scene's roles and extents ---------------------------------------
    ground, backdrop = geometry.owner_roles()
    for owner in np.flatnonzero(backdrop):
        result.warnings.append(
            f"'{geometry.owner_names[owner]}' encloses the whole scene and was "
            "treated as a backdrop (ignored).")
    geo_lo, geo_hi = geometry.bounds(~backdrop)
    # The focus is what the scan is about: everything except a ground plane
    # or terrain, which would otherwise stretch the grid - and the cameras -
    # out over empty land.
    focus_lo, focus_hi = geometry.bounds(~backdrop & ~ground)
    if focus_lo is None:
        focus_lo, focus_hi = geo_lo, geo_hi
    big = float(np.max(focus_hi - focus_lo))
    seal = request.seal_size
    contexts = []

    def all_open(lo, hi, voxel):
        return [[True, True], [True, True], [True, True]]

    def all_closed(lo, hi, voxel):
        return [[False, False], [False, False], [False, False]]

    def context(lo, hi, open_faces, note, min_voxel=0.0):
        report(STAGE_SPACE, None, f"Voxelising {note}")
        ctx = _Context(geometry, lo, hi, voxels=preset['voxels'], samples=preset['samples'],
                       rng=rng, exclude=backdrop, open_faces=open_faces,
                       seal_size=seal, tick=lambda: report(STAGE_SPACE, None),
                       min_voxel=min_voxel)
        contexts.append(ctx)
        show('surface', points=_few(ctx.points, 40_000, shown))
        return ctx

    regions = []

    def found_region(region, seed):
        regions.append(region)
        cells = np.flatnonzero(region.mask)
        if len(cells) > 60_000:
            cells = shown.choice(cells, 60_000, replace=False)
        centres = region.ctx.grid.centres(np.stack(np.unravel_index(cells, region.mask.shape), 1))
        show('region', space=region.kind, name=region.name,
             seed=np.asarray(seed, np.float32), points=centres.astype(np.float32),
             voxel=region.ctx.grid.voxel)

    # ---- object / collection systems ------------------------------------------
    for group in request.objects:
        t_lo, t_hi = geometry.bounds(group.owners)
        if t_lo is None:
            result.warnings.append(f"{group.name} has no render-visible surfaces and was skipped.")
            continue
        radius = 0.5 * float(np.linalg.norm(t_hi - t_lo))
        tan_min = min(request.tan_h, request.tan_v)
        fit = radius / max(1e-6, math.sin(math.atan(tan_min))) * 1.08
        centre = 0.5 * (t_lo + t_hi)
        ctx = context(t_lo - fit, t_hi + fit, all_open, group.name)
        mask = ctx.around(group.owners, ctx.steps_for(seal))
        if not mask.any():
            result.warnings.append(f"{group.name} has no open space around it and was skipped.")
            continue
        found_region(_Region(group.name, OBJECT, ctx, mask, owners=group.owners,
                             frame=(centre, radius, fit)).adopt(group), centre)

    # ---- interior and exterior systems ----------------------------------------
    if request.blobs:
        world = []

        def whole_scene():
            """The scene-wide context, padded so exteriors have room."""
            if not world:
                extent = focus_hi - focus_lo
                centres = np.array([b.center for b in request.blobs if not b.bounded]
                                   or [0.5 * (focus_lo + focus_hi)])
                lo = np.minimum(focus_lo, centres.min(axis=0))
                hi = np.maximum(focus_hi, centres.max(axis=0))
                margin = 0.3 * big
                top = 0.3 * max(float(extent[2]), 0.25 * big)
                below = margin if any(b.allow_below for b in request.blobs) else 0.02 * big
                lo = lo - np.array([margin, margin, below])
                hi = hi + np.array([margin, margin, top])

                # A face is open world when the grid reaches past the scanned
                # structures on that side; a space touching one is outside.
                def faces(lo, hi, voxel):
                    return [[bool(lo[a] < focus_lo[a] - voxel), bool(hi[a] > focus_hi[a] + voxel)]
                            for a in range(3)]
                world.append(context(lo, hi, faces, "the scene"))
            return world[0]

        def solid(blob):
            result.warnings.append(f"{blob.name} sits inside solid geometry and was "
                                   "skipped - move it into open space.")

        structures = np.flatnonzero(~ground & ~backdrop & np.isfinite(geometry.owner_lo[:, 0]))

        def nearby_box(c):
            """Box around the structures an exterior blob stands next to.

            A blob outside a house means the space around that house, not
            every street of the city it sits in: take the structures within
            reach of the nearest one, plus room for cameras around them.
            Returns None when that is most of the scene anyway.
            """
            if not len(structures):
                return None
            lo_o, hi_o = geometry.owner_lo[structures], geometry.owner_hi[structures]
            gap = np.linalg.norm(np.maximum(0.0, np.maximum(lo_o - c, c - hi_o)), axis=1)
            # Only what is reasonably close can belong to "this building";
            # keeps the clustering small in a huge scene.
            local = np.flatnonzero(gap <= float(gap.min()) + 0.25 * big)
            label, linker = _cluster_boxes(lo_o[local], hi_o[local], tol=max(0.05, 0.004 * big))
            # The building, not the road or floor slab the blob floats over.
            candidates = np.flatnonzero(linker) if linker.any() else np.arange(len(local))
            # Size counts as well as distance: a blob between a house and a
            # small statue means the house, even when the statue is nearer.
            best, nearest = np.inf, label[candidates[0]]
            for cluster in np.unique(label[candidates]):
                inside = local[label == cluster]
                extent = float(np.linalg.norm(hi_o[inside].max(axis=0) - lo_o[inside].min(axis=0)))
                score = float(gap[inside].min()) / max(extent, 1e-3)
                if score < best:
                    best, nearest = score, cluster
            members = local[label == nearest]
            c_lo, c_hi = lo_o[members].min(axis=0), hi_o[members].max(axis=0)
            size = float(np.linalg.norm(c_hi - c_lo))
            # Neighbours right next to that building are part of its setting.
            chosen = gap <= float(gap[members].min()) + max(0.35 * size,
                                                            float(np.min(gap)) * 1.5)
            chosen[members] = True
            lo = np.minimum(lo_o[chosen].min(axis=0), c)
            hi = np.maximum(hi_o[chosen].max(axis=0), c)
            extent = hi - lo
            margin = 0.45 * float(np.max(extent))
            lo = lo - np.array([margin, margin, 0.02 * float(np.max(extent))])
            hi = hi + np.array([margin, margin, max(margin * 0.6, 0.3 * float(extent[2]))])
            g = whole_scene()
            if np.prod(hi - lo) >= 0.5 * np.prod(g.hi - g.lo):
                return None
            return lo, hi

        def refine(blob, region, g, seal, coarse):
            """The room again, in a box of its own at a finer voxel size.

            The scene-wide grid can be too coarse to see a door, a tunnel
            or free space among furniture. When a box around the room is
            materially finer, the room is found again there; a narrow
            passage that runs out of the box widens that side of it.
            """
            c = blob.center
            lo_w, hi_w = g.mask_bounds(region)
            pad = max(seal, 0.25 * float(np.max(hi_w - lo_w)))
            lo_b, hi_b = np.maximum(lo_w - pad, g.lo), np.minimum(hi_w + pad, g.hi)
            found = None
            for _attempt in range(3):
                voxel = max((float(np.prod(hi_b - lo_b)) / preset['voxels']) ** (1.0 / 3.0),
                            ROOM_MIN_VOXEL)
                if not coarse and voxel > 0.7 * g.grid.voxel:
                    break
                local = context(lo_b, hi_b, all_open, blob.name, min_voxel=ROOM_MIN_VOXEL)
                e2, r2, _s = local.enclose(c, local.steps_for(seal))
                if e2 is None:
                    # The finer grid sees the openings as wider than the setting.
                    e2, r2, _s = local.enclose(c, local.widest_steps(seal))
                    if e2 is None:
                        break
                    seal = max(seal, 2 * e2 * local.grid.voxel)
                    result.warnings.append(
                        f"{blob.name} has openings wider than Doorway Width; openings "
                        f"up to {seal:.1f} m were treated as doors.")
                found = _Region(blob.name, INTERIOR, local, r2)
                grow = False
                for axis in range(3):
                    for side, index in ((0, 0), (1, -1)):
                        sl = [slice(None)] * 3
                        sl[axis] = index
                        at_scene = (lo_b[axis] <= g.lo[axis] + g.grid.voxel if side == 0
                                    else hi_b[axis] >= g.hi[axis] - g.grid.voxel)
                        if r2[tuple(sl)].any() and not at_scene:
                            # Generously: every retry is a new grid.
                            grow = True
                            if side == 0:
                                lo_b[axis] = max(g.lo[axis], lo_b[axis] - 2.0 * pad)
                            else:
                                hi_b[axis] = min(g.hi[axis], hi_b[axis] + 2.0 * pad)
                if not grow:
                    break
            return found

        def resolve(blob):
            c = blob.center
            seal = blob.seal
            if blob.bounded:
                pad = 0.05 * blob.radius
                ctx = context(c - blob.radius - pad, c + blob.radius + pad, all_closed, blob.name)
                _level, region, _seeded = ctx.enclose(c, ctx.steps_for(seal))
                if region is None:
                    solid(blob)
                    return None
                region &= _sphere_mask(ctx.grid, c, blob.radius)
                kind = EXTERIOR if blob.mode == EXTERIOR else INTERIOR
                if kind == EXTERIOR and not blob.allow_below:
                    region[:, :, :ctx.z_floor] = False
                return _Region(blob.name, kind, ctx, region)
            g = whole_scene()
            enclosed_at, region, seeded = g.enclose(c, g.steps_for(seal))
            # The scene-wide grid can be too coarse to see a door, or to find
            # free space among furniture. Then the blob is looked at again in
            # a box of its own, grown until its space is enclosed or the box
            # reaches the scene - interiors get the resolution they need.
            coarse = seal > 0 and g.grid.voxel > seal / 3.0
            if blob.mode != EXTERIOR:
                if enclosed_at is None and blob.mode == INTERIOR and seeded:
                    # Openings wider than the doorway setting: seal as wide
                    # as the grid allows rather than give up on the room.
                    widest_at, wider, _s = g.enclose(c, g.widest_steps(seal))
                    if wider is not None:
                        enclosed_at, region = widest_at, wider
                        result.warnings.append(
                            f"{blob.name} has openings wider than Doorway Width; openings "
                            f"up to {2 * widest_at * g.grid.voxel:.1f} m were treated as doors.")
                        seal = max(seal, 2 * widest_at * g.grid.voxel)
                if enclosed_at is not None:
                    local = refine(blob, region, g, seal, coarse)
                    if local is not None:
                        return local
                    return _Region(blob.name, INTERIOR, g, region)
                if coarse or not seeded:
                    half = max(4.0 * blob.radius, 0.06 * big, 3.0 * seal)
                    g_lo, g_hi = g.lo, g.hi
                    # Past this size a box's voxels are too coarse to resolve
                    # a door, so growing further cannot find a room.
                    widest = (seal / 3.0) * preset['voxels'] ** (1.0 / 3.0) if seal > 0 else big
                    while True:
                        lo_b, hi_b = np.maximum(c - half, g_lo), np.minimum(c + half, g_hi)
                        if (np.all(hi_b - lo_b >= 0.6 * (g_hi - g_lo))
                                or float(np.max(hi_b - lo_b)) > 1.5 * widest):
                            break

                        def faces(lo, hi, voxel, lo_b=lo_b, hi_b=hi_b):
                            return [[bool(lo_b[a] > g_lo[a] + voxel) or g.open_faces[a][0],
                                     bool(hi_b[a] < g_hi[a] - voxel) or g.open_faces[a][1]]
                                    for a in range(3)]
                        local = context(lo_b, hi_b, faces, blob.name)
                        e2, r2, _s = local.enclose(c, local.steps_for(seal))
                        if e2 is not None:
                            return _Region(blob.name, INTERIOR, local, r2)
                        half *= 3.0
                if blob.mode == INTERIOR:
                    region = g.flood_from(c, range(1, g.steps_for(seal) + 1))
                    if region is None:
                        solid(blob)
                        return None
                    region &= _sphere_mask(g.grid, c, blob.radius)
                    result.warnings.append(
                        f"{blob.name} is open to the outside (no ceiling or a large "
                        "opening); its radius was used as the room's bounds.")
                    return _Region(blob.name, INTERIOR, g, region)
            elif enclosed_at is not None:
                result.warnings.append(f"{blob.name} is enclosed, so it scans that enclosed space.")
                return _Region(blob.name, EXTERIOR, g, region)
            if not seeded:
                solid(blob)
                return None
            near = nearby_box(c) if blob.reach == 'NEARBY' else None
            if near is not None:
                # The outside of the buildings around the blob, in a box of
                # its own at the resolution that box affords.
                local = context(near[0], near[1], all_open, blob.name)
                region = local.flood_from(c, range(local.steps_for(seal), 0, -1))
                if region is not None and region.any():
                    if not blob.allow_below:
                        region[:, :, :local.z_floor] = False
                    return _Region(blob.name, EXTERIOR, local, region)
            # The outside, with door-sized openings sealed so rooms that are
            # only reachable through them stay out of an exterior scan.
            region = g.flood_from(c, range(g.steps_for(seal), 0, -1))
            if region is None:
                solid(blob)
                return None
            if not blob.allow_below:
                region[:, :, :g.z_floor] = False
            return _Region(blob.name, EXTERIOR, g, region)

        total = len(request.blobs)
        for number, blob in enumerate(request.blobs):
            report(STAGE_SPACE, 0.05 + 0.8 * number / total, f"Analysing {blob.name}")
            # A second blob in a space already found adds nothing new.
            owner = next((r for r in regions
                          if r.kind != OBJECT and r.contains(blob.center)), None)
            if owner is not None:
                owner.name += " + " + blob.name
                owner.views = max(owner.views, blob.views)
                owner.cap = owner.cap + blob.max_cameras if owner.cap and blob.max_cameras else 0
                owner.stop = owner.stop and (blob.stop_when_covered or not blob.max_cameras)
                continue
            found = resolve(blob)
            if found is None:
                continue
            if not found.mask.any():
                result.warnings.append(f"{blob.name} found no open space to scan.")
                continue
            found_region(found.adopt(blob), blob.center)
    if not regions:
        raise PlanError("No open space to scan was found for any system.")
    for region in regions:
        result.regions.append({"name": region.name, "kind": region.kind,
                               "voxels": int(np.count_nonzero(region.mask)),
                               "voxel": region.ctx.grid.voxel})
    timings['free_space'] = time.perf_counter() - started

    # ---- targets: surfaces bordering the scanned space ----------------------
    # Every region gets its own share of targets and of the objective: a
    # blob in a small room must not be outvoted by a large exterior just
    # because the exterior has more square metres.
    report(STAGE_SPACE, 0.9, "Choosing surfaces to cover")
    share = preset['targets'] // len(regions) + 1
    interiors = [region for region in regions if region.kind == INTERIOR]
    parts = {k: [] for k in ("p", "n", "w", "o", "r", "k")}
    claimed = {}
    for index, region in enumerate(regions):
        ctx = region.ctx
        taken = claimed.setdefault(id(ctx), np.zeros(len(ctx.points), dtype=bool))
        mine = _dilate(region.mask, 2).ravel()[ctx.sample_flat] & ~taken
        if region.owners is not None:
            # An object system covers only its own objects' surfaces - and
            # covers them again even where a room system already does, which
            # is exactly how it earns them more cameras.
            wanted = np.zeros(len(geometry.owner_names) + 1, bool)
            wanted[:len(region.owners)] = region.owners
            mine = _dilate(region.mask, 2).ravel()[ctx.sample_flat]
            mine &= wanted[np.minimum(ctx.owner_ids, len(wanted) - 1)]
        taken |= mine
        idx = np.flatnonzero(mine)
        if region.kind == EXTERIOR and len(idx):
            # A room scanned by its own blob belongs to that blob, even when
            # an unglazed opening lets the outside flood reach into it.
            for other in interiors:
                idx = idx[~other.holds(ctx.points[idx])]
        if not len(idx):
            continue
        picked, w = _thin(ctx.points[idx], share, rng)
        sel = idx[picked]
        parts["p"].append(ctx.points[sel]); parts["n"].append(ctx.normals[sel])
        parts["w"].append(w / w.sum()); parts["o"].append(ctx.owner_ids[sel])
        parts["r"].append(np.full(len(sel), index))
        parts["k"].append(np.full(len(sel), float(region.views)))
    if not parts["p"]:
        raise PlanError("No surfaces border the scanned space.")
    targets = np.concatenate(parts["p"]).astype(np.float64)
    target_normals = np.concatenate(parts["n"]).astype(np.float64)
    weights = np.concatenate(parts["w"])
    target_owner = np.concatenate(parts["o"])
    target_region = np.concatenate(parts["r"])
    target_k = np.concatenate(parts["k"])
    if ground.any():
        # Ground still needs coverage, but it should not outvote buildings
        # and objects just because it has the most square metres.
        on_ground = np.zeros(len(ground) + 1, bool)
        on_ground[:len(ground)] = ground
        weights = weights * np.where(on_ground[np.minimum(target_owner, len(ground))], 0.35, 1.0)
    weights = weights / weights.sum()
    T = len(targets)

    # ---- per-region capture distances ---------------------------------------
    params = []
    for index, region in enumerate(regions):
        voxel = region.ctx.grid.voxel
        count = int(np.count_nonzero(region.mask))
        span = float(count * voxel ** 3) ** (1.0 / 3.0)
        r_lo, r_hi = region.ctx.mask_bounds(region.mask)
        diag = float(np.linalg.norm(r_hi - r_lo))
        if region.kind == OBJECT:
            _centre, radius, fit = region.frame
            d_ref = max((fit - radius) * 0.85, 3 * voxel)
            band = (0.0, fit)
            view = fit + 2.0 * radius
        elif region.kind == INTERIOR:
            d_ref = max(0.8 * span, 4 * voxel)
            band = (0.0, np.inf)
            view = min(max(diag, 2.5 * d_ref), 4.0 * d_ref)
        else:
            # Exterior distances follow the size of what this region looks
            # at - the facades around it - not the whole padded domain.
            seen_here = targets[target_region == index]
            if not len(seen_here):
                seen_here = targets
            t_ext = float(np.linalg.norm(np.ptp(seen_here, axis=0)))
            d_ref = max(0.2 * t_ext, 5 * voxel)
            band = (0.0, 1.8 * d_ref)
            view = 3.0 * d_ref
        params.append(dict(kind=region.kind, d_ref=d_ref, band=band, view=view,
                           near=0.3 * d_ref, voxel=voxel))

    # ---- candidate viewpoints -------------------------------------------------
    # Each region spreads its own share of viewpoints: by volume alone, the
    # outside of a house would take nearly every candidate from its rooms.
    report(STAGE_SPACE, 0.95, "Spreading candidate viewpoints")
    per_region = max(8, preset['candidates'] // len(regions))
    parts, part_regions, part_spacing, part_tight = [], [], [], []
    for r, region in enumerate(regions):
        ctx, voxel = region.ctx, region.ctx.grid.voxel
        clearance = region.clearance
        if region.kind == OBJECT and clearance > 0:
            # An orbit must keep room between the object and the camera distance.
            clearance = min(clearance, 0.35 * region.frame[2])
        if clearance > 0:
            clear_steps = max(0, int(math.ceil(clearance / voxel)) - 1)
        else:
            clear_steps = max(1, int(round(params[r]['d_ref'] * 0.12 / voxel)))
        eligible = region.mask & ~_dilate(ctx.barriers[1], clear_steps)
        roomy = eligible.copy()           # away from surfaces: not a tight passage
        eligible |= region.mask & _middle_of_passages(ctx, voxel)
        if not eligible.any():
            eligible = region.mask & ~_dilate(ctx.barriers[1], 1)
        if not eligible.any():
            eligible = region.mask
        cells = np.argwhere(eligible)
        if region.limit_height and region.kind != OBJECT:
            # Height above the floor of each column of the scanned space, so a
            # multi-storey building gets eye-level views on every floor.
            floor = np.full(ctx.grid.shape[:2], ctx.grid.shape[2], dtype=np.int64)
            region_cells = np.argwhere(region.mask)
            np.minimum.at(floor, (region_cells[:, 0], region_cells[:, 1]), region_cells[:, 2])
            # The floor is the lowest free space in the surroundings, not
            # under each column: over a table, "eye level" still means eye
            # level above the room's floor, not above the tabletop.
            reach = int(min(40, max(2, math.ceil(1.5 / voxel))))
            for axis in (0, 1):
                spread_floor = floor.copy()
                for step in range(1, reach + 1):
                    lead = [slice(None)] * 2
                    trail = [slice(None)] * 2
                    lead[axis], trail[axis] = slice(step, None), slice(None, -step)
                    np.minimum(spread_floor[tuple(lead)], floor[tuple(trail)],
                               out=spread_floor[tuple(lead)])
                    np.minimum(spread_floor[tuple(trail)], floor[tuple(lead)],
                               out=spread_floor[tuple(trail)])
                floor = spread_floor
            # The first free voxel sits 1.5 voxels above the floor surface.
            height = (cells[:, 2] - floor[cells[:, 0], cells[:, 1]] + 1.5) * voxel
            keep = (height >= region.height_min) & (height <= region.height_max)
            if keep.any():
                cells = cells[keep]
            else:
                result.warnings.append(f"{region.name}: no open space in the height "
                                       "range; using all heights.")
        if not len(cells):
            continue
        volume_r = len(cells) * voxel ** 3
        if len(cells) > 400_000:
            cells = cells[rng.choice(len(cells), 400_000, replace=False)]
        pos = ctx.grid.centres(cells) + (rng.random((len(cells), 3)) - 0.5) * voxel * 0.8
        pick, _w = _thin(pos, per_region * 3, rng, dimension=3.0)
        pos = pos[pick]
        if region.kind == EXTERIOR:
            for other in interiors:
                if len(pos):
                    pos = pos[~other.holds(pos)]
            if not len(pos):
                continue
        own_targets = targets[target_region == r]
        ok = np.ones(len(pos), dtype=bool)
        if len(own_targets):
            nearest = _nearest_distance(pos, own_targets)
            ok = (nearest >= params[r]['band'][0]) & (nearest <= params[r]['band'][1])
            if ok.any():
                pos = pos[ok]
        pick, _w = _thin(pos, per_region, rng, dimension=3.0)
        pos = pos[pick]
        parts.append(pos)
        part_regions.append(np.full(len(pos), r))
        cell, inside = ctx.grid.ijk(pos)
        tight = np.zeros(len(pos), dtype=bool)
        tight[inside] = ~roomy.ravel()[ctx.grid.flat(cell[inside])]
        part_tight.append(tight)
        part_spacing.append(np.full(len(pos), (volume_r * ok.mean() / max(1, len(pos)))
                                    ** (1.0 / 3.0)))
    if not parts:
        raise PlanError("There is no room for cameras in the scanned space.")
    positions = np.concatenate(parts)
    region_of = np.concatenate(part_regions)
    spacing_of = np.concatenate(part_spacing)
    tight_of = np.concatenate(part_tight)
    C = len(positions)
    timings['candidates'] = time.perf_counter() - started
    show('targets', points=_few(targets, 30_000, shown))
    show('candidates', points=positions.astype(np.float32))

    # ---- visibility per candidate --------------------------------------------
    # Each viewpoint is tested in its own region's grid, against the targets
    # of every region that shares that grid.
    res = preset['cube']
    cube_dirs = _cube_directions(res)
    pix_angle = (math.pi / 2.0) / res
    cos_div = math.cos(math.radians(preset['diversity_deg']))
    tan_h, tan_v = request.tan_h * 0.95, request.tan_v * 0.95
    yaw_bins, pitch_bins = 72, 36
    yaw_win = max(1, int(round(math.degrees(2 * math.atan(tan_h)) / 5.0)))
    pitch_win = max(1, int(round(math.degrees(2 * math.atan(tan_v)) / 5.0)))
    cos_grazing, cos_good = math.cos(math.radians(84.0)), math.cos(math.radians(35.0))

    vis_idx, vis_u, vis_q = [None] * C, [None] * C, [None] * C
    views = []   # (candidate, selection-into-candidate, quality, forward)
    static = []  # each view's gain before anything is chosen
    batch = 16
    ctx_index = {id(ctx): i for i, ctx in enumerate(contexts)}
    region_ctx = np.array([ctx_index[id(region.ctx)] for region in regions])
    done = 0
    for ctx in contexts:
        cand = np.flatnonzero(region_ctx[region_of] == ctx_index[id(ctx)])
        if not len(cand):
            continue
        tsub = np.flatnonzero(region_ctx[target_region] == ctx_index[id(ctx)])
        field_flat = ctx.field()
        voxel = ctx.grid.voxel
        local_targets = targets[tsub]
        for start in range(0, len(cand), batch):
            report(STAGE_VISIBILITY, done / C, f"Viewpoint {done + 1:,} of {C:,}")
            chunk = cand[start:start + batch]
            done += len(chunk)
            max_view = max(params[r]['view'] for r in region_of[chunk])
            depth = _depth_maps(field_flat, ctx.grid, positions[chunk], cube_dirs, max_view)
            # Pass 1: the cube map settles most targets at once. A target at
            # or in front of its pixel's first hit is visible; one far behind
            # it is hidden. Only the thin band in between - a pixel straddling
            # a silhouette, or a surface just behind a thin one - is uncertain.
            pending = []
            rays_o, rays_d, rays_l = [], [], []
            for local, c in enumerate(chunk):
                p = params[region_of[c]]
                diff = local_targets - positions[c]
                dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
                close = np.flatnonzero((dist > 1e-6) & (dist <= p['view']))
                d = dist[close]
                u = diff[close] / np.maximum(d, 1e-12)[:, None]
                first = depth[local, _cube_pixel(u, res)] if len(u) else np.zeros(0)
                sure = d <= first + 1.5 * voxel
                maybe = np.flatnonzero(~sure & (d <= first + 1.5 * voxel + d * pix_angle * 1.5))
                pending.append((c, tsub[close], d, u, sure, maybe))
                if len(maybe):
                    rays_o.append(np.repeat(positions[c][None], len(maybe), axis=0))
                    rays_d.append(u[maybe])
                    rays_l.append(d[maybe] - 1.8 * voxel)
            # Pass 2: march the uncertain ones exactly, target by target.
            if rays_o:
                hits = _march(field_flat, ctx.grid, np.concatenate(rays_o),
                              np.concatenate(rays_d), np.concatenate(rays_l))
                clear = ~np.isfinite(hits)
                offset = 0
                for _c, _near, _d, _u, sure, maybe in pending:
                    if len(maybe):
                        sure[maybe] = clear[offset:offset + len(maybe)]
                        offset += len(maybe)
            for c, near_ok, d, u, seen, _maybe in pending:
                p = params[region_of[c]]
                if not len(near_ok):
                    vis_idx[c] = np.zeros(0, np.int32); vis_u[c] = np.zeros((0, 3))
                    vis_q[c] = np.zeros(0); continue
                near_ok, d, u = near_ok[seen], d[seen], u[seen]
                cos_i = np.abs(np.einsum('ij,ij->i', u, target_normals[near_ok]))
                q_inc = np.clip((cos_i - cos_grazing) / (cos_good - cos_grazing), 0.0, 1.0)
                q_dist = np.where(d <= p['d_ref'], 1.0, p['d_ref'] / np.maximum(d, 1e-9))
                q_dist *= np.clip(d / p['near'], 0.25, 1.0)
                q = q_inc * q_dist
                keep = q > 0.02
                near_ok, u, q = near_ok[keep], u[keep], q[keep]
                vis_idx[c] = near_ok.astype(np.int32); vis_u[c] = u; vis_q[c] = q
                if not len(near_ok):
                    continue
                # Propose headings: a yaw/pitch histogram of what is visible,
                # box-filtered to the field of view, then non-maximum suppressed.
                w = q * weights[near_ok]
                yaw = (np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360.0) / 5.0
                pitch = (np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1))) + 90.0) / 5.0
                yi = np.clip(yaw.astype(np.int64), 0, yaw_bins - 1)
                pi = np.clip(pitch.astype(np.int64), 0, pitch_bins - 1)
                hist = np.bincount(yi * pitch_bins + pi, weights=w,
                                   minlength=yaw_bins * pitch_bins).reshape(yaw_bins, pitch_bins)
                score = _box_sum(hist, yaw_win, pitch_win)
                headings = []
                for _ in range(preset['views_per_candidate']):
                    flat = int(np.argmax(score))
                    best = score.flat[flat]
                    if best <= 0 or (headings and best < 0.06 * headings[0][0]):
                        break
                    y0, p0 = divmod(flat, pitch_bins)
                    headings.append((best, y0, p0))
                    ys = (np.arange(-(yaw_win * 6 // 10), yaw_win * 6 // 10 + 1) + y0) % yaw_bins
                    p_lo = max(0, p0 - pitch_win * 6 // 10)
                    p_hi = min(pitch_bins, p0 + pitch_win * 6 // 10 + 1)
                    score[np.ix_(ys, np.arange(p_lo, p_hi))] = 0.0
                forwards = []
                for _best, y0, p0 in headings:
                    yaw_r = math.radians((y0 + 0.5) * 5.0)
                    pitch_r = math.radians(np.clip((p0 + 0.5) * 5.0 - 90.0, -80.0, 80.0))
                    forwards.append((math.cos(pitch_r) * math.cos(yaw_r),
                                     math.cos(pitch_r) * math.sin(yaw_r), math.sin(pitch_r)))
                centroid = (u * w[:, None]).sum(axis=0)
                if np.linalg.norm(centroid) > 0.35 * w.sum():
                    forwards.append(tuple(centroid / np.linalg.norm(centroid)))
                if regions[region_of[c]].kind == OBJECT:
                    look = regions[region_of[c]].frame[0] - positions[c]
                    if np.linalg.norm(look) > 1e-6:
                        forwards.append(tuple(look / np.linalg.norm(look)))
                own_views = []
                for forward in forwards:
                    f, right, up = _basis(forward)
                    z = u @ f
                    x = u @ right
                    y = u @ up
                    zs = np.maximum(z, 1e-9)
                    inside = (z > 0.05) & (np.abs(x) <= tan_h * zs) & (np.abs(y) <= tan_v * zs)
                    if not inside.any():
                        continue
                    sel = np.flatnonzero(inside)
                    rx = x[sel] / (tan_h * zs[sel])
                    ry = y[sel] / (tan_v * zs[sel])
                    frame = 1.0 - 0.3 * (rx * rx + ry * ry) * 0.5
                    qv = (q[sel] * frame).astype(np.float32)
                    own_views.append((float(np.dot(weights[near_ok[sel]], qv)),
                                      (c, sel.astype(np.int32), qv, f)))
                # A heading far weaker than this viewpoint's best is almost never
                # chosen; dropping it keeps the greedy search short.
                if own_views:
                    best = max(score for score, _view in own_views)
                    views.extend(view for score, view in own_views if score >= 0.12 * best)
                    static.extend(score for score, _view in own_views if score >= 0.12 * best)
    timings['visibility'] = time.perf_counter() - started
    if not views:
        raise PlanError("No viewpoint can see the surfaces to scan.")
    static = np.asarray(static)
    strong = np.flatnonzero(static >= 0.01 * static.max())
    views = [views[i] for i in strong]
    static = static[strong]

    # ---- lazy-greedy submodular selection -------------------------------------
    report(STAGE_SELECTION, 0.0, f"Choosing from {len(views):,} candidate views")
    observable = np.zeros(T, dtype=bool)
    for c, sel, _q, _f in views:
        observable[vis_idx[c][sel]] = True
    # Each surface's target view count comes from the system it belongs to.
    K = target_k
    k_slots = max(3, int(K.max()))
    coverage = np.zeros(T)
    stored = np.zeros((T, k_slots, 3))
    stored_count = np.zeros(T, dtype=np.int64)
    chosen_positions = []
    chosen = []

    def gain(v, apply=False):
        c, sel, qv, _f = views[v]
        idx = vis_idx[c][sel]
        u = vis_u[c][sel]
        dots = np.einsum('tkj,tj->tk', stored[idx], u).max(axis=1)
        novelty = np.clip((1.0 - dots) / (1.0 - cos_div), 0.0, 1.0)
        add = qv * novelty
        value = float(np.sum(weights[idx] * np.minimum(add, np.maximum(0.0, K[idx] - coverage[idx]))
                             / K[idx]))
        if apply:
            coverage[idx] += add
            slot = stored_count[idx] % k_slots
            stored[idx, slot] = u
            stored_count[idx] += 1
        return value

    def spread(v):
        if not chosen_positions:
            return 1.0
        gap = float(np.min(np.linalg.norm(np.asarray(chosen_positions)
                                          - positions[views[v][0]], axis=1)))
        return 0.55 + 0.45 * min(1.0, gap / max(spacing_of[views[v][0]], 1e-9))

    # Before anything is chosen every direction is novel and nothing is
    # saturated, so the first gains reduce to a weighted sum of qualities.
    heap = [(-float(np.dot(weights[vis_idx[c][sel]] / K[vis_idx[c][sel]], qv)), v)
            for v, (c, sel, qv, _f) in enumerate(views)]
    heapq.heapify(heap)
    reachable = float(weights[observable].sum()) or 1.0
    budget = request.max_cameras
    R = len(regions)
    caps = np.array([region.cap for region in regions])
    stops = np.array([region.stop for region in regions])
    used = np.zeros(R, dtype=np.int64)
    finished = np.zeros(R, dtype=bool)     # a system that needs no more cameras
    view_region = np.array([region_of[view[0]] for view in views])
    # Each system judges "covered" by its own surfaces and its own best view.
    first = np.zeros(R)
    initial = np.zeros(len(views))
    for neg, v in heap:
        initial[v] = -neg
    np.maximum.at(first, view_region, initial)
    own_weight = np.array([float(weights[(target_region == r) & observable].sum()) or 1.0
                           for r in range(R)])
    if caps.sum() > budget:
        result.warnings.append(
            f"The systems ask for {int(caps.sum())} cameras but Max Cameras (total) is "
            f"{budget}; raise it in the calculation settings to give every system its count.")

    def owed():
        # Budget still owed to systems that asked for a number of cameras.
        return int(np.maximum(caps - used, 0)[(caps > 0) & ~finished].sum())

    def covered(r):
        mine = target_region == r
        return float(weights[mine & (coverage >= 0.8 * K)].sum()) / own_weight[r]

    while heap and len(chosen) < budget and not finished.all():
        if len(chosen) % 4 == 0:
            report(STAGE_SELECTION, len(chosen) / budget,
                   f"{len(chosen)} cameras placed")
        _neg, v = heapq.heappop(heap)
        owner = view_region[v]
        if finished[owner]:
            continue
        if caps[owner] and used[owner] >= caps[owner]:
            finished[owner] = True      # it has the cameras it asked for
            continue
        if not caps[owner] and len(chosen) >= budget - owed():
            finished[owner] = True      # what is left belongs to counted systems
            continue
        value = gain(v) * spread(v)
        if heap and value < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-value, v))
            continue
        if value <= 1e-9:
            break
        if stops[owner] and value < 0.015 * first[owner] and used[owner] >= 2:
            if covered(owner) >= 0.97 or value < 0.004 * first[owner]:
                finished[owner] = True
                continue
        gain(v, apply=True)
        chosen.append(v)
        used[owner] += 1
        show('camera', position=positions[views[v][0]].astype(np.float32),
             forward=np.asarray(views[v][3], np.float32),
             up=_basis(views[v][3])[2].astype(np.float32))
        chosen_positions.append(positions[views[v][0]])
    # A system that asked for a number of cameras - and not to stop when
    # covered - gets topped up with its own best remaining views, within the
    # total limit.
    taken = set(chosen)
    for owner in np.flatnonzero((caps > 0) & ~stops):
        pool = [(-gain(v) * spread(v), v) for v in np.flatnonzero(view_region == owner)
                if v not in taken]
        heapq.heapify(pool)
        while pool and used[owner] < caps[owner] and len(chosen) < budget:
            _neg, v = heapq.heappop(pool)
            value = gain(v) * spread(v)
            if pool and value < -pool[0][0] - 1e-12:
                heapq.heappush(pool, (-value, v))
                continue
            if value <= 0.0:
                break
            gain(v, apply=True)
            chosen.append(v)
            taken.add(v)
            used[owner] += 1
            show('camera', position=positions[views[v][0]].astype(np.float32),
                 forward=np.asarray(views[v][3], np.float32),
                 up=_basis(views[v][3])[2].astype(np.float32))
            chosen_positions.append(positions[views[v][0]])
        if used[owner] < caps[owner]:
            why = ("Max Cameras (total) reached" if len(chosen) >= budget
                   else "no other viewpoint adds anything new")
            result.warnings.append(
                f"{regions[owner].name}: {used[owner]} of {caps[owner]} cameras - {why}.")
    timings['selection'] = time.perf_counter() - started
    for index, entry in enumerate(result.regions):
        entry["cameras"] = int(used[index])
        entry["asked"] = int(caps[index])
        entry["covered"] = covered(index)
        # Tunnels and corridors are seen a short stretch at a time and can
        # take many cameras; say so when an automatic system spent a lot there.
        tight = int(sum(1 for v in chosen if view_region[v] == index and tight_of[views[v][0]]))
        entry["tight"] = tight
        if not caps[index] and tight >= max(20, 0.3 * used[index]):
            result.warnings.append(
                f"{regions[index].name}: {tight} of {int(used[index])} cameras went into tight "
                "passages. Give the system a Cameras count to limit it.")
    if not chosen:
        raise PlanError("No useful camera could be placed.")

    # ---- order along a short path, so stepping through them is natural -------
    order = _path_order(np.asarray(chosen_positions))
    chosen = [chosen[i] for i in order]
    result.positions = np.array([positions[views[v][0]] for v in chosen])
    result.forwards = np.array([views[v][3] for v in chosen])
    result.ups = np.array([_basis(f)[2] for f in result.forwards])

    full = float(weights[coverage >= K * 0.8].sum())
    seen = float(weights[coverage > 0.05].sum())
    used = [ctx for ctx in contexts if any(region.ctx is ctx for region in regions)]
    result.stats = dict(
        cameras=len(chosen), targets=T, candidates=C, views=len(views),
        voxel=min(ctx.grid.voxel for ctx in used),
        voxels=[round(ctx.grid.voxel, 4) for ctx in used],
        grids=[ctx.grid.shape for ctx in used], contexts=len(contexts),
        samples=sum(len(ctx.points) for ctx in contexts),
        covered=full / reachable, seen=seen / reachable,
        hidden=1.0 - reachable, mean_views=float(np.sum(weights * np.minimum(coverage, K + 2))
                                                 / max(1e-12, weights.sum())),
        seconds=time.perf_counter() - started, timings=timings,
    )
    result.target_points = targets.astype(np.float32)
    result.target_coverage = coverage.astype(np.float32)
    show('coverage', points=result.target_points,
         level=np.clip(coverage / K, 0.0, 1.0).astype(np.float32))
    exterior = [region.ctx.grid.voxel for region in regions if region.kind == EXTERIOR]
    if exterior and max(exterior) > 0.02 * big:
        result.warnings.append(
            f"The exterior is large for this quality (voxel {max(exterior):.2f}). "
            "Limit a blob to its radius or pick a Scan collection to focus it.")
    return result


def _middle_of_passages(ctx, voxel, half_width=0.25):
    """Voxels along the middle of spaces too tight for the clearance.

    A tunnel or hallway narrower than twice the clearance has no voxel that
    far from its walls, so it would get no cameras at all. Its middle - where
    the distance to the surfaces peaks across the passage - is still a good
    place for one, as long as the passage is at least ``2 * half_width`` wide.
    """
    field = ctx.field().reshape(ctx.grid.shape).astype(np.int16)
    field[field == 255] = 0
    need = int(min(9, max(2, math.ceil(half_width / voxel))))
    middle = field >= need
    for axis in range(3):
        lead = [slice(None)] * 3
        trail = [slice(None)] * 3
        lead[axis], trail[axis] = slice(1, None), slice(None, -1)
        middle[tuple(lead)] &= field[tuple(lead)] >= field[tuple(trail)]
        middle[tuple(trail)] &= field[tuple(trail)] >= field[tuple(lead)]
    return middle


def _box_sum(hist, yaw_win, pitch_win):
    """Sum over a (yaw, pitch) window centred on every bin; yaw wraps."""
    hy, hp = yaw_win // 2, pitch_win // 2
    padded = np.concatenate([hist[-hy - 1:], hist, hist[:hy + 1]], axis=0) if hy else hist
    table = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    table[1:, 1:] = padded.cumsum(0).cumsum(1)
    ny, np_ = hist.shape
    y = np.arange(ny) + (hy + 1 if hy else 0)
    y0, y1 = y - hy, y + hy + 1
    p = np.arange(np_)
    p0, p1 = np.clip(p - hp, 0, np_), np.clip(p + hp + 1, 0, np_)
    return (table[y1][:, p1] - table[y0][:, p1] - table[y1][:, p0] + table[y0][:, p0])


def _cluster_boxes(lo, hi, tol):
    """Group boxes that touch (within ``tol``) into connected clusters.

    A building is usually many objects - walls, roof, windows - whose bounds
    touch. Flat, wide pieces such as floors, roads and terrain patches are
    kept out of the linking, because they would join everything standing on
    them into one cluster.
    """
    n = len(lo)
    label = np.arange(n)
    extent = hi - lo
    wide = np.maximum(extent[:, 0], extent[:, 1])
    linker = ~((extent[:, 2] < 0.1 * wide) & (wide > 3.0 * np.median(np.max(extent, axis=1))))
    if n < 2:
        return label, linker
    ids = np.flatnonzero(linker)
    pairs_i, pairs_j = [], []
    for start in range(0, len(ids), 512):
        rows = ids[start:start + 512]
        touch = np.all((lo[rows, None] <= hi[None, ids] + tol)
                       & (hi[rows, None] + tol >= lo[None, ids]), axis=2)
        i, j = np.nonzero(touch)
        pairs_i.append(rows[i])
        pairs_j.append(ids[j])
    pi, pj = np.concatenate(pairs_i), np.concatenate(pairs_j)
    for _ in range(64):
        low = np.minimum(label[pi], label[pj])
        before = label.copy()
        np.minimum.at(label, pi, low)
        np.minimum.at(label, pj, low)
        label = label[label]
        if np.array_equal(label, before):
            break
    return label, linker


def _nearest_distance(points, targets, chunk=512):
    """Distance from each point to its nearest target (chunked brute force)."""
    out = np.empty(len(points))
    t2 = np.einsum('ij,ij->i', targets, targets)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        d2 = np.einsum('ij,ij->i', p, p)[:, None] + t2[None] - 2.0 * p @ targets.T
        out[start:start + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def _path_order(points):
    """Nearest-neighbour tour improved by 2-opt: a short camera walk."""
    n = len(points)
    if n <= 2:
        return list(range(n))
    dist = np.linalg.norm(points[:, None] - points[None], axis=2)
    start = int(np.argmin(points[:, 0] + points[:, 1] + points[:, 2]))
    order, unvisited = [start], np.ones(n, dtype=bool)
    unvisited[start] = False
    for _ in range(n - 1):
        row = np.where(unvisited, dist[order[-1]], np.inf)
        nxt = int(np.argmin(row))
        order.append(nxt)
        unvisited[nxt] = False
    if n <= 300:
        order = np.array(order)
        for _ in range(4):
            improved = False
            for i in range(1, n - 2):
                a, b = order[i - 1], order[i]
                c, d = order[i + 1:], np.append(order[i + 2:], -1)
                valid = d >= 0
                delta = np.full(len(c), np.inf)
                delta[valid] = (dist[a, c[valid]] + dist[b, d[valid]]
                                - dist[a, b] - dist[c[valid], d[valid]])
                j = int(np.argmin(delta))
                if delta[j] < -1e-9:
                    order[i:i + j + 2] = order[i:i + j + 2][::-1]
                    improved = True
            if not improved:
                break
        order = order.tolist()
    return order
