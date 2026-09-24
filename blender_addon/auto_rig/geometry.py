# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""The scene as the planner sees it: unique meshes plus instance transforms.

Pure numpy. ``scene_proxy`` fills it on Blender's main thread; the planner
reads it on a worker thread. Each unique mesh is stored once, in its own
space, however many times it is instanced - a forest of ten thousand
scattered trees costs one tree plus ten thousand matrices.

Sampling is area-weighted in *world* space across the whole scene, so a
sample budget spreads evenly over surfaces whatever the object count,
triangle density or object scale. A unit cube scaled into a 40 m ground slab
gets the samples its 3200 m² deserve, not the 6 m² of its local mesh.
"""

import numpy as np

_CHUNK = 1 << 20


def _cofactor(m):
    """Cofactor matrices: area vectors transform as ``cof(M) @ a``."""
    c0 = np.cross(m[..., :, 1], m[..., :, 2])
    c1 = np.cross(m[..., :, 2], m[..., :, 0])
    c2 = np.cross(m[..., :, 0], m[..., :, 1])
    return np.stack([c0, c1, c2], axis=-1)


class MeshData:
    """One unique triangle mesh in local space."""

    __slots__ = ("verts", "tris", "area_vectors", "cumulative", "area", "lo", "hi")

    def __init__(self, verts, tris):
        self.verts = np.ascontiguousarray(verts, dtype=np.float32).reshape(-1, 3)
        self.tris = np.ascontiguousarray(tris, dtype=np.int32).reshape(-1, 3)
        # Half cross products: their length is the triangle area, and they
        # transform exactly under any linear map, non-uniform scale included.
        vectors = np.empty((len(self.tris), 3), dtype=np.float32)
        for start in range(0, len(self.tris), _CHUNK):
            tri = self.verts[self.tris[start:start + _CHUNK]].astype(np.float64)
            vectors[start:start + len(tri)] = 0.5 * np.cross(
                tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.area_vectors = vectors
        self.cumulative = np.cumsum(np.linalg.norm(vectors, axis=1), dtype=np.float64)
        self.area = float(self.cumulative[-1]) if len(vectors) else 0.0
        if len(self.verts):
            self.lo = self.verts.min(axis=0).astype(np.float64)
            self.hi = self.verts.max(axis=0).astype(np.float64)
        else:
            self.lo = self.hi = np.zeros(3)

    def world_areas(self, cofactor):
        """Per-triangle world area under one instance's linear transform."""
        out = np.empty(len(self.tris), dtype=np.float64)
        for start in range(0, len(self.tris), _CHUNK):
            out[start:start + _CHUNK] = np.linalg.norm(
                self.area_vectors[start:start + _CHUNK] @ cofactor.T, axis=1)
        return out


class GeometrySet:
    def __init__(self):
        self.meshes = []
        self.owner_names = []
        self._inst_mesh = []
        self._inst_matrix = []
        self._inst_owner = []
        self.inst_mesh = np.zeros(0, np.int32)
        self.inst_matrix = np.zeros((0, 4, 4))
        self.inst_owner = np.zeros(0, np.int32)
        self.inst_area = np.zeros(0)
        self.inst_uniform = np.zeros(0, bool)
        self.inst_lo = np.zeros((0, 3))
        self.inst_hi = np.zeros((0, 3))
        self.owner_lo = np.zeros((0, 3))
        self.owner_hi = np.zeros((0, 3))
        self.owner_area = np.zeros(0)
        self.triangles = 0

    # ---- building (main thread) ----------------------------------------
    def add_mesh(self, verts, tris):
        mesh = MeshData(verts, tris)
        if mesh.area <= 0.0 or not len(mesh.tris):
            return -1
        self.meshes.append(mesh)
        return len(self.meshes) - 1

    def add_owner(self, name):
        self.owner_names.append(str(name))
        return len(self.owner_names) - 1

    def add_instance(self, mesh_index, matrix, owner):
        self._inst_mesh.append(mesh_index)
        self._inst_matrix.append(matrix)
        self._inst_owner.append(owner)

    def finalize(self):
        count = len(self._inst_mesh)
        self.inst_mesh = np.asarray(self._inst_mesh, dtype=np.int32)
        self.inst_matrix = (np.asarray(self._inst_matrix, dtype=np.float64).reshape(count, 4, 4)
                            if count else np.zeros((0, 4, 4)))
        self.inst_owner = np.asarray(self._inst_owner, dtype=np.int32)
        self._inst_mesh, self._inst_matrix, self._inst_owner = [], [], []
        owners = len(self.owner_names)
        if not count:
            return self
        linear = self.inst_matrix[:, :3, :3]
        cof = _cofactor(linear)
        gram = np.einsum('nji,njk->nik', cof, cof)
        scale2 = np.trace(gram, axis1=1, axis2=2) / 3.0
        deviation = np.linalg.norm(gram - scale2[:, None, None] * np.eye(3), axis=(1, 2))
        # Rotation plus uniform scale: every triangle scales by the same
        # factor, so the local area distribution is already exact.
        self.inst_uniform = deviation <= 0.02 * np.maximum(scale2, 1e-30)
        local_area = np.array([m.area for m in self.meshes])[self.inst_mesh]
        self.inst_area = local_area * np.sqrt(np.maximum(scale2, 0.0))
        for i in np.flatnonzero(~self.inst_uniform):
            self.inst_area[i] = float(self.meshes[self.inst_mesh[i]].world_areas(cof[i]).sum())
        lo = np.array([m.lo for m in self.meshes])[self.inst_mesh]
        hi = np.array([m.hi for m in self.meshes])[self.inst_mesh]
        corners = np.stack([np.where(np.array(bits, bool), hi, lo)
                            for bits in np.ndindex(2, 2, 2)], axis=1)
        world = np.einsum('nij,nkj->nki', linear, corners) + self.inst_matrix[:, None, :3, 3]
        self.inst_lo, self.inst_hi = world.min(axis=1), world.max(axis=1)
        self.owner_lo = np.full((owners, 3), np.inf)
        self.owner_hi = np.full((owners, 3), -np.inf)
        np.minimum.at(self.owner_lo, self.inst_owner, self.inst_lo)
        np.maximum.at(self.owner_hi, self.inst_owner, self.inst_hi)
        self.owner_area = np.bincount(self.inst_owner, weights=self.inst_area,
                                      minlength=owners)
        self.triangles = int(sum(len(self.meshes[i].tris) for i in self.inst_mesh))
        return self

    # ---- queries (any thread) --------------------------------------------
    @property
    def is_empty(self):
        return not len(self.inst_mesh) or not np.any(self.inst_area > 0)

    def _owner_pick(self, owners):
        pick = self.inst_area > 0
        if owners is not None:
            wanted = np.zeros(max(len(owners), int(self.inst_owner.max(initial=0)) + 1), bool)
            wanted[:len(owners)] = owners
            pick &= wanted[self.inst_owner]
        return pick

    def bounds(self, owners=None):
        """World bounds of every instance, or of the given owner mask."""
        pick = self._owner_pick(owners)
        if not pick.any():
            return None, None
        return self.inst_lo[pick].min(axis=0), self.inst_hi[pick].max(axis=0)

    def owner_roles(self):
        """(ground, backdrop) boolean masks over owners.

        Ground: one large flat surface at the bottom of the scene - it matters
        for occlusion but should not pull every camera out over empty terrain.
        Backdrop: a shell that encloses everything else, like a sky dome;
        left in, it would make the whole world one "interior".
        """
        n = len(self.owner_names)
        ground = np.zeros(n, bool)
        backdrop = np.zeros(n, bool)
        live = np.isfinite(self.owner_lo[:, 0]) if n else np.zeros(0, bool)
        if n < 2 or live.sum() < 2:
            return ground, backdrop
        extent = self.owner_hi - self.owner_lo
        order = np.argsort(-np.max(np.where(live[:, None], extent, 0), axis=1))
        for owner in order[:3]:
            others = live & ~backdrop
            others[owner] = False
            if not others.any():
                break
            o_lo = self.owner_lo[others].min(axis=0)
            o_hi = self.owner_hi[others].max(axis=0)
            inner = float(np.max(o_hi - o_lo))
            if (np.all(self.owner_lo[owner] <= o_lo) and np.all(self.owner_hi[owner] >= o_hi)
                    and float(np.min(extent[owner])) >= 2.5 * inner):
                backdrop[owner] = True
        keep = live & ~backdrop
        s_lo = self.owner_lo[keep].min(axis=0)
        s_hi = self.owner_hi[keep].max(axis=0)
        span = s_hi - s_lo
        xy = max(float(span[0]), float(span[1]), 1e-9)
        for owner in np.flatnonzero(keep):
            e = extent[owner]
            wide = max(float(e[0]), float(e[1]))
            if (wide >= 0.5 * xy and float(e[2]) <= 0.2 * wide
                    and self.owner_lo[owner, 2] <= s_lo[2] + 0.1 * max(float(span[2]), 1e-9)):
                ground[owner] = True
        if ground.all() or (keep & ~ground).sum() == 0:
            ground[:] = False
        return ground, backdrop

    def owner_closed(self):
        """Boolean over owners: every mesh of the object is closed (a solid).

        A solid's faces point outwards, so a camera behind one cannot see it;
        an open sheet - a plane, a card, a single-sided wall - may be seen
        from either side. Almost closed counts as closed: a mesh with a
        stray hole is still a solid for this purpose.
        """
        closed_mesh = np.zeros(len(self.meshes), dtype=bool)
        for index, mesh in enumerate(self.meshes):
            tris = mesh.tris
            if not len(tris):
                continue
            edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
            edges.sort(axis=1)
            key = edges[:, 0].astype(np.int64) * (int(len(mesh.verts)) + 1) + edges[:, 1]
            _unique, counts = np.unique(key, return_counts=True)
            closed_mesh[index] = np.mean(counts == 2) >= 0.98
        closed = np.ones(len(self.owner_names), dtype=bool)
        if len(self.inst_owner):
            np.logical_and.at(closed, self.inst_owner, closed_mesh[self.inst_mesh])
        return closed

    def sample(self, *, voxel, lo, hi, cap, rng, per_voxel=6.0, exclude=None):
        """Area-weighted surface points, normals and owner ids inside a box.

        Density aims at ``per_voxel`` samples per voxel face so that walls
        voxelise watertight; ``cap`` bounds memory on vast scenes.

        Three ways to weight an instance's triangles, cheapest first:
        rotation + uniform scale reuses the mesh's local area distribution;
        non-uniform scale is weighted in world space, batched per mesh; an
        instance the box clips is weighted per triangle with the outside
        dropped, and a huge triangle sampled only where it is in the box.
        """
        lo, hi = np.asarray(lo), np.asarray(hi)
        pick = ((self.inst_area > 0) & np.all(self.inst_hi >= lo, axis=1)
                & np.all(self.inst_lo <= hi, axis=1))
        if exclude is not None and exclude.any():
            pick &= ~self._owner_pick(exclude)
        pick = np.flatnonzero(pick)
        empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                 np.zeros(0, np.int32))
        if not len(pick):
            return empty
        linear = self.inst_matrix[:, :3, :3]
        size = np.maximum(self.inst_hi[pick] - self.inst_lo[pick], 1e-12)
        part = np.clip((np.minimum(self.inst_hi[pick], hi)
                        - np.maximum(self.inst_lo[pick], lo)) / size, 0.0, 1.0)
        thin = size < 0.05 * size.max(axis=1, keepdims=True)
        part = np.where(thin, 1.0, part)
        dims = np.maximum(1, 3 - thin.sum(axis=1))
        estimate = np.prod(part, axis=1) ** np.where(dims == 3, 2.0 / 3.0, 1.0)
        mesh_of = self.inst_mesh[pick]
        inside_area = self.inst_area[pick] * estimate
        full_area = self.inst_area[pick].copy()
        uniform = self.inst_uniform[pick]
        clipped = (estimate < 0.999) & ((estimate < 0.6) | ~uniform)
        # Uniform instances that are mostly inside keep the cheap local
        # distribution; their few outside samples are dropped below.
        direct = uniform & ~clipped

        exact = {}          # k -> cumulative world areas of one clipped instance
        batched = {}        # mesh -> (k indices, flat cumulative, per-instance totals)
        large = []          # (world triangles, owner) sampled on their clipped plane
        for mesh_index in np.unique(mesh_of[~uniform & ~clipped]):
            ks = np.flatnonzero(~uniform & ~clipped & (mesh_of == mesh_index))
            mesh = self.meshes[mesh_index]
            if len(ks) * len(mesh.tris) <= 4_000_000:
                cof = _cofactor(linear[pick[ks]])
                areas = np.linalg.norm(np.einsum('tj,kij->kti', mesh.area_vectors, cof), axis=2)
                totals = areas.sum(axis=1)
                batched[mesh_index] = (ks, np.cumsum(areas.ravel()), totals)
                inside_area[ks] = full_area[ks] = totals
            else:
                clipped[ks] = True
        for k in np.flatnonzero(clipped):
            row = pick[k]
            mesh = self.meshes[mesh_of[k]]
            areas = mesh.world_areas(_cofactor(linear[row]))
            inside = float(areas.sum())
            if estimate[k] < 0.999:
                # Keep every triangle whose bounds overlap the box - a ground
                # made of two huge triangles has its centre far outside it.
                inside = 0.0
                for start in range(0, len(mesh.tris), _CHUNK):
                    tri = mesh.verts[mesh.tris[start:start + _CHUNK]].astype(np.float64)
                    world_t = tri @ linear[row].T + self.inst_matrix[row, :3, 3]
                    t_lo, t_hi = world_t.min(axis=1), world_t.max(axis=1)
                    span = np.maximum(t_hi - t_lo, 1e-12)
                    over = np.clip((np.minimum(t_hi, hi) - np.maximum(t_lo, lo)) / span, 0.0, 1.0)
                    over = np.where(span < 0.05 * span.max(axis=1, keepdims=True), 1.0, over)
                    fraction = np.prod(over, axis=1)
                    block = areas[start:start + _CHUNK]
                    block[fraction <= 0] = 0.0
                    inside += float(np.sum(block * fraction))
                    # A huge triangle mostly outside the box would waste
                    # nearly every sample; sample only its part in the box.
                    huge = (fraction > 0) & (fraction < 0.25) & (block > 50 * voxel * voxel)
                    if huge.any():
                        large.append((world_t[huge], self.inst_owner[row]))
                        block[huge] = 0.0
            cumulative = np.cumsum(areas)
            exact[k] = cumulative
            inside_area[k] = inside
            full_area[k] = cumulative[-1] if len(cumulative) else 0.0
        total_area = float(inside_area.sum())
        if total_area <= 0:
            return empty
        wanted = int(np.clip(per_voxel * total_area / (voxel * voxel), 20_000, cap))
        density = wanted / total_area
        # Every instance is sampled over the surface that can reach the box
        # at the density its inside part needs; samples falling outside the
        # box are dropped below.
        counts = np.floor(density * full_area + rng.random(len(pick))).astype(np.int64)
        limit = int(cap * 6)
        if counts.sum() > limit:
            counts = np.floor(counts * (limit / counts.sum())).astype(np.int64)

        points, normals, owners = [], [], []

        def emit(mesh, inst, tri, n):
            v = mesh.verts[mesh.tris[tri]].astype(np.float64)
            r1 = np.sqrt(rng.random(n))[:, None]
            r2 = rng.random(n)[:, None]
            local = (1 - r1) * v[:, 0] + r1 * (1 - r2) * v[:, 1] + r1 * r2 * v[:, 2]
            m = linear[inst]
            world = np.einsum('nij,nj->ni', m, local) + self.inst_matrix[inst, :3, 3]
            keep = np.all((world >= lo) & (world <= hi), axis=1)
            if not keep.any():
                return
            world, m, inst = world[keep], m[keep], inst[keep]
            world_n = np.einsum('nij,nj->ni', _cofactor(m),
                                mesh.area_vectors[tri[keep]].astype(np.float64))
            world_n /= np.maximum(np.linalg.norm(world_n, axis=1, keepdims=True), 1e-30)
            points.append(world.astype(np.float32))
            normals.append(world_n.astype(np.float32))
            owners.append(self.inst_owner[inst])

        def chunks(ks):
            instance = np.repeat(ks, counts[ks])
            for start in range(0, len(instance), 400_000):
                yield instance[start:start + 400_000]

        for mesh_index in np.unique(mesh_of[direct]):
            mesh = self.meshes[mesh_index]
            for part_k in chunks(np.flatnonzero(direct & (mesh_of == mesh_index))):
                n = len(part_k)
                tri = np.searchsorted(mesh.cumulative, rng.random(n) * mesh.area, side='right')
                emit(mesh, pick[part_k], np.minimum(tri, len(mesh.tris) - 1), n)
        for mesh_index, (ks, cumulative, totals) in batched.items():
            mesh = self.meshes[mesh_index]
            count_t = len(mesh.tris)
            starts = np.concatenate([[0.0], np.cumsum(totals)[:-1]])
            for part_k in chunks(ks):
                n = len(part_k)
                local = np.searchsorted(ks, part_k)      # ks is sorted
                u = starts[local] + rng.random(n) * totals[local]
                flat = np.searchsorted(cumulative, u, side='right')
                tri = np.clip(flat - local * count_t, 0, count_t - 1)
                emit(mesh, pick[part_k], tri, n)
        for k, cumulative in exact.items():
            if not counts[k] or not len(cumulative) or cumulative[-1] <= 0:
                continue
            mesh = self.meshes[mesh_of[k]]
            for part_k in chunks(np.array([k])):
                n = len(part_k)
                tri = np.searchsorted(cumulative, rng.random(n) * cumulative[-1], side='right')
                emit(mesh, pick[part_k], np.minimum(tri, len(mesh.tris) - 1), n)
        for triangles, owner in large:
            for p, n in _clipped_triangle_samples(triangles, lo, hi, density, rng):
                points.append(p)
                normals.append(n)
                owners.append(np.full(len(p), owner, dtype=np.int32))
        if not points:
            return empty
        return np.concatenate(points), np.concatenate(normals), np.concatenate(owners)


def _clipped_triangle_samples(triangles, lo, hi, density, rng):
    """Uniform samples on the part of each triangle inside a box.

    Points are drawn uniformly in the box's footprint on the triangle's
    dominant projection plane, lifted onto the triangle's plane, and kept
    when inside both the triangle and the box. A uniform projected density
    is a uniform surface density, so this matches ordinary area sampling.
    """
    for tri in triangles:
        a, b, c = tri
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length <= 0:
            continue
        normal = normal / length
        drop = int(np.argmax(np.abs(normal)))
        keep_axes = [axis for axis in range(3) if axis != drop]
        t_lo = np.maximum(tri.min(axis=0), lo)[keep_axes]
        t_hi = np.minimum(tri.max(axis=0), hi)[keep_axes]
        if np.any(t_hi <= t_lo):
            continue
        # Projected area shrinks by |n_drop|, so draw more to compensate.
        count = int(density * np.prod(t_hi - t_lo) / abs(normal[drop]))
        count = min(count, 4_000_000)
        if count <= 0:
            continue
        uv = t_lo + rng.random((count, 2)) * (t_hi - t_lo)
        point = np.empty((count, 3))
        point[:, keep_axes[0]] = uv[:, 0]
        point[:, keep_axes[1]] = uv[:, 1]
        point[:, drop] = a[drop] - ((uv[:, 0] - a[keep_axes[0]]) * normal[keep_axes[0]]
                                    + (uv[:, 1] - a[keep_axes[1]]) * normal[keep_axes[1]]
                                    ) / normal[drop]
        # Inside-triangle test in the projection plane (same-side signs).
        p2 = uv
        a2, b2, c2 = a[keep_axes], b[keep_axes], c[keep_axes]

        def side(p, q, r):
            return (q[0] - p[0]) * (r[:, 1] - p[1]) - (q[1] - p[1]) * (r[:, 0] - p[0])

        s1, s2, s3 = side(a2, b2, p2), side(b2, c2, p2), side(c2, a2, p2)
        inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
        inside &= np.all((point >= lo) & (point <= hi), axis=1)
        if inside.any():
            chosen = point[inside].astype(np.float32)
            yield chosen, np.repeat(normal[None].astype(np.float32), len(chosen), axis=0)
