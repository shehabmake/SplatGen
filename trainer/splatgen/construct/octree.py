"""Steps 2 and 3 - detail map and adaptive split.

Samples are binned into an octree of cubic cells. Every level keeps weighted
running sums per cell, so the statistics that decide splitting come straight
from sums, with no per-cell loops:

    colour variance     how much the colour changes inside the cell (detail)
    normal spread       1 - |mean normal|: curvature or a corner
    object mix          more than one object id in the cell

A cell splits into its occupied children while it has too much detail and
is still larger than the finest size its samples can resolve (their pixel
footprint). Cells that stop become leaves; each leaf becomes one splat (two
on a colour edge, see ``shape``).
"""

import math

import torch

# Moment layout (per cell): weighted sums of weight, normal, colour, colour^2.
W, N, C, CC = 0, slice(1, 4), slice(4, 7), slice(7, 10)
MOMENTS = 10


def sample_moments(s):
    """Per-sample contribution to the cell sums (N, MOMENTS)."""
    w = s["w"][:, None]
    c = s["col"]
    return torch.cat([w, w * s["nrm"], w * c, w * c * c], dim=1)


def pack(q):
    return q[:, 0] | (q[:, 1] << 21) | (q[:, 2] << 42)


def unpack(keys):
    mask = (1 << 21) - 1
    return torch.stack([keys & mask, (keys >> 21) & mask, (keys >> 42) & mask], dim=1)


class Octree:
    def __init__(self, samples, config, log=print):
        self.cfg = config
        pos = samples["pos"]
        self.origin = pos.min(dim=0).values - 1e-4
        extent = float((pos.max(dim=0).values - self.origin).max())
        self.top = extent / config.top_cells
        fine = float(torch.quantile(samples["foot"][: 2_000_000], 0.02)) * config.pixel_scale
        self.levels = max(1, min(18, math.ceil(math.log2(max(self.top / max(fine, 1e-9), 1.0))) + 1))
        self.sizes = [self.top / (2 ** level) for level in range(self.levels)]
        self.moments = sample_moments(samples)
        self.samples = samples
        self.cells = []
        for level, size in enumerate(self.sizes):
            q = torch.floor((pos - self.origin) / size).long().clamp_min(0)
            keys, inverse = torch.unique(pack(q), return_inverse=True)
            sums = torch.zeros(len(keys), MOMENTS, device=pos.device).index_add_(0, inverse, self.moments)
            foot = torch.full((len(keys),), float("inf"), device=pos.device).scatter_reduce(
                0, inverse, samples["foot"], "amin")
            count = torch.bincount(inverse, minlength=len(keys))
            obj_lo = torch.full((len(keys),), 2 ** 30, device=pos.device, dtype=torch.int32).scatter_reduce(
                0, inverse, samples["obj"], "amin")
            obj_hi = torch.full((len(keys),), -1, device=pos.device, dtype=torch.int32).scatter_reduce(
                0, inverse, samples["obj"], "amax")
            self.cells.append({"keys": keys, "inverse": inverse, "sums": sums, "foot": foot,
                               "count": count, "mixed": obj_lo != obj_hi})
        log(f"octree: {self.levels} levels, cell {self.sizes[0]:.4g} -> {self.sizes[-1]:.4g}, "
            f"{len(self.cells[-1]['keys']):,} finest cells")

    # -- statistics ----------------------------------------------------------------

    @staticmethod
    def color_std(sums):
        w = sums[:, W].clamp_min(1e-12)[:, None]
        mean = sums[:, C] / w
        var = (sums[:, CC] / w - mean * mean).clamp_min(0)
        return var.mean(dim=1).sqrt()

    @staticmethod
    def normal_spread(sums):
        w = sums[:, W].clamp_min(1e-12)
        return 1.0 - (sums[:, N].norm(dim=1) / w).clamp(max=1.0)

    # -- the split decision -------------------------------------------------------------

    def select(self, forced=None):
        """Leaves as a list of (level, cell index tensor, 'wanted to split' flag)."""
        cfg = self.cfg
        forced = forced or {}
        leaves = []
        active = torch.arange(len(self.cells[0]["keys"]), device=self.origin.device)
        for level in range(self.levels):
            cell = self.cells[level]
            sums = cell["sums"][active]
            detail = ((self.color_std(sums) > cfg.color_threshold)
                      | (self.normal_spread(sums) > cfg.normal_threshold)
                      | cell["mixed"][active])
            if level in forced:
                detail |= torch.isin(cell["keys"][active], forced[level])
            resolvable = (self.sizes[level] / 2) >= cell["foot"][active] * cfg.pixel_scale
            can = resolvable & (cell["count"][active] >= cfg.min_samples_split) & (level + 1 < self.levels)
            split = detail & can
            leaves.append((level, active[~split], detail[~split]))
            if level + 1 >= self.levels or not bool(split.any()):
                break
            parents = cell["keys"][active[split]]
            child = self.cells[level + 1]
            parent_of_child = pack(unpack(child["keys"]) // 2)
            active = torch.nonzero(torch.isin(parent_of_child, parents)).squeeze(-1)
        return leaves

    def balance(self, leaves, max_iterations=24):
        """Grade the leaves so neighbouring splats differ at most 2x in size.

        A big splat's Gaussian tail reaches well into the next cell; next to a
        finely split edge that tail would paint over the detail. Any leaf that
        touches (26-neighbourhood) a region refined two or more levels deeper
        is replaced by its children, until no such leaf is left (the classic
        2:1 balance of adaptive octrees)."""
        dev = self.origin.device
        offsets = torch.tensor([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                                if (x, y, z) != (0, 0, 0)], device=dev)
        by_level = {level: (cells, flag) for level, cells, flag in leaves}
        for _ in range(max_iterations):
            # Keys, per level L, of the regions that hold leaves at level >= L + 2.
            deep = {}
            for level, (cells, _flag) in by_level.items():
                if len(cells) == 0:
                    continue
                q = unpack(self.cells[level]["keys"][cells])
                for up in range(level - 2, -1, -1):
                    keys = pack(q >> (level - up))
                    deep[up] = torch.cat([deep[up], keys]) if up in deep else keys
            changed = False
            for level in sorted(by_level):
                cells, flag = by_level[level]
                if level not in deep or len(cells) == 0 or level + 1 >= self.levels:
                    continue
                q = unpack(self.cells[level]["keys"][cells])
                near = (q[:, None, :] + offsets[None]).clamp_min(0)
                hit = torch.isin(pack(near.reshape(-1, 3)), torch.unique(deep[level])).reshape(len(cells), -1).any(1)
                if not hit.any():
                    continue
                parents = self.cells[level]["keys"][cells[hit]]
                child = self.cells[level + 1]
                kids = torch.nonzero(torch.isin(pack(unpack(child["keys"]) // 2), parents)).squeeze(-1)
                by_level[level] = (cells[~hit], flag[~hit])
                old_cells, old_flag = by_level.get(level + 1, (kids[:0], torch.zeros(0, dtype=torch.bool, device=dev)))
                by_level[level + 1] = (torch.cat([old_cells, kids]),
                                       torch.cat([old_flag, torch.zeros(len(kids), dtype=torch.bool, device=dev)]))
                changed = True
                break                       # recompute the deep regions after each change
            if not changed:
                break
        return [(level, *by_level[level]) for level in sorted(by_level)]

    def leaf_of_samples(self, leaves):
        """Leaf index (into the concatenated leaves) of every sample."""
        n = len(self.samples["pos"])
        out = torch.full((n,), -1, dtype=torch.long, device=self.origin.device)
        offset = 0
        for level, cells, _flag in leaves:
            lookup = torch.full((len(self.cells[level]["keys"]),), -1, dtype=torch.long, device=out.device)
            lookup[cells] = torch.arange(len(cells), device=out.device) + offset
            mine = lookup[self.cells[level]["inverse"]]
            take = mine >= 0
            out[take] = mine[take]
            offset += len(cells)
        return out
