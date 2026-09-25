"""Settings of the direct (training-free) splat construction."""

from dataclasses import asdict, dataclass, fields


@dataclass
class ConstructConfig:
    # -- sampling ------------------------------------------------------------
    max_samples: int = 12_000_000     # surface samples kept in memory (pixel stride adapts)
    test_every: int = 8               # views held out for evaluation (0 = use all)
    edge_px_ratio: float = 3.0        # neighbour distance (in footprints) that marks a depth edge
    # -- octree / detail -------------------------------------------------------
    pixel_scale: float = 1.0          # smallest splat = this many pixel footprints
    top_cells: int = 24               # cells along the scene's longest side at the top level
    color_threshold: float = 0.035    # split when colour std-dev in a cell exceeds this
    normal_threshold: float = 0.015   # split when 1 - |mean normal| exceeds this (curvature)
    min_samples_split: int = 6        # never split cells with fewer samples
    balance: bool = True              # grade splat sizes: neighbours differ at most 2x
    edge_split: bool = True           # split finest cells along colour edges into two splats
    edge_contrast: float = 0.08       # colour change across a finest cell that counts as an edge
    # -- shape -----------------------------------------------------------------
    coverage: float = 1.75            # sigma = coverage * sqrt(in-plane variance)
    min_splat_px: float = 0.5         # no splat axis smaller than this many pixel footprints
    thickness: float = 0.08           # disk thickness relative to the smaller in-plane sigma
    opacity: float = 0.95
    # -- colour ------------------------------------------------------------------
    sh_degree: int = 3
    sh_regularization: float = 0.02   # ridge weight of higher SH bands (scaled by band)
    roughness_regularization: float = 4.0   # extra ridge on rough materials (raw roughness pass)
    # -- check & repeat --------------------------------------------------------------
    rounds: int = 3
    error_threshold: float = 0.06     # mean abs error of a splat's pixels that forces a split
    color_correction: float = 1.0     # step of the linear colour solve (0..2)
    check_views: int = 64             # views rendered per round (evenly spread)
    correction_passes: int = 3        # iterations of the colour solve per round
    # -- background ------------------------------------------------------------------
    background: bool = True           # add a far shell of splats for the sky / world
    background_resolution: int = 48   # cube-map cells per face side
    # -- system -----------------------------------------------------------------------
    device: str = "auto"
    backend: str = "auto"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})
