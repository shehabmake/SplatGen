"""Training configuration and presets.

Defaults follow the reference 3D Gaussian Splatting paper / gsplat's
simple_trainer. Every field is plain data so a config round-trips through
JSON (run folders, the web UI and the CLI all share it).
"""

from dataclasses import asdict, dataclass, fields


@dataclass
class TrainConfig:
    # -- schedule ----------------------------------------------------------
    steps: int = 7000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    # -- data --------------------------------------------------------------
    downscale: int = 1               # 1, 2, 4, 8
    test_every: int = 8              # hold out every Nth camera for evaluation; 0 = none
    background: str = "black"        # black | white | random
    mask_mode: str = "none"          # none | ignore (loss only where mask is set)
    # -- initialisation -------------------------------------------------------
    init: str = "points"             # points | random | ply
    init_ply: str = ""
    init_random_count: int = 100_000
    init_opacity: float = 0.1
    init_scale: float = 1.0
    # -- loss --------------------------------------------------------------
    ssim_weight: float = 0.2
    # -- learning rates (means is multiplied by the scene extent) -------------
    lr_means: float = 1.6e-4
    lr_means_final: float = 0.01     # factor reached at the last step
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    # -- densification ------------------------------------------------------
    densify_start: int = 500
    densify_stop: int = 5000
    densify_every: int = 100
    densify_grad_threshold: float = 0.0002
    grow_scale3d: float = 0.01       # split above this (x extent), clone below
    prune_opacity: float = 0.005
    prune_scale3d: float = 0.1       # prune larger than this (x extent) after the first reset
    reset_opacity_every: int = 3000
    max_gaussians: int = 3_000_000   # growth stops here; 0 = unlimited
    # -- system ------------------------------------------------------------
    device: str = "auto"             # auto | cuda | cpu | mps
    backend: str = "auto"            # auto | gsplat | torch
    seed: int = 0
    log_every: int = 10
    eval_every: int = 0              # evaluate the test views every N steps; 0 = at the end
    checkpoint_every: int = 0        # 0 = only at the end / when stopped
    export_formats: tuple = ("ply", "splat")

    def to_dict(self):
        data = asdict(self)
        data["export_formats"] = list(self.export_formats)
        return data

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in (data or {}).items() if k in known}
        if "export_formats" in values:
            values["export_formats"] = tuple(values["export_formats"])
        return cls(**values)


PRESETS = {
    "preview": {
        "label": "Preview",
        "description": "Quick look in a few minutes: half resolution, 2,000 steps.",
        "config": {"steps": 2000, "downscale": 2, "sh_degree": 1,
                   "densify_start": 300, "densify_stop": 1500,
                   "reset_opacity_every": 100000, "max_gaussians": 500_000},
    },
    "standard": {
        "label": "Standard",
        "description": "Good quality for most scenes: 7,000 steps at full resolution.",
        "config": {"steps": 7000, "densify_stop": 5000},
    },
    "high": {
        "label": "High quality",
        "description": "The full 3DGS schedule: 30,000 steps.",
        "config": {"steps": 30000, "densify_stop": 15000},
    },
}


def preset_config(name, overrides=None):
    data = TrainConfig().to_dict()
    data.update(PRESETS[name]["config"])
    data.update(overrides or {})
    return TrainConfig.from_dict(data)
