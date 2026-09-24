# SplatGen

Gaussian-splat training from 3D software scenes. Instead of reconstructing a
scene from photos, SplatGen takes everything the 3D application already knows —
exact cameras, geometry, materials and lighting — and hands it to a trainer.

## Repository layout

| Path | What it is |
|---|---|
| `blender_addon/` | **SplatGen Prepare** Blender extension (Blender 5.3): camera placement, dataset build, raw data export. |
| `docs/RAW_DATASET.md` | The `Dataset(Raw)` format: the contract between the add-on and the trainer. |
| `tools/build_addon_zip.py` | Packages `blender_addon/` into an installable zip in `dist/`. |

The trainer app and its web viewer will live alongside these.

## Building and installing the add-on

```
python tools/build_addon_zip.py        # -> dist/SplatGen_Prepare_<version>.zip
```

In Blender 5.3: Preferences > Add-ons > Install from Disk, pick the zip, enable
*SplatGen Prepare*. Place cameras, save the .blend, then **Build dataset**.

## What a build writes

```
SplatGen_<project>/<timestamp>/
├── Dataset(Default)/   legacy: images, masks, depth, COLMAP cameras/images/points
├── sg_metadata/        legacy: normals and scene guidance
└── Dataset(Raw)/       raw trainer input: render passes, materials, ids, cameras,
                        scene mesh, voxels, world and probes — see docs/RAW_DATASET.md
```
