# SplatGen Prepare 5.3.1

A preparation-only derivative of SplatGen Pro 5.3 pointcloud, for Blender 5.3.x
on Windows x64. Only camera placement and dataset building are available.

## Install

In Blender 5.3, open Preferences > Add-ons, use the menu's Install from Disk,
and select SplatGen_Prepare_5.3.zip. Enable SplatGen Prepare. Disable the original
SplatGen Pro first: the editions share operator identifiers and should not be
enabled together. The extension has its own ID, splatgen_prepare.

## Workflow

1. Open the 3D View sidebar > SplatGen. Preferences can move it to Scene Properties.
2. Add cameras using Smart rig, Manual (current view or mesh faces), or Saved rigs.
3. Review the camera queue, adjust camera settings, and check/fix coverage.
4. Save the .blend file, select an output directory, and set image/sampling options.
5. Build Dataset, or render images and generate the point cloud separately.
6. Open the output folder to use the generated dataset in a separate application.

Preserved outputs include rendered images, masks, metric depth, cameras.txt,
images.txt, points3D.txt, dataset manifests and optional numerical scene metadata.
Camera preview and coverage overlays remain because they help place cameras.
There is no Gaussian-splat viewer, dataset point-cloud display, splat import/export,
trainer, training method selector, runtime activation or bundled GPU runtime.
The old Dataset preview popover was removed along with the viewer implementation.

## Raw data export (5.3.1)

Every build also writes `Dataset(Raw)` next to `Dataset(Default)`: the beauty
render and its full Cycles lighting split, a clay (lighting-only) render,
position/depth/normal/true-normal/UV/object-coordinate/pointiness/backfacing/AO,
per-pixel material parameters read from each material's shader inputs, object
and material ids, motion vectors, the noisy image and denoising albedo/normal,
`cameras.json` plus a COLMAP copy, a world-space scene mesh (PLY), collision
voxels, a world/HDRI panorama and reflection-probe panoramas, and
`scene_description.json` + `manifest.json`.

The legacy `Dataset(Default)` and `sg_metadata` output is unchanged. Beauty
passes come from the same render as the legacy image; the other renders run
after the point cloud. Settings are under Dataset settings > Raw data, where
**Export raw data for this build** also upgrades an existing build, rendering
only what is missing. The format is documented in `docs/RAW_DATASET.md` of the
SplatGen repository.

The point cloud is dataset input; generating it does not train Gaussian splats.
References to training images in dataset fields describe the files' intended use
by an external application. They do not enable training inside this build.

The add-on uses Blender's Python, NumPy and rendering engine. No extra Python
packages are bundled or installed. Blender's own GPU support is unchanged;
the removed CUDA libraries belonged only to the packaged trainers.

## Validation

Tested in Blender 5.3.0 Alpha, build b2e052b7172a (2026-09-20): registration,
Prepare interface property/operator references for Smart/Manual/Saved cameras,
a full two-camera 64x64 CPU dataset render and export producing 83 points,
completion without trainer/viewer imports, and unregister/re-register cleanup.
This is a small automated smoke test, not an exhaustive visual or large-scene test.

See REMOVED_COMPONENTS.md for removed features and dependencies.
The original input ZIP is unchanged. Original author and GPL license are retained.
