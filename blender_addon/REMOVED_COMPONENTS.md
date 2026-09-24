# Removed components — SplatGen Prepare 5.3

| Removed | Why / effect |
|---|---|
| runtime/splatgen_v2/ including splatgen-v2.exe | Bundled modified Brush application; removes the V2 training backend. |
| runtime/python313/site-packages/ | Entire training package payload: PyTorch, CUDA/cuDNN/cuBLAS libraries, gsplat native kernels, fused-ssim and supporting Python packages. Preparation uses Blender's own Python/NumPy. |
| Remaining runtime/ files | Training runtime metadata and assets are no longer needed. |
| splat_training/ | Removes training methods, configuration, presets, job management, activation, operators and trainer UI. |
| trainer_worker.py, splatgen_one_trainer.py, WORKER_REQUIREMENTS.txt | Removes standalone training/worker implementation and its dependency specification. |
| preview_transport.py, color_pipeline.py | Removes live trainer preview transport and trainer input color-conversion staging. Dataset rendering still preserves its existing color export logic. |
| splat_manager/ | Removes Gaussian-splat loading, viewing, display settings, instance management and viewer export tools. |
| licenses/, third_party_licenses/ | Attribution files for trainer components no longer redistributed. The add-on GPL LICENSE and original author attribution are retained. |
| Old Pro manuals, training/research docs, release/marketing/support/validation documents | Replaced by preparation-specific README, local guide, notices and this report so the new build does not advertise unavailable features. The camera-rig guide remains. |

## Changes to retained code

- Register only preparation, camera and coverage components.
- Remove Train/View navigation, trainer/viewer imports, trainer status rendering,
  runtime/GPU activation properties and Pro upgrade UI/operators.
- Keep only PREPARE in the workflow enum, including the workflow operator.
- Remove the Dataset preview popover and viewer shortcut operators.
- Remove post-build viewer refresh and automatic trainer handoff; completing a
  dataset now only records its output paths.
- Remove trainer checks from output-directory updates and preparation busy checks.
- Remove trainer runtime/log diagnostics, so intentionally absent files are not
  reported as a broken installation.
- Name the extension SplatGen Prepare and use the ID splatgen_prepare.
- Keep Blender 5.3.x compatibility requirements; remove the native Gaussian-splat
  capability check because dataset preparation does not need the viewer API.

## Retained

Camera creation, Smart rig, saved rigs, queue/review, coverage checks and overlays,
rendered images, masks/depth, cameras.txt, images.txt, points3D.txt, dataset versioning,
optional numerical dataset metadata, stop/progress controls, settings and shortcuts.
No embedded training method or trained-splat viewer remains. Dataset points and
camera/coverage overlays are preparation features, not trained Gaussian splats.

The adjacent SplatGen_Prepare_5.3_removed_files.csv is the exact source-archive
removal inventory (path, original uncompressed size and compressed payload size).
Files replaced with edition-specific content are listed separately in the build
manifest. No user's installed Blender libraries or original archive were changed.
