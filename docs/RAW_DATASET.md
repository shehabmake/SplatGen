# SplatGen raw dataset (`Dataset(Raw)`) — format v2

This is the contract between the Blender add-on (`blender_addon/`) and the
SplatGen trainer app. The add-on writes it; the trainer reads it. Schema id:
`splatgen-raw-dataset-v2` (in `manifest.json`).

## Where it lives

Every build folder now holds three siblings:

```
SplatGen_<project>/<timestamp>/
├── Dataset(Default)/   legacy portable dataset — written exactly as before
├── sg_metadata/        legacy SplatGen metadata — written exactly as before
└── Dataset(Raw)/       everything below (new)
```

The legacy folders are byte-for-byte what the old add-on produced (this is
checked by the test harness), so old and new approaches can be compared on the
same build.

## Folder layout

Per-view files are named `frame_NNNN_<pass>.exr`, for example
`geometry/normal/frame_0003_normal.exr` or
`appearance/clay/diffuse_direct/frame_0003_clay_diffuse_direct.exr`:

* `frame_NNNN` is the same number as `Dataset(Default)/images/frame_NNNN.png`
  and the names in `images.txt`, so every pass lines up with its camera.
* `<pass>` is the pass key used in `manifest.json`, so a file says what it
  holds even when it is copied out of its folder.
* The frame number comes first, so files sort by camera.

Scene-level files are named after what they hold: `world_radiance.exr`,
`probe_000_radiance.exr`, `probe_000_distance.exr`, `scene_mesh.ply`, …

```
Dataset(Raw)/
├── manifest.json               table of contents — read this first
├── scene_description.json      objects, lights, materials, world, render settings
├── README.txt
├── appearance/
│   ├── combined/               beauty render, scene-linear RGBA
│   ├── lighting/
│   │   ├── diffuse_direct/     diffuse_indirect/     diffuse_color/
│   │   ├── glossy_direct/      glossy_indirect/      glossy_color/
│   │   ├── transmission_direct/ transmission_indirect/ transmission_color/
│   │   ├── volume_direct/      volume_indirect/
│   │   ├── emission/
│   │   └── environment/
│   └── clay/                   every surface replaced by 0.8 grey Lambert
│       ├── combined/  diffuse_direct/  diffuse_indirect/  diffuse_color/
├── geometry/
│   ├── position/  depth/  normal/  true_normal/  uv/
│   ├── object_coords/  pointiness/  backfacing/  ambient_occlusion/
├── material/
│   ├── base_color/  roughness/  metallic/          (base surface)
│   ├── specular/  ior/  anisotropic/                (reflection)
│   ├── coat/  sheen/                                (surface layers)
│   ├── transmission/  subsurface/  alpha/           (light transport)
│   ├── emission_color/  emission_strength/          (emission)
│   └── material_valid/                              (trust mask, see below)
├── ids/
│   ├── object_id/  material_id/
│   └── id_map.json             id -> object / material name
├── motion_denoising/
│   ├── motion_vector/
│   ├── noisy_image/            combined before the denoiser
│   ├── denoising_albedo/
│   └── denoising_normal/
├── cameras/
│   ├── cameras.json            intrinsics + poses in Blender, OpenCV and COLMAP form
│   └── colmap/                 cameras.txt  images.txt  points3D.txt (copies)
├── scene/
│   ├── scene_mesh.ply          all render-visible geometry, world space
│   ├── scene_mesh.json         per-object triangle ranges, bounds
│   ├── collision_voxels.npz    free / surface / enclosed grid
│   └── collision_voxels.json
└── environment/
    ├── world/
    │   ├── world_radiance.exr      the world alone, equirectangular
    │   ├── world_source_<name>     copy of every HDRI the world uses
    │   └── world.json          background strength, mapping, source paths
    └── probes/
        ├── probe_000_radiance.exr  probe_000_distance.exr
        └── probes.json         positions, how they were chosen
```

## Conventions

| Topic | Convention |
|---|---|
| World frame | Blender world space: right-handed, **Z up**, scene units (metres by default). Same frame as the legacy COLMAP files — the export transform is the identity. |
| Pixel origin | Top-left. Every raster matches the legacy RGB with no flipping. |
| EXR | Single-part OpenEXR. Colour **and vector** passes use colour channels `R,G,B` (plus `A` where there is alpha or coverage), so every viewer opens them as an image; scalar passes use one channel `V`. For vectors `R = X`, `G = Y`, `B = Z` — each pass's `space` in the manifest spells it out. |
| Colour | Scene-linear in Blender's working space (recorded under `color_management`); no view transform (AgX/Filmic) applied. The legacy PNGs *do* have the view transform. |
| Camera axes | `c2w_blender`: looks down −Z, +Y up. `c2w_opencv` / `w2c_opencv`: +Z forward, +Y down (COLMAP). |
| Intrinsics | Pixels, `PINHOLE` (`fx, fy, cx, cy`), principal point from the top-left pixel corner. Pixel centre `(u+0.5, v+0.5)`. |
| Depth | Planar camera Z (not ray length). Background = `1e10`. Identical to legacy `Dataset(Default)/depth`. |
| Position | World-space position of the first visible surface. |
| Normals | World space. `normal` = shading normal (normal maps applied); `true_normal` = geometric face normal. |
| AOV coverage | RGBA shader-AOV passes (`base_color`, `true_normal`, `object_coords`, `emission_color`) carry coverage in **A** (1 where a surface with an editable material was hit). |
| IDs | Integers stored in float32. `0` = background / none. Names in `ids/id_map.json`. Not anti-aliased. |
| Motion | Pixels. R,G = motion to the previous frame, B,A = to the next frame. Zero for a static scene (every camera is a still at the current frame). |
| Equirect | `u = 0.5 − atan2(y, x) / 2π` (centre looks along +X, u = 0.25 along +Y), top row = +Z. Identical to Blender's Environment Texture, so a world or probe render can be plugged straight back in as a world. |

### Useful identities

These hold per pixel and are verified by the test harness:

```
noisy_image  =  (diffuse_direct + diffuse_indirect) * diffuse_color
             +  (glossy_direct  + glossy_indirect)  * glossy_color
             +  (transmission_direct + transmission_indirect) * transmission_color
             +  volume_direct + volume_indirect + emission + environment

combined     =  denoise(noisy_image)        (equal when the denoiser is off)

position     =  c2w_opencv · ( (u+0.5−cx)/fx · depth, (v+0.5−cy)/fy · depth, depth, 1 )
```

`clay/diffuse_color` is 0.8 everywhere a surface was hit, so
`clay/diffuse_direct + clay/diffuse_indirect` is the scene's irradiance
(lighting without albedo) seen through a neutral surface.

## How each output is produced

| Output | Source |
|---|---|
| combined, lighting split, geometry built-ins (position, depth, normal, uv, AO), ids, motion, denoising | Cycles render passes of **the same render** that writes the legacy RGB — one render per camera, no extra cost beyond pass memory. |
| material/*, true_normal, object_coords, pointiness, backfacing | Shader **AOV Output** nodes temporarily added to every editable material, wired to whatever feeds the BSDF input (so image textures and procedurals are evaluated per pixel). Unlinked inputs become constants. Non-Principled BSDFs are mapped where it makes sense (Diffuse, Glossy, Glass, Emission, …). |
| object_id / material_id | Each object and material temporarily gets a unique `pass_index`; ids are stable across re-runs because `id_map.json` is reused by name. If any material *reads* the pass index, the scene's own indices are kept and the map says so (`mode: user_pass_index`). With the ID passes switched off the map is still written (`mode: planned`) because the scene mesh and scene description use it. |
| clay | A second render per camera with a View Layer material override (grey Lambert). Volume-only objects are hidden for it (they would turn into solid grey boxes). Always Cycles. |
| world | A temporary scene that shares only the World, rendered with an equirectangular camera. Always Cycles. |
| probes | The full scene rendered with an equirectangular camera from each probe position. **Light Probe Sphere** objects in the scene are used if present; otherwise `Automatic Probes` are placed in free space (from the voxel grid) near the middle of the geometry. |
| scene_mesh.ply | Dependency-graph evaluated geometry (modifiers, Geometry Nodes, instances, curves and text) of render-visible objects, in world space. Faces carry `object_id` and `material_id` from `id_map.json`. |
| collision_voxels.npz | Surface voxelisation of the scene mesh; free space is flood-filled from the grid border **and from every dataset camera**, so the inside of a room is free while the inside of a thick wall is `enclosed`. |

Every scene change is recorded on an undo stack and restored when the render
finishes; the harness checks that the scene is identical before and after.
Leftovers from a crash are named `__SPLATGEN_RAW__…` and purged on load.

## manifest.json

```jsonc
{
  "schema": "splatgen-raw-dataset-v2",
  "resolution": [W, H],
  "frame_count": N,
  "frames": [{"index": 0, "stem": "frame_0000", "camera_name": "...",
              "legacy_image": "../Dataset(Default)/images/frame_0000.png"}],
  "passes": {
    "roughness": {
      "path_pattern": "material/roughness/{stem}_roughness.exr",
      "channels": ["V"], "pixel_type": "float",
      "space": "raw shader input value",
      "status": "complete",          // complete | partial | missing | unavailable | disabled
      "frames_present": N,
      "missing_frames": [...],        // only when partial
      "reason": "..."                 // why unavailable / disabled
    }
  },
  "files": {"cameras": "cameras/cameras.json", "scene_mesh": "scene/scene_mesh.ply", ...},
  "materials": {"<name>": {"shader": "principled", "linked_inputs": ["coat"], ...}},
  "conventions": {...},
  "color_management": {...},
  "status": "complete",               // or "incomplete" with "issues": [...]
  "notes": [...]
}
```

A pass is `unavailable` when the render engine cannot produce it — for
example Cycles writes no motion vectors while motion blur is on, and with EEVEE
as the beauty engine the Cycles lighting split (direct/indirect/color, volume)
and denoising passes do not exist. The reason is stored. Pointiness is only
meaningful with Cycles. For the full set, render the beauty with Cycles.

## `material_valid`

| Value | Meaning |
|---|---|
| 1.0 | Material has a Principled BSDF; every material pass is read from it. |
| 0.5 | Another BSDF (Diffuse, Glossy, Glass, Emission, …); passes it has are read, the rest are defaults. |
| 0.25 | No recognisable BSDF; viewport colour/roughness/metallic are used. |
| 0 | No material, or a linked (library) material that cannot be edited. |

## Storage

Settings live in *Dataset settings → Raw data*:

* **Color precision** — Half (default) or Float for the appearance/lighting/clay passes.
* **Color compression** — ZIP or PIZ (lossless), or DWAA (lossy, much smaller).
* **Data compression** — ZIP (lossless, default) or PXR24 (float32 rounded to
  24 bits: about 5 significant digits, e.g. ~0.3 mm at 10 m). Depth, ids and
  motion vectors are always stored exactly.
* Every group can be switched off, and the whole raw export can be disabled.

Blender only honours half precision for colour-channel outputs: normals are
stored as half, position and UV stay float32 for precision, and scalar passes
(`V`) are float32 — their flat regions compress very well.

Measured on the test scene at 960×540 (Cycles, 16 samples), per view (before normals moved to half precision in v2, so geometry is now slightly smaller):

| Group | ZIP + ZIP (default) | DWAA color + PXR24 data |
|---|---:|---:|
| combined | 0.8 MiB | 0.1 MiB |
| lighting split (13 passes) | 5.3 MiB | 2.5 MiB |
| clay (4 passes) | 1.5 MiB | 0.5 MiB |
| geometry (9 passes) | 8.6 MiB | 3.0 MiB |
| material (14 passes) | 0.7 MiB | 0.4 MiB |
| ids | 0.02 MiB | 0.02 MiB |
| motion & denoising | 1.9 MiB | 1.0 MiB |
| **total raw** | **18.8 MiB** | **7.4 MiB** |
| legacy Dataset(Default) | 0.8 MiB | 0.8 MiB |

Size scales with pixel count (×4 at 1920×1080) and with render noise — clean,
denoised or high-sample renders compress better. The biggest items are
`position`, `uv` and `object_coords` in float32; PXR24 shrinks position ~8×.
`position` is also exactly recoverable from `depth` + `cameras.json` (see the
identity above), so it is the first group to switch off if space is tight.
With the smallest settings the harness still verifies position to ~1e-5 m,
exact depth, and the lighting identity to within 1 %.

## Using it in the trainer (notes for the planned approaches)

* **Initialise splats without training.** `geometry/position` + `geometry/normal`
  (or `true_normal`) give an exact surface sample per pixel; `material/base_color`,
  `roughness`, `metallic` and `alpha` give per-splat material; `ids/*` group
  splats per object/material. `scene/scene_mesh.ply` is an alternative,
  view-independent source for positions and normals, and
  `scene/collision_voxels.npz` marks space that must stay empty (floater
  removal) and space no camera can see (`enclosed`).
* **Supervise with more than RGB.** Depth, normals, albedo (`material/base_color`
  or `motion_denoising/denoising_albedo`), roughness/metallic and ids are all
  pixel-aligned with the RGB and share the camera model in `cameras/cameras.json`.
* **Layered training (diffuse first, specular on top).** Use the identity above:
  fit `(diffuse_direct + diffuse_indirect) * diffuse_color` first, then add the
  glossy and transmission terms. Train the layers against `noisy_image` (or
  render with the denoiser off) — the light passes are not denoised, so they
  only sum to `combined` when denoising is off. `clay/*` gives the incident
  lighting without albedo; `environment/world` and `environment/probes` give the
  distant and local illumination for relighting.

## Known limitations

* Geometry for `scene_mesh.ply` / voxels is evaluated at **viewport** modifier
  levels (Python has no render-level depsgraph). Rendered passes use render levels.
* Material passes read the BSDF input, not a fully evaluated closure: a
  Principled BSDF nested inside a node **group** is not found (the material
  falls back to `material_valid = 0.25`), and a Mix Shader reports its first
  Principled input.
* The clay override replaces every material, so emissive meshes stop emitting
  and alpha-clipped geometry becomes solid in the clay render only.
* Linked (library) materials and objects cannot be edited: they get no
  material AOVs and keep their own pass index (listed in `id_map.json`).
* Auxiliary renders (clay, world, probes) always use Cycles, even when the
  beauty render uses EEVEE.

## Changes from v1

* Per-view files renamed from `frame_NNNN.exr` to `frame_NNNN_<pass>.exr`.
* Normal, position, UV and denoising normal use channels `R,G,B` instead of
  `X,Y,Z` (same values); normals are now stored as half.
* `world_equirect.exr` → `world_radiance.exr`, `source_<name>` →
  `world_source_<name>`, `probe_000/radiance.exr` → `probe_000_radiance.exr`
  (likewise `distance`).
