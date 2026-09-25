# Building splats directly from raw data

`splatgen build` (or **Build from raw data** in the app) turns a SplatGen raw
export (`Dataset(Raw)`, see [RAW_DATASET.md](RAW_DATASET.md)) into a finished
Gaussian-splat file **without training**. Every splat parameter is computed
from the data: where the surface is, how much detail it carries, what colour
it has from each camera. The result is a standard 3DGS `.ply`, so it opens in
any viewer and can go straight into the trainer for a short polish.

```
splatgen build <build folder> --out my_scene              # build only
splatgen build <build folder> --out my_scene --polish 2000  # build, then train 2,000 steps
splatgen build <build folder> --set rounds=4 color_threshold=0.02
```

Colours are fitted to the same display images a normal trainer fits
(`Dataset(Default)/images`), and the same views are held out (every
`test_every`-th), so the scores of built, polished and trained splats compare
directly.

## The pipeline

| Step | What happens | Code |
|---|---|---|
| 1. Surface samples | Every usable pixel of every build view becomes a point: position, normal, colour, object / material id, roughness and its **pixel footprint** (world size of the pixel there). | `construct/samples.py` |
| 2. Detail map | Samples are binned into an octree. Per cell, weighted sums give colour variation, curvature (normal spread) and whether several objects meet. | `construct/octree.py` |
| 3. Split | Cells with too much detail split into their children, down to the size the pixels can resolve. A 2:1 balance then grades sizes so a big splat never borders a much finer region. | `Octree.select`, `Octree.balance` |
| 4. Shape | Each leaf becomes a flat disk in the tangent plane, sized and stretched by its samples' spread. Leaves that still hold a colour edge are split **by colour** into two thin splats along the edge. | `construct/shape.py` |
| 5. Colour | Samples are grouped per (splat, camera); spherical harmonics are solved per splat by regularised least squares. Rough materials (roughness pass) are pushed towards view-independent colour; glossy ones keep their reflections. | `construct/color.py` |
| 6. Check & repeat | The splats are rendered into the build views. A linear least-squares solve refines the base colours against the real images, and cells whose error stays high are forced to split in the next round. | `construct/builder.py` |

A far shell of splats reproduces the sky / world behind the geometry
(`construct/background.py`).

### Samples

* Positions come from the `position` pass (or depth back-projected).
* Normals: the geometric `true_normal` AOV where it is valid, then Cycles'
  `normal` pass, then normals derived from the position pass. AOVs only exist
  on materials the exporter could extend, so objects without a material
  (`material_valid = 0`) automatically use the fallbacks.
* Pixels that are blends of two surfaces are dropped: silhouettes (a
  neighbour is sky or far away along the surface) and object-id borders.
* Weight = `cos(view angle) / (footprint / median footprint)^2`: frontal,
  close views resolve the surface best.
* When the views hold more pixels than `max_samples`, pixels are skipped on
  an even grid (offset per view), and the footprint grows accordingly.

### Detail and splitting

For every octree cell the builder keeps weighted sums of weight, normal,
colour and colour^2, so the split test is pure arithmetic:

* `color_std > color_threshold`: texture, text, edges
* `1 - |mean normal| > normal_threshold`: curvature and corners
* two object ids in one cell

A cell splits only while half its size is still at least `pixel_scale`
pixel footprints: there is no detail finer than the pixels that saw it.

### Shapes and edges

Shapes use exact, centred statistics (two passes over the samples, never
`E[x^2] - E[x]^2` in float32, which loses small cells to cancellation). The
disk's two axes are the eigenvectors of the in-plane covariance, with
`sigma = coverage * sqrt(eigenvalue)` so neighbours overlap without holes,
and never below `min_splat_px` footprints. Thickness along the normal is
`thickness` times the smaller axis.

A leaf that wanted to split but already is at pixel size and still has a
colour change above `edge_contrast` is an **edge leaf**. A linear colour
model `c(x) = c0 + G (x - mu)` gives the direction of the change. Its samples
are projected on the colour-change axis and separated by 1-D k-means, so the
two halves follow the actual border wherever it crosses the cell. Each half
becomes a long, thin splat with its own colour. This is what keeps text and
sharp borders crisp.

### Colour

Per splat and camera, the weighted mean colour of that camera's samples is
one observation along the direction camera → splat. The SH coefficients solve

    min  sum_views  w_v |SH(d_v) - c_v|^2  +  lambda_l * |coeffs of band l|^2

with `lambda_l = sh_regularization * l(l+1) * (1 + roughness_regularization *
roughness)`. Views are weighted by `sqrt(pixel count)`, so a close camera
counts more without drowning the others.

### Check & repeat

With the geometry fixed, every rendered pixel is a fixed blend of splat
colours, `image = W c`. The base colours are refined by solving the linear
least-squares problem `min |W c - gt|^2` over the build views with a
Jacobi-preconditioned iteration

    c <- c + a * (W^T r) / (W^T 1),   r = gt - W c

Both products come from one backward pass through the renderer. The blend
weights of a pixel sum to at most one, so the iteration converges for
`0 < a < 2` (`color_correction`, default 1). This is a direct solve of a
linear system: positions, sizes and opacities are never optimised.

Afterwards each sample knows its rendered error. Cells whose worst error
exceeds `error_threshold` are forced to split in the next round, and the
build repeats from step 3 (`rounds`, default 3).

## Settings

All fields of `ConstructConfig` (`trainer/splatgen/construct/config.py`) can
be set with `--set key=value` or in the app under *Build settings*.

| Setting | Default | Effect |
|---|---|---|
| `color_threshold` | 0.035 | Lower = more, smaller splats in textured areas |
| `normal_threshold` | 0.015 | Lower = finer splats on curved surfaces |
| `pixel_scale` | 1.0 | Smallest cell in pixel footprints |
| `edge_split` / `edge_contrast` | on / 0.08 | Two-splat edges for text and borders |
| `balance` | on | 2:1 size grading next to fine detail |
| `coverage` | 1.75 | Splat size vs. its area: higher is smoother and blurrier |
| `min_splat_px` | 0.5 | Smallest splat axis in pixel footprints |
| `opacity` | 0.95 | Opacity of surface splats |
| `sh_degree` | 3 | View-dependent colour |
| `rounds` | 3 | Check & repeat rounds |
| `correction_passes` | 3 | Colour-solve iterations per round |
| `error_threshold` | 0.06 | Mean abs error that forces a split |
| `background` | on | Sky / world shell |
| `test_every` | 8 | Held-out views (same rule as training) |

## Output

`point_cloud.ply` (standard 3DGS PLY with full SH), optional `.splat`, and
`construct.json` with the per-round report (splats, edge splats, check PSNR,
cells over the error limit) and the held-out scores.

## Polish

`--polish N` (or **Build + polish** in the app) trains N steps starting from
the built splats (`init = "ply"`), with settings suited to a good start:
slow position learning rate, all SH bands active from step one, a short
densification window and no opacity reset (a reset would throw the
construction away).

## Limits

* Surfaces are opaque disks. Glass and other transmissive materials show
  what is behind them in the images, which opaque splats cannot reproduce;
  polishing handles part of it.
* Standard 3DGS rasterisers widen every splat by a fixed screen-space filter
  and draw overlapping splats nearest-first. Neighbouring coplanar splats
  therefore bleed slightly towards the camera. Training learns to compensate;
  a direct build cannot, which is the main gap a short polish closes.
* Detail is limited to what the pixels resolved: text finer than a pixel in
  every view cannot be recovered.
