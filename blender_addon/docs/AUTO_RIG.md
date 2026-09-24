# Smart camera rig — method and interface (5.3.26)

Open **Prepare → 1 · Add cameras → Smart rig**. Calculation settings use the
sliders button beside Calculate cameras. The shared queue is above this step;
coverage is the separate second step. Manual and Saved are alternative camera
methods in the same selector.

The user adds **systems** — Interior, Exterior (each with a sphere *blob* moved
into the space) or Object / Collection — and presses Calculate cameras. All
enabled systems are planned in one joint selection. The algorithm below is
unchanged from 5.3.15; 5.3.16 changed the interface, let systems combine, and
added the live preview. This note records the method, the research it follows,
and its limits. Code: `auto_rig/` (`planner.py` and `geometry.py` are pure
numpy and import nothing from Blender).

## What makes a good camera set for a synthetic Splat

The add-on renders exact poses, so the photogrammetry requirement of image
overlap for feature matching does not apply. What remains:

1. **Redundancy.** Every visible surface is seen by several cameras (default 3).
2. **Angular diversity.** Those views come from genuinely different directions,
   so geometry is disambiguated and view-dependent colour is sampled. COVER
   (Coverage Optimization for Camera View Selection, 2026) and Camera Splatting
   (2025) both score camera sets by per-point observation frequency and
   directional uniformity, and find these dominate reconstruction quality.
3. **Resolution and angle.** Views not too far and not grazing. Smith et al.
   2018 ("Aerial path planning for urban scene reconstruction") model this as a
   reconstructability heuristic over visibility, distance and incidence.
4. **Viewpoint spread.** Cameras distributed through the space the Splat will
   be viewed from. Farthest View Sampling is a strong baseline in Camera
   Splatting's comparisons.

## Algorithm

This is the classic sampling-based view-planning structure (Scott, Roth &
Rivest 2003, "View planning for automated 3D object reconstruction and
inspection", ACM Computing Surveys) with a submodular coverage objective
optimised greedily (Roberts et al. 2017, "Submodular Trajectory Optimization
for Aerial 3D Scanning", ICCV).

1. **Scene proxy.** Evaluated render-visible geometry, including modifiers,
   geometry nodes and instances, is read once per unique mesh with
   `foreach_get`; instances are matrices. Surfaces are sampled area-weighted in
   world space, so non-uniform scale and huge ground triangles are exact.
2. **Free space.** Samples are voxelised. Each blob's free space is flooded
   with vectorised run sweeps: each free voxel knows its run along every axis,
   so a flood converges in a few sweeps rather than one step per voxel.
3. **Interior or exterior.** Openings are sealed progressively (dilation of the
   occupancy) up to *Seal openings*. The blob is **interior** at the smallest
   sealing where its space no longer reaches the open world; otherwise it is
   **exterior**. A leak is proven as soon as a flood touches the open world,
   and sealing levels are bisected. An exterior uses the maximum sealing grown
   back, so rooms only reachable through doors stay out of it.
4. **Multi-resolution.** Each analysis box gets the same voxel budget. When the
   scene-wide grid is too coarse to resolve a door, a blob is re-examined in a
   growing box of its own until its space is enclosed. Exterior *Nearby* builds
   a box around the cluster of touching objects nearest the blob, plus its
   immediate neighbours. A large flat ground and an enclosing shell (sky dome)
   are recognised and kept from dominating.
5. **Targets.** Surface samples bordering each region, thinned evenly. Each
   region gets an equal share of targets and objective, so a small room is
   never outvoted by a large exterior.
6. **Candidates.** Viewpoints spread evenly through each region, clear of
   surfaces, and within a region-specific distance band of its targets.
7. **Visibility.** Per candidate, a depth cube map is ray-marched through the
   occupancy grid with a Chebyshev clearance field, so rays stride through open
   air. Targets near a pixel's first hit are visible and targets far behind it
   hidden; the uncertain band in between is resolved by exact per-target rays.
   A yaw/pitch histogram of visible targets, box-filtered to the field of view
   and non-maximum suppressed, proposes each candidate's best headings.
8. **Selection.** Lazy-greedy maximisation of
   `Σ_t w_t · min(K, Σ_views q · novelty) / K`, where
   *q* = incidence × distance × frame-position quality and
   *novelty* reduces the gain of a direction within the diversity angle of one
   already observed at that target. Gains shrink monotonically, so lazy
   evaluation stays exact. A farthest-view factor spreads positions. Selection
   stops when surfaces are covered or at *Max Cameras*.
9. **Output.** Cameras are ordered along a short path (nearest neighbour +
   2-opt), use the global lens/sensor/clipping, have zero roll (world up), and
   enter the render queue in *SplatGen Auto Rig*.

## Systems (5.3.16)

- Interior / Exterior systems are blobs with an explicit mode, so step 3's
  classification is forced to the user's choice; the multi-resolution and
  leak handling below still apply.
- Object / Collection systems get their own context around the objects. Their
  free space is flooded from beside the objects themselves, so an object in a
  room is orbited from inside the room. Their targets are only those
  objects' surfaces; they are added on top of any room or exterior targets.
- Each system has its own **Views per surface** (K); the saturating objective
  uses the K of each surface's system, so raising it on an object system
  earns that object more distinct views.
- A `visual` callback receives subsampled snapshots (surfaces, flooded voxels,
  candidates, each accepted camera, final coverage) for the live viewport
  preview (`auto_rig/preview.py`). It uses its own random stream and draws on
  the main thread from cached batches; plans are identical with it on or off.

## Per-system settings and coverage (5.3.17)

- Clearance, seal size, height range and cameras-below-ground are carried by
  each system (defaults filled from the request). Sealing is applied per call
  on shared analysis grids, so systems with different seal sizes still share
  one voxelisation. Quality, lens and the total camera limit stay global.
- A system's **Cameras** value caps it during the joint lazy-greedy selection
  and then tops it up with its own best remaining views; budget for these is
  reserved within the total limit.
- **Camera coverage** (`auto_rig/coverage.py`) reuses the same grids and
  marcher: each queued camera renders a coarse depth image through the voxel
  scene at its own intrinsics; per surface point it accumulates view count,
  resultant viewing direction (angular spread = 2·acos(|Σd|/n)) and best
  metres-per-pixel. Problem areas group Unseen/Weak points by proximity and
  surface orientation, tiled to camera-sized pieces; fix cameras are chosen
  from a Fibonacci sphere of directions around the area normal, inside the
  cameras' own space with a clear line of sight, ≥30° apart.

## Narrow passages, per-system stop and coverage scope (5.3.18)

- **Completing a sealed space.** Sealing at level *r* closes doors and every
  passage narrower than 2*r* voxels. The old grow-back of *r − 1* voxels ended
  a tunnel at its mouth. Now the space is flooded again through free space
  sealed only to ~0.15 m (level *n*), blocked by the *other* sealed spaces
  grown back with a cube (Chebyshev) dilation so their floor-wall corners are
  claimed too; the result is grown back *n − 1* voxels. Tunnels, closets and
  doorways return; the next room and the outside stay out; the light sealing
  keeps the flood out of hollow solids whose sampling has pinholes.
- **Candidates in tight spaces.** Besides voxels farther than the clearance
  from every surface, local maxima of the Chebyshev clearance field at least
  0.25 m from surfaces are eligible: the middle line of a passage.
- **Interiors in their own grid** whenever that grid is materially finer than
  the scene's; a passage that runs out of the box widens it (up to twice).
  If the Doorway Width cannot enclose a room, the smallest sealing that can
  (up to 12 levels) is used with a warning instead of a blob-sized sphere.
- **Stop per system.** Each system has its own first gain and covered share;
  stop-when-covered systems finish independently once their marginal value
  drops below 1.5% of their first (and ≥97% covered, or below 0.4%). Counted
  systems with stop off are topped up to exactly their count; budget for
  counted systems is reserved until they finish.
- **Coverage scope.** The judged space is found like the rig's: level-1
  reachability from the cameras; at the doorway sealing level the outside is
  the flood from the open grid faces. Cameras in the outside make it count;
  cameras in rooms make every reachable enclosed space count; both are
  completed through narrow passages. Surfaces off that space are tested too
  and kept when a camera sees them. Outside, unseen surfaces beyond
  4 × the median view distance, and bare ground farther than 1.5 × that from
  any structure, are out of reach.
- **Coverage visibility.** Closed meshes (≥98% of edges shared by two
  triangles) are visible from their front only, so thin walls are not seen
  through; open sheets are two-sided and oriented toward the judged space.
  Rays from cameras outside the analysis box start where they enter it. On the
  tunnel test scene 96-100% of claimed-visible points were confirmed by exact
  Blender ray casts; the misses are faces buried in other geometry.
- **Redundant cameras** are removed greedily, least useful first, only when
  every surface they see keeps ≥ K views and no less angular spread than it
  needs; the counts update after each removal, so the whole list is jointly
  safe.

## Object spaces, photographable surfaces (5.3.19)

- **Which space an object is captured from.** Every sealed space found beside
  an Object system's surface is grown back to the surface through free space
  only (cube steps) and scores one vote per surface sample whose face opens
  into it, taken 2 voxels in front along the normal. The best space wins;
  another joins only at ≥ 90% of its votes. A face buried in a wall casts no
  vote, and no vote reaches through a slab, so a sofa sinking an inch through
  the floor or a cabinet poking through a wall is captured from its room.
- **Growing spaces back.** A sealed space and all other sealed spaces now grow
  back to the walls together, one cube step at a time, each voxel going to the
  first to reach it; the narrow-passage flood is blocked by the others' share.
  A room's space stops at its doors.
- **Sealing depth** is capped at 24 voxel levels (was 12), so Doorway Width
  holds on fine grids.
- **Coverage judges photographable surfaces only.** Removed: faces in contact
  with or within 0.6 voxel of a parallel surface in front of them (picture
  backs, walls behind pictures, floors under furniture) - their cell takes a
  visible sample instead; solid faces whose front does not face the judged
  space (outer faces of thin walls) unless photographed; surfaces glimpsed in
  an outside that is not judged. The outside is judged only when at least
  5% of the cameras stand in it. Cameras are assigned to spaces by growing
  the outside and the rooms back together. The analysis box extends at least
  1.2 × Doorway Width beyond the structures.

## Measured behaviour (Blender 5.3 Alpha, this machine)

| Scene | Quality | Result | Time |
|---|---|---|---|
| House, open door + window, furniture; blobs inside + outside | Standard | interior classified Interior, exterior excludes it; 83–97% covered | 8–9 s |
| Three rooms with 0.9 m doors + street, inside a 1,500-building city | Standard | 3 separate interiors at 0.2 m voxels, exterior at 0.11 m around the apartment | 11–12 s |
| 5,000 objects + 55,000 instances + 2.6 M-triangle mesh | Standard | scene read in 2.7 s (15 s before optimisation) | 10–11 s |
| Single object, with and without ground | Standard | full orbit; no cameras below a ground plane | 6–8 s |

"Covered" is the weighted fraction of observable surface with at least 0.8 × K
quality-weighted, distinct views. It is a planning estimate. **Check coverage**
remains the independent diagnostic.

## Limits

- An unglazed opening larger than *Doorway Width* joins a room to the outside
  unless a smaller sealing encloses it (then a warning names the size used).
  Glass modelled as geometry blocks correctly.
- A chamber reached only through a narrow tunnel is its own room: give it its
  own Interior system. The tunnel itself belongs to both.
- Camera coverage uses one grid for the whole check; in a city-sized scene it
  is coarse (about 0.5 m), so small interior detail is judged coarsely.
- A room without a ceiling is open to the sky; force the blob to Interior and
  scale it to cover the room (its radius bounds the room).
- Two Interior systems in one space merge; add one per room to scan every room.
- Very large exteriors are coarse at a fixed budget; use Nearby, Limit to
  Radius or a Scene Geometry collection.
- Planning is geometric. It does not know which materials are reflective,
  which Camera Splatting shows benefit from denser angular sampling; raise
  *Views per Surface* for such scenes.
