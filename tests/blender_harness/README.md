# Headless add-on test harness

Runs the real add-on inside Blender's Python module (`bpy` from PyPI), builds a
dataset from a small test scene, and checks the output.

## Setup (once)

```
python3.11 -m venv .bpyenv
.bpyenv/bin/pip install bpy==5.0.1 "numpy<2"
.bpyenv/bin/pip install --no-deps OpenImageIO   # Blender ships it; pip bpy does not
```

The add-on targets Blender 5.3; `run_build.py` copies it to a temporary folder
and relaxes the version check there (never in the source). Cycles runs on CPU.

## Run

```
# baseline: legacy only
HARNESS_SETUP=tests/blender_harness/scenarios/raw_off.py \
  .bpyenv/bin/python tests/blender_harness/run_build.py blender_addon /tmp/sg_base 3

# default build with the raw export
.bpyenv/bin/python tests/blender_harness/run_build.py blender_addon /tmp/sg_raw 3

# content checks; the second argument compares legacy output with the baseline
.bpyenv/bin/python tests/blender_harness/verify.py /tmp/sg_raw/out/SplatGen_scene/<timestamp> \
                                                    /tmp/sg_base/out/SplatGen_scene/<timestamp>
```

`run_build.py <addon> <work_dir> [cameras]` also prints `SCENE STATE RESTORED`
when every temporary change to the scene was undone. `HARNESS_SETUP` runs a
script before the build and `HARNESS_POST` after it (both see `bpy`, `os`,
`work`, `snapshot`); see `scenarios/`.

`verify.py` checks, among others: light passes sum to the noisy image, position
equals back-projected depth, raw depth equals legacy depth, normals are unit
length, ids match `id_map.json`, material values match the test materials,
clay albedo is 0.8, PLY and voxels are consistent, cameras sit in free space,
and (with a baseline) that `Dataset(Default)` is identical to the legacy build.

Blender prints `Not freed memory blocks: 12-14` at exit when value-type shader
AOVs were rendered through the legacy capture; the count does not grow with the
number of views.
