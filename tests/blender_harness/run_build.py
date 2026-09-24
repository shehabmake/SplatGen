"""Headless end-to-end build harness for the SplatGen add-on on pip bpy.

usage: python run_build.py <addon_src> <work_dir> [n_cameras]
"""
import sys, os, shutil, math, time, importlib, json
src, work = sys.argv[1], sys.argv[2]
ncam = int(sys.argv[3]) if len(sys.argv) > 3 else 3
shutil.rmtree(work, ignore_errors=True)
pkg_root = os.path.join(work, "pkgs"); os.makedirs(pkg_root)
pkg = os.path.join(pkg_root, "splatgen_prepare")
shutil.copytree(src, pkg, ignore=shutil.ignore_patterns("__pycache__"))
# pip bpy is 5.0.x; the add-on pins 5.3. Relax the guard in the test copy only.
init = os.path.join(pkg, "__init__.py")
s = open(init).read().replace("if bpy.app.version[:2] != (5, 3):", "if bpy.app.version[:2] < (5, 0):")
open(init, "w").write(s)
sys.path.insert(0, pkg_root)
import bpy
addon = importlib.import_module("splatgen_prepare")
addon.register()
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 4
scene.cycles.device = 'CPU'
scene.render.resolution_x = 80; scene.render.resolution_y = 60; scene.render.resolution_percentage = 100
# Scene: cube with principled material, a textured plane, a sphere with glass, a light.
cube = bpy.data.objects["Cube"]
m = cube.data.materials[0]
p = next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
p.inputs["Base Color"].default_value = (0.8, 0.2, 0.1, 1)
p.inputs["Roughness"].default_value = 0.35
p.inputs["Metallic"].default_value = 0.25
noise = m.node_tree.nodes.new("ShaderNodeTexNoise")
m.node_tree.links.new(noise.outputs["Fac"], p.inputs["Coat Weight"])
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -1))
plane = bpy.context.active_object; plane.name = "Floor"
pm = bpy.data.materials.new("FloorMat"); plane.data.materials.append(pm)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.6, location=(1.8, 0.5, -0.4))
sph = bpy.context.active_object; sph.name = "GlassBall"
gm = bpy.data.materials.new("Glass")
gp = next(n for n in gm.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
gp.inputs["Transmission Weight"].default_value = 1.0
gp.inputs["Emission Strength"].default_value = 0.5
gp.inputs["Emission Color"].default_value = (0.1, 0.3, 1.0, 1)
sph.data.materials.append(gm)
bpy.ops.mesh.primitive_monkey_add(location=(-1.8, 0.4, -0.2))
bpy.context.active_object.name = "NoMaterialMonkey"
cfg = scene.SCENERAY_SPLAT
# Cameras on a ring.
for i in range(ncam):
    a = 2 * math.pi * i / ncam
    bpy.ops.object.camera_add(location=(7 * math.cos(a), 7 * math.sin(a), 3))
    cam = bpy.context.active_object
    cam.name = f"TestCam{i}"
    d = -cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    item = cfg.camera_queue.add(); item.camera = cam
old_default_cam = bpy.data.objects.get("Camera")
blend = os.path.join(work, "scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend)

def snapshot():
    """Everything the raw export is allowed to touch only temporarily."""
    sc = bpy.context.scene; vl = bpy.context.view_layer
    state = {
        "engine": sc.render.engine, "samples": sc.cycles.samples,
        "res": (sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage),
        "camera": sc.camera.name if sc.camera else None,
        "comp_group": sc.compositing_node_group.name if sc.compositing_node_group else None,
        "use_compositing": sc.render.use_compositing,
        "film_transparent": sc.render.film_transparent,
        "override": vl.material_override.name if vl.material_override else None,
        "aovs": [(a.name, a.type) for a in vl.aovs],
        "flags": {a: getattr(vl, a) for a in dir(vl) if a.startswith("use_pass")},
        "cflags": {a: getattr(vl.cycles, a) for a in dir(vl.cycles) if a.startswith("use_pass") or a == "denoising_store_passes"},
        "pass_index": {o.name: o.pass_index for o in bpy.data.objects},
        "mat_pass_index": {m.name: m.pass_index for m in bpy.data.materials},
        "hide_render": {o.name: o.hide_render for o in bpy.data.objects},
        "mat_nodes": {m.name: sorted(n.name for n in m.node_tree.nodes) for m in bpy.data.materials if m.node_tree},
        "mat_links": {m.name: len(m.node_tree.links) for m in bpy.data.materials if m.node_tree},
        "objects": sorted(o.name for o in bpy.data.objects),
        "cameras": sorted(c.name for c in bpy.data.cameras),
        "scenes": sorted(s.name for s in bpy.data.scenes),
        "materials": sorted(m.name for m in bpy.data.materials),
        "node_groups": sorted(g.name for g in bpy.data.node_groups),
        "clip": {o.name: (o.data.clip_start, o.data.clip_end) for o in bpy.data.objects if o.type == 'CAMERA'},
    }
    return state
cfg = bpy.context.scene.SCENERAY_SPLAT
cfg.output_dir = os.path.join(work, "out")
extra = os.environ.get("HARNESS_SETUP")
if extra:
    exec(open(extra).read())
before = snapshot()
t0 = time.time()
res = bpy.ops.splatray.build_dataset('EXEC_DEFAULT')
print("BUILD RESULT", res, f"{time.time()-t0:.1f}s")
print("RENDER STATUS", cfg.render_status)
print("POINT STATUS", cfg.point_status)
after = snapshot()
changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
print("SCENE STATE RESTORED" if not changed else f"SCENE STATE CHANGED {changed}")
post = os.environ.get("HARNESS_POST")
if post:
    exec(open(post).read())
for dirpath, dirs, files in sorted(os.walk(os.path.join(work, "out"))):
    rel = os.path.relpath(dirpath, work)
    print("DIR", rel, len(files), sorted(files)[:4])
addon.unregister()
print("UNREGISTER OK")
