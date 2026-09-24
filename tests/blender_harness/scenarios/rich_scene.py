import math, numpy as np
sc = bpy.context.scene
# HDRI: a synthetic equirect image saved to disk (warm +X, cool elsewhere)
H, W = 64, 128
img = bpy.data.images.new("synthetic_sky", W, H, float_buffer=True)
v, u = np.mgrid[0:H, 0:W]
phi = (0.5 - (u + 0.5) / W) * 2 * math.pi
px = np.zeros((H, W, 4), np.float32)
px[..., 0] = 0.2 + 2.0 * np.clip(np.cos(phi), 0, None); px[..., 1] = 0.3; px[..., 2] = 0.5 + ((H - 1 - v) / H); px[..., 3] = 1
img.pixels.foreach_set(px[::-1].ravel())
hdr_path = os.path.join(work, "sky.exr")
img.filepath_raw = hdr_path; img.file_format = 'OPEN_EXR'; img.save()
world = sc.world; nt = world.node_tree
bg = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBackground")
env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(hdr_path)
mp = nt.nodes.new("ShaderNodeMapping"); tc = nt.nodes.new("ShaderNodeTexCoord")
nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
nt.links.new(env.outputs["Color"], bg.inputs["Color"])
# fog box (volume only)
bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, -2.5, 0)); fog = bpy.context.active_object; fog.name = "FogBox"
fm = bpy.data.materials.new("Fog"); fnt = fm.node_tree
for n in list(fnt.nodes): fnt.nodes.remove(n)
out = fnt.nodes.new("ShaderNodeOutputMaterial"); vol = fnt.nodes.new("ShaderNodeVolumePrincipled")
fnt.links.new(vol.outputs[0], out.inputs["Volume"]); fog.data.materials.append(fm)
# emissive panel
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 2.5, 0.5), rotation=(math.pi/2, 0, 0)); pan = bpy.context.active_object; pan.name = "EmissivePanel"
em = bpy.data.materials.new("Emitter"); ent = em.node_tree
for n in list(ent.nodes): ent.nodes.remove(n)
o2 = ent.nodes.new("ShaderNodeOutputMaterial"); e = ent.nodes.new("ShaderNodeEmission"); e.inputs["Strength"].default_value = 5
ent.links.new(e.outputs[0], o2.inputs["Surface"]); pan.data.materials.append(em)
# user light probe sphere
bpy.ops.object.lightprobe_add(type='SPHERE', location=(0, 0, 2)); bpy.context.active_object.name = "UserProbe"
# collection instance of a small cone, not linked to the scene itself
col = bpy.data.collections.new("Props")
bpy.ops.mesh.primitive_cone_add(radius1=0.3, depth=0.6, location=(0, 0, 0)); cone = bpy.context.active_object; cone.name = "ConeProto"
for c in list(cone.users_collection): c.objects.unlink(cone)
col.objects.link(cone)
inst = bpy.data.objects.new("ConeInstance", None); inst.instance_type = 'COLLECTION'; inst.instance_collection = col
inst.location = (-1.5, -1.5, -0.7); sc.collection.objects.link(inst)
# text object
bpy.ops.object.text_add(location=(1.5, -1.8, -0.95)); bpy.context.active_object.name = "Label"
bpy.ops.wm.save_mainfile()
