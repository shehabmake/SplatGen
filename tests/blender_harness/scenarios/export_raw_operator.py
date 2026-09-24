import glob
builds = glob.glob(os.path.join(work, "out/*/*/"))
print("RAW BEFORE OPERATOR", [os.path.isdir(os.path.join(b, "Dataset(Raw)")) for b in builds])
bpy.context.scene.splatgen_raw.enabled = True
before2 = snapshot()
print("EXPORT OP", bpy.ops.splatgen.export_raw_data('EXEC_DEFAULT'))
after2 = snapshot()
print("SCENE STATE 2", "RESTORED" if before2 == after2 else {k: (before2[k], after2[k]) for k in before2 if before2[k] != after2[k]})
