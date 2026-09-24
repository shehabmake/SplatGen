import glob, time
time.sleep(1.1)  # new timestamped folder
print("SECOND BUILD", bpy.ops.splatray.build_dataset('EXEC_DEFAULT'))
builds = sorted(glob.glob(os.path.join(work, "out/*/*/")))
print("BUILDS", [os.path.basename(b.rstrip('/')) for b in builds])
