import json, glob
root = glob.glob(os.path.join(work, "out/*/*/Dataset(Raw)"))[0]
m = json.load(open(os.path.join(root, "manifest.json")))
print("MANIFEST", m["status"], m["issues"], m["notes"])
w = json.load(open(os.path.join(root, "environment/world/world.json")))
print("WORLD", w["source_images"], w["world"]["environment_textures"][0]["mapping"])
p = json.load(open(os.path.join(root, "environment/probes/probes.json")))
print("PROBES", [(x["source"], x["name"], x["position"]) for x in p["probes"]])
mi = json.load(open(os.path.join(root, "scene/scene_mesh.json")))
print("MESH OBJECTS", [(o["name"], o["instance"], o["type"], o["triangle_count"]) for o in mi["objects"]])
print("MATERIAL REPORT", {k: v.get("shader") for k, v in m["materials"].items()})
