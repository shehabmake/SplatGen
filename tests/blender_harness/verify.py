"""Content checks for a raw build. usage: verify.py <raw_build_dir> [<baseline_build_dir>]"""
import sys, os, json, hashlib, glob
import numpy as np, OpenImageIO as oiio
build = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else None
R = os.path.join(build, "Dataset(Raw)")
fails = []
def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)
def exr(rel, stem="frame_0000"):
    p = os.path.join(R, rel, stem + ".exr") if not rel.endswith(".exr") else os.path.join(R, rel)
    i = oiio.ImageInput.open(p); a = np.asarray(i.read_image(oiio.FLOAT)); i.close()
    return a
man = json.load(open(os.path.join(R, "manifest.json")))
ids = json.load(open(os.path.join(R, "ids/id_map.json")))
cams = json.load(open(os.path.join(R, "cameras/cameras.json")))
oid = {e["name"]: e["id"] for e in ids["objects"]}; mid = {e["name"]: e["id"] for e in ids["materials"]}
print("id map", oid, mid)
for f in cams["frames"]:
    stem = f["stem"]
    comb = exr("motion_denoising/noisy_image", stem)[..., :3]
    g = lambda k: exr("appearance/lighting/" + k, stem)[..., :3]
    recon = (g("diffuse_direct") + g("diffuse_indirect")) * g("diffuse_color") \
          + (g("glossy_direct") + g("glossy_indirect")) * g("glossy_color") \
          + (g("transmission_direct") + g("transmission_indirect")) * g("transmission_color") \
          + g("volume_direct") + g("volume_indirect") + g("emission") + g("environment")
    err = np.abs(comb - recon)
    check(np.percentile(err, 99) < 0.02 * max(1e-3, comb.max()), f"{stem}: noisy image == sum of light passes (p99 err {np.percentile(err,99):.4g}, max comb {comb.max():.3g})")
    depth = exr("geometry/depth", stem)[..., 0]
    legacy_depth = np.asarray(oiio.ImageInput.open(os.path.join(build, "Dataset(Default)/depth", stem + ".exr")).read_image(oiio.FLOAT))[..., 0]
    check(np.array_equal(depth, legacy_depth), f"{stem}: raw depth identical to legacy depth")
    pos = exr("geometry/position", stem)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    fg = depth < 1e9
    x = (u + 0.5 - f["cx"]) / f["fx"] * depth; y = (v + 0.5 - f["cy"]) / f["fy"] * depth
    Xc = np.stack([x, y, depth, np.ones_like(depth)], -1)
    Xw = Xc @ np.array(f["c2w_opencv"]).T
    perr = np.linalg.norm(Xw[..., :3] - pos, axis=-1)[fg]
    check(np.median(perr) < 1e-3 * np.median(depth[fg]), f"{stem}: position == back-projected depth (median err {np.median(perr):.2e})")
    n = exr("geometry/normal", stem); nl = np.linalg.norm(n, axis=-1)[fg]
    check(abs(np.median(nl) - 1) < 0.02, f"{stem}: normals unit length (median {np.median(nl):.3f})")
    tn = exr("geometry/true_normal", stem); cov = tn[..., 3] > 0.99
    tnl = np.linalg.norm(tn[..., :3], axis=-1)[cov]
    check(cov.any() and abs(np.median(tnl) - 1) < 0.02, f"{stem}: true normals unit length where covered ({cov.sum()} px)")
    o = exr("ids/object_id", stem)[..., 0]; m = exr("ids/material_id", stem)[..., 0]
    check(np.all(o == np.round(o)) and set(np.unique(o).astype(int)) <= set(oid.values()) | {0}, f"{stem}: object ids integral & in map {sorted(set(np.unique(o).astype(int)))}")
    check(np.all(m == np.round(m)) and set(np.unique(m).astype(int)) <= set(mid.values()) | {0}, f"{stem}: material ids integral & in map {sorted(set(np.unique(m).astype(int)))}")
    def at(name, key, ch=None):
        mask = o == oid[name]
        # interior pixels only (ids are not AA'd, AOVs are)
        from numpy.lib.stride_tricks import sliding_window_view as sw
        pad = np.pad(mask, 1); inner = mask & pad[:-2,1:-1] & pad[2:,1:-1] & pad[1:-1,:-2] & pad[1:-1,2:]
        a = exr("material/" + key, stem)
        a = a[..., :3] if a.shape[-1] == 4 else a[..., 0]
        return a[inner], inner.sum()
    if (o == oid["Cube"]).sum() > 20:
        bc, n_ = at("Cube", "base_color"); check(np.allclose(np.median(bc, 0), [0.8, 0.2, 0.1], atol=0.01), f"{stem}: cube base color {np.median(bc,0)}")
        r, _ = at("Cube", "roughness"); check(abs(np.median(r) - 0.35) < 0.01, f"{stem}: cube roughness {np.median(r):.3f}")
        mt, _ = at("Cube", "metallic"); check(abs(np.median(mt) - 0.25) < 0.01, f"{stem}: cube metallic {np.median(mt):.3f}")
        c, _ = at("Cube", "coat"); check(c.std() > 0.01 and 0 <= c.min() and c.max() <= 1, f"{stem}: cube coat is the noise texture (std {c.std():.3f})")
        mv, _ = at("Cube", "material_valid"); check(abs(np.median(mv) - 1) < 1e-3, f"{stem}: cube material_valid 1")
    if (o == oid["GlassBall"]).sum() > 20:
        t, _ = at("GlassBall", "transmission"); check(abs(np.median(t) - 1) < 0.01, f"{stem}: glass transmission {np.median(t):.3f}")
        es, _ = at("GlassBall", "emission_strength"); check(abs(np.median(es) - 0.5) < 0.01, f"{stem}: glass emission strength {np.median(es):.3f}")
    if (o == oid["NoMaterialMonkey"]).sum() > 20:
        mv, _ = at("NoMaterialMonkey", "material_valid"); check(np.median(mv) < 1e-3, f"{stem}: material-less monkey material_valid 0")
    clay = exr("appearance/clay/diffuse_color", stem)[..., :3]
    cm = (o > 0)
    check(np.allclose(np.median(clay[cm], 0), 0.8, atol=0.01), f"{stem}: clay diffuse color 0.8 ({np.median(clay[cm],0)})")
    uv = exr("geometry/uv", stem); check(uv[..., :2].min() >= -1e-4 and uv[..., :2].max() <= 1 + 1e-4, f"{stem}: uv in [0,1]")
# legacy identity
if base:
    def files(root):
        out = {}
        for d, _, fs in os.walk(root):
            for n in fs:
                p = os.path.join(d, n); out[os.path.relpath(p, root)] = p
        return out
    for sub in ("Dataset(Default)",):
        a, b = files(os.path.join(base, sub)), files(os.path.join(build, sub))
        check(set(a) == set(b), f"legacy {sub} file list identical ({len(a)} files)")
        diff = []
        for k in sorted(set(a) & set(b)):
            if k.endswith(".exr") or k.endswith(".png"):
                ia = np.asarray(oiio.ImageInput.open(a[k]).read_image(oiio.FLOAT)); ib = np.asarray(oiio.ImageInput.open(b[k]).read_image(oiio.FLOAT))
                if not np.array_equal(ia, ib): diff.append(k)
            else:
                ta = [l for l in open(a[k]) if "generation" not in l]; tb = [l for l in open(b[k]) if "generation" not in l]
                if ta != tb: diff.append(k)
        check(not diff, f"legacy {sub} contents identical to baseline build {diff[:5]}")
# colmap copy
for n in ("cameras.txt", "images.txt", "points3D.txt"):
    check(open(os.path.join(R, "cameras/colmap", n), "rb").read() == open(os.path.join(build, "Dataset(Default)", n), "rb").read(), f"colmap/{n} identical to legacy")
# ply
ply = open(os.path.join(R, "scene/scene_mesh.ply"), "rb").read()
hdr = ply[:ply.index(b"end_header\n") + 11].decode()
nv = int(hdr.split("element vertex ")[1].split()[0]); nf = int(hdr.split("element face ")[1].split()[0])
check(len(ply) == len(hdr) + nv * 24 + nf * (1 + 12 + 8), f"ply size consistent ({nv} verts, {nf} faces)")
fd = np.frombuffer(ply[len(hdr) + nv * 24:], dtype=[("n","u1"),("v","<i4",(3,)),("o","<i4"),("m","<i4")])
check(fd["v"].max() < nv and set(np.unique(fd["o"])) <= set(oid.values()), f"ply faces valid, object ids {sorted(set(np.unique(fd['o'])))}")
vox = np.load(os.path.join(R, "scene/collision_voxels.npz"))
lab = vox["labels"]; print("voxels", lab.shape, np.bincount(lab.ravel(), minlength=3))
check((lab == 2).sum() > 0 and (lab == 1).sum() > 0, "voxels have surface and enclosed cells (cube/monkey interiors)")
# cameras in free space
for f in cams["frames"]:
    idx = np.floor((np.array(f["position"]) - vox["origin"]) / vox["voxel_size"]).astype(int)
    check(lab[tuple(idx)] == 0, f"camera {f['camera_name']} voxel is free")
if os.path.exists(os.path.join(R, "environment/world/world_equirect.exr")):
  w = exr("environment/world/world_equirect.exr"); print("world", w.shape, w[..., :3].mean((0, 1)))
  check(w.shape[1] == 2 * w.shape[0], "world equirect is 2:1")
if os.path.exists(os.path.join(R, "environment/probes/probes.json")):
  pr = json.load(open(os.path.join(R, "environment/probes/probes.json")))
  print("probes", [(p["id"], np.round(p["position"], 2).tolist()) for p in pr["probes"]])
  d = exr("environment/probes/probe_000/distance.exr")[..., 0]
  check(np.isfinite(d).all() and d.min() > 0, f"probe distance positive (min {d.min():.3f}, median {np.median(d):.3f})")
print("\n%d FAILURES" % len(fails) if fails else "\nALL CHECKS PASSED")
