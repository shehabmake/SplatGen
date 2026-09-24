# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""The on-disk contract of the raw dataset, in one place.

The raw dataset lives beside the legacy output of a build and never touches
it::

    <build timestamp>/
        Dataset(Default)/      legacy portable dataset (unchanged)
        sg_metadata/           legacy SplatGen metadata (unchanged)
        Dataset(Raw)/          everything below
            manifest.json
            scene_description.json
            appearance/        combined, lighting split, clay
            geometry/          position, depth, normals, uv, ...
            material/          per-pixel material parameters
            ids/               object / material ids + id_map.json
            motion_denoising/  motion vectors, denoising albedo / normal
            cameras/           colmap/ copy + cameras.json
            scene/             scene mesh + collision voxels
            environment/       world render + reflection probes

Every per-view raster is ``<folder>/frame_NNNN.exr`` with the same frame
number as the legacy ``Dataset(Default)/images/frame_NNNN.*`` image, so any
pass lines up with the legacy RGB and with ``images.txt`` by name alone.

``PASSES`` is the single registry of per-view render passes. Capture,
publishing, completeness checks and the manifest are all driven from it, so
adding a pass is one entry here and nothing else.
"""

from pathlib import Path

SCHEMA = "splatgen-raw-dataset-v1"
RAW_FOLDER = "Dataset(Raw)"
#: Temporary capture folder inside the raw root; removed when a stage ends.
WORK_FOLDER = ".work"
MANIFEST = "manifest.json"
SCENE_DESCRIPTION = "scene_description.json"
README = "README.txt"

APPEARANCE = "appearance"
GEOMETRY = "geometry"
MATERIAL = "material"
IDS = "ids"
MOTION = "motion_denoising"
CAMERAS = "cameras"
SCENE = "scene"
ENVIRONMENT = "environment"

ID_MAP = f"{IDS}/id_map.json"
CAMERAS_JSON = f"{CAMERAS}/cameras.json"
COLMAP_FOLDER = f"{CAMERAS}/colmap"
SCENE_MESH = f"{SCENE}/scene_mesh.ply"
SCENE_MESH_INFO = f"{SCENE}/scene_mesh.json"
VOXELS = f"{SCENE}/collision_voxels.npz"
VOXELS_INFO = f"{SCENE}/collision_voxels.json"
WORLD_FOLDER = f"{ENVIRONMENT}/world"
WORLD_RENDER = f"{WORLD_FOLDER}/world_equirect.exr"
WORLD_INFO = f"{WORLD_FOLDER}/world.json"
PROBES_FOLDER = f"{ENVIRONMENT}/probes"
PROBES_INFO = f"{PROBES_FOLDER}/probes.json"

#: Render sources. ``beauty`` passes come from the same render as the legacy
#: RGB image; ``clay`` passes come from a second render with every surface
#: replaced by a neutral diffuse material.
BEAUTY = "beauty"
CLAY = "clay"

#: Precision/codec policies. ``appearance`` follows the user's settings;
#: data passes always stay lossless and keep the precision listed here.
POLICY_APPEARANCE = "appearance"
POLICY_FLOAT = "float32"
POLICY_HALF = "half"


def _pass(key, folder, source, sockets, item, *, group, label, flags=(),
          aov=None, policy=POLICY_FLOAT, space="", description=""):
    return {
        "key": key,
        "folder": folder,
        "source": source,
        "sockets": tuple(sockets),
        # File Output item type: RGBA, VECTOR or FLOAT.
        "item": item,
        # Settings toggle that owns this pass (see properties.py).
        "group": group,
        "label": label,
        # (owner, attribute) switches that make Blender produce the pass.
        # owner is "layer" (ViewLayer) or "cycles" (ViewLayer.cycles).
        "flags": tuple(flags),
        # (aov_name, "COLOR"|"VALUE") for shader AOV passes.
        "aov": aov,
        "policy": policy,
        "space": space,
        "description": description,
    }


_L = "layer"
_C = "cycles"

PASSES = {}


def _register(*definitions):
    for definition in definitions:
        PASSES[definition["key"]] = definition


# -- 1. Rendered appearance --------------------------------------------------
_register(
    _pass("combined", f"{APPEARANCE}/combined", BEAUTY, ("Image",), "RGBA",
          group="always", label="Combined", policy=POLICY_APPEARANCE,
          space="scene-linear RGB + alpha, before compositing and view transform",
          description="The beauty render as Blender computed it, scene linear."),
)
for _key, _socket, _flag, _owner in (
    ("diffuse_direct", "Diffuse Direct", "use_pass_diffuse_direct", _L),
    ("diffuse_indirect", "Diffuse Indirect", "use_pass_diffuse_indirect", _L),
    ("diffuse_color", "Diffuse Color", "use_pass_diffuse_color", _L),
    ("glossy_direct", "Glossy Direct", "use_pass_glossy_direct", _L),
    ("glossy_indirect", "Glossy Indirect", "use_pass_glossy_indirect", _L),
    ("glossy_color", "Glossy Color", "use_pass_glossy_color", _L),
    ("transmission_direct", "Transmission Direct", "use_pass_transmission_direct", _L),
    ("transmission_indirect", "Transmission Indirect", "use_pass_transmission_indirect", _L),
    ("transmission_color", "Transmission Color", "use_pass_transmission_color", _L),
    ("volume_direct", "Volume Direct", "use_pass_volume_direct", _C),
    ("volume_indirect", "Volume Indirect", "use_pass_volume_indirect", _C),
    ("emission", "Emission", "use_pass_emit", _L),
    ("environment", "Environment", "use_pass_environment", _L),
):
    _register(_pass(
        _key, f"{APPEARANCE}/lighting/{_key}", BEAUTY, (_socket,), "RGBA",
        group="lighting", label=_socket, flags=((_owner, _flag),),
        policy=POLICY_APPEARANCE, space="scene-linear RGB",
        description=f"Cycles {_socket} light pass of the beauty render.",
    ))

_register(
    _pass("clay_combined", f"{APPEARANCE}/clay/combined", CLAY, ("Image",),
          "RGBA", group="clay", label="Clay Combined",
          policy=POLICY_APPEARANCE, space="scene-linear RGB + alpha",
          description="Scene lit as-is with every surface a neutral 0.8 grey Lambert."),
)
for _key, _socket, _flag in (
    ("clay_diffuse_direct", "Diffuse Direct", "use_pass_diffuse_direct"),
    ("clay_diffuse_indirect", "Diffuse Indirect", "use_pass_diffuse_indirect"),
    ("clay_diffuse_color", "Diffuse Color", "use_pass_diffuse_color"),
):
    _register(_pass(
        _key, f"{APPEARANCE}/clay/{_key[len('clay_'):]}", CLAY, (_socket,),
        "RGBA", group="clay", label=f"Clay {_socket}", flags=((_L, _flag),),
        policy=POLICY_APPEARANCE, space="scene-linear RGB",
        description=f"{_socket} of the clay render: lighting without albedo.",
    ))

# -- 2. Geometry & spatial data ----------------------------------------------
_register(
    _pass("position", f"{GEOMETRY}/position", BEAUTY, ("Position",), "VECTOR",
          group="geometry", label="Position", flags=((_L, "use_pass_position"),),
          space="Blender world space, metres (scene units)",
          description="World-space position of the first visible surface."),
    _pass("depth", f"{GEOMETRY}/depth", BEAUTY, ("Depth", "Z"), "FLOAT",
          group="geometry", label="Depth", flags=((_L, "use_pass_z"),),
          space="camera-space planar Z distance; background = 1e10",
          description="Metric camera depth, identical to legacy Dataset(Default)/depth."),
    _pass("normal", f"{GEOMETRY}/normal", BEAUTY, ("Normal",), "VECTOR",
          group="geometry", label="Normal", flags=((_L, "use_pass_normal"),),
          policy=POLICY_HALF, space="Blender world space, shading normal (normal maps applied)",
          description="World-space shading normal."),
    _pass("true_normal", f"{GEOMETRY}/true_normal", BEAUTY, ("sg_true_normal",),
          "RGBA", group="geometry", label="True Normal",
          aov=("sg_true_normal", "COLOR"), policy=POLICY_HALF,
          space="Blender world space geometric normal in RGB; A = AOV coverage",
          description="Geometric (face) normal without bump or normal maps."),
    _pass("uv", f"{GEOMETRY}/uv", BEAUTY, ("UV",), "VECTOR", group="geometry",
          label="UV", flags=((_L, "use_pass_uv"),),
          space="X,Y = active render UV map, Z = coverage",
          description="Texture coordinates of the active render UV map."),
    _pass("object_coords", f"{GEOMETRY}/object_coords", BEAUTY,
          ("sg_object_coords",), "RGBA", group="geometry",
          label="Object Coordinates", aov=("sg_object_coords", "COLOR"),
          space="object-local position in RGB; A = AOV coverage",
          description="Position in the owning object's local space."),
    _pass("pointiness", f"{GEOMETRY}/pointiness", BEAUTY, ("sg_pointiness",),
          "FLOAT", group="geometry", label="Pointiness",
          aov=("sg_pointiness", "VALUE"), policy=POLICY_HALF,
          space="0..1 Cycles curvature approximation (0.5 = flat)",
          description="Cycles Geometry Pointiness (convex > 0.5 > concave)."),
    _pass("backfacing", f"{GEOMETRY}/backfacing", BEAUTY, ("sg_backfacing",),
          "FLOAT", group="geometry", label="Backfacing",
          aov=("sg_backfacing", "VALUE"), policy=POLICY_HALF,
          space="1 = the camera sees the back of the face",
          description="Geometry Backfacing output, pixel-filtered."),
    _pass("ambient_occlusion", f"{GEOMETRY}/ambient_occlusion", BEAUTY,
          ("Ambient Occlusion", "AO"), "RGBA", group="geometry",
          label="Ambient Occlusion", flags=((_L, "use_pass_ambient_occlusion"),),
          policy=POLICY_HALF, space="0..1, uses the World AO distance",
          description="Cycles ambient occlusion pass."),
)

# -- 3. Material properties (shader AOVs injected for the render) ------------
MATERIAL_CHANNELS = (
    # key, aov type, Principled inputs (newest name first), label
    ("base_color", "COLOR", ("Base Color",), "Base Color"),
    ("roughness", "VALUE", ("Roughness",), "Roughness"),
    ("metallic", "VALUE", ("Metallic",), "Metallic"),
    ("specular", "VALUE", ("Specular IOR Level", "Specular"), "Specular"),
    ("ior", "VALUE", ("IOR",), "IOR"),
    ("anisotropic", "VALUE", ("Anisotropic",), "Anisotropic"),
    ("coat", "VALUE", ("Coat Weight", "Clearcoat"), "Coat"),
    ("sheen", "VALUE", ("Sheen Weight", "Sheen"), "Sheen"),
    ("transmission", "VALUE", ("Transmission Weight", "Transmission"), "Transmission"),
    ("subsurface", "VALUE", ("Subsurface Weight", "Subsurface"), "Subsurface"),
    ("alpha", "VALUE", ("Alpha",), "Alpha"),
    ("emission_color", "COLOR", ("Emission Color", "Emission"), "Emission Color"),
    ("emission_strength", "VALUE", ("Emission Strength",), "Emission Strength"),
)
for _key, _type, _inputs, _label in MATERIAL_CHANNELS:
    _register(_pass(
        _key, f"{MATERIAL}/{_key}", BEAUTY, (f"sg_{_key}",),
        "RGBA" if _type == "COLOR" else "FLOAT", group="material",
        label=_label, aov=(f"sg_{_key}", _type), policy=POLICY_HALF,
        space=("linear RGB; A = AOV coverage" if _type == "COLOR"
               else "raw shader input value"),
        description=f"Principled BSDF '{_inputs[0]}' input, textures evaluated.",
    ))
_register(
    _pass("material_valid", f"{MATERIAL}/material_valid", BEAUTY,
          ("sg_material_valid",), "FLOAT", group="material",
          label="Material Validity", aov=("sg_material_valid", "VALUE"),
          policy=POLICY_HALF,
          space="1 = Principled BSDF, 0.5 = other BSDF, 0.25 = viewport fallback, 0 = none",
          description="How trustworthy the material passes are at each pixel."),
)

# -- 4. Object & material identification --------------------------------------
_register(
    _pass("object_id", f"{IDS}/object_id", BEAUTY, ("Object Index", "IndexOB"),
          "FLOAT", group="ids", label="Object ID",
          flags=((_L, "use_pass_object_index"),),
          space="integer id stored as float; 0 = background; see ids/id_map.json",
          description="Per-pixel object id (not anti-aliased)."),
    _pass("material_id", f"{IDS}/material_id", BEAUTY,
          ("Material Index", "IndexMA"), "FLOAT", group="ids",
          label="Material ID", flags=((_L, "use_pass_material_index"),),
          space="integer id stored as float; 0 = none; see ids/id_map.json",
          description="Per-pixel material id (not anti-aliased)."),
)

# -- 5. Motion & denoising ------------------------------------------------------
_register(
    _pass("motion_vector", f"{MOTION}/motion_vector", BEAUTY, ("Vector",),
          "RGBA", group="motion", label="Motion Vectors",
          flags=((_L, "use_pass_vector"),),
          space="pixels; RG = to previous frame, BA = to next frame",
          description="Scene motion at the current frame (zero for static scenes)."),
    _pass("noisy_image", f"{MOTION}/noisy_image", BEAUTY, ("Noisy Image",),
          "RGBA", group="motion", label="Noisy Image",
          flags=((_C, "denoising_store_passes"),), policy=POLICY_APPEARANCE,
          space="scene-linear RGB + alpha",
          description="Combined before denoising. The lighting passes sum to "
                      "this image exactly; with the denoiser on they do not "
                      "sum to appearance/combined."),
    _pass("denoising_albedo", f"{MOTION}/denoising_albedo", BEAUTY,
          ("Denoising Albedo",), "RGBA", group="motion",
          label="Denoising Albedo", flags=((_C, "denoising_store_passes"),),
          policy=POLICY_HALF, space="linear RGB",
          description="Cycles denoising albedo feature pass."),
    _pass("denoising_normal", f"{MOTION}/denoising_normal", BEAUTY,
          ("Denoising Normal",), "VECTOR", group="motion",
          label="Denoising Normal", flags=((_C, "denoising_store_passes"),),
          policy=POLICY_HALF, space="as written by Cycles for its denoiser",
          description="Cycles denoising normal feature pass."),
)


def passes_for(source):
    return [definition for definition in PASSES.values()
            if definition["source"] == source]


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def frame_stem(frame_index):
    return f"frame_{int(frame_index):04d}"


def raw_root(build_dir):
    """``Dataset(Raw)`` of a build, from the build or its legacy dataset."""
    from ..building_data import paths as bd_paths

    return bd_paths.build_root(build_dir) / RAW_FOLDER


def work_dir(root):
    return Path(root) / WORK_FOLDER


def pass_file(root, key, frame_index):
    return Path(root) / PASSES[key]["folder"] / f"{frame_stem(frame_index)}.exr"


def view_complete(root, frame_index, keys):
    for key in keys:
        path = pass_file(root, key, frame_index)
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True
