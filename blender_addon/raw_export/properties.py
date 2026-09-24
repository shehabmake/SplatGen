# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Raw dataset settings, stored on the scene as ``Scene.splatgen_raw``.

Kept apart from ``SCENERAY_SPLAT`` on purpose: the legacy settings and the
legacy output are frozen for comparison, and nothing here can change them.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import PropertyGroup


class SplatGenRawSettings(PropertyGroup):
    enabled: BoolProperty(
        name="Export Raw Data",
        description=(
            "Also write Dataset(Raw): render passes, material and geometry "
            "data, ids, cameras, scene mesh, voxels and environment captures. "
            "The legacy Dataset(Default) output is unchanged either way"
        ),
        default=True,
    )
    # -- what to write ---------------------------------------------------
    lighting: BoolProperty(
        name="Lighting Split",
        description="Diffuse/glossy/transmission direct, indirect and color, "
                    "volume, emission and environment passes",
        default=True,
    )
    clay: BoolProperty(
        name="Clay Render",
        description="A second render per camera with every surface replaced "
                    "by neutral grey, for lighting without albedo",
        default=True,
    )
    geometry: BoolProperty(
        name="Geometry Passes",
        description="Position, depth, normal, true normal, UV, object "
                    "coordinates, pointiness, backfacing and ambient occlusion",
        default=True,
    )
    material: BoolProperty(
        name="Material Passes",
        description="Per-pixel base color, roughness, metallic, specular, IOR, "
                    "anisotropic, coat, sheen, transmission, subsurface, alpha "
                    "and emission, read from each material's shader inputs",
        default=True,
    )
    ids: BoolProperty(
        name="Object & Material IDs",
        description="Per-pixel object and material ids with a name map",
        default=True,
    )
    motion: BoolProperty(
        name="Motion & Denoising",
        description="Motion vectors and the Cycles denoising albedo/normal",
        default=True,
    )
    scene_mesh: BoolProperty(
        name="Scene Mesh",
        description="Every render-visible mesh, evaluated and in world space, "
                    "as one binary PLY with per-face object and material ids",
        default=True,
    )
    collision_voxels: BoolProperty(
        name="Collision Voxels",
        description="Occupancy grid: free space, surface and enclosed volume",
        default=True,
    )
    world: BoolProperty(
        name="World / HDRI",
        description="Equirectangular render of the world, plus a copy of any "
                    "environment image it uses",
        default=True,
    )
    probes: BoolProperty(
        name="Reflection Probes",
        description="Equirectangular radiance and distance captures of the "
                    "whole scene from probe positions",
        default=True,
    )
    # -- storage -----------------------------------------------------------
    appearance_precision: EnumProperty(
        name="Appearance Precision",
        description="Bit depth of the color passes. Geometry, id and motion "
                    "passes always keep the precision they need",
        items=[
            ("HALF", "Half (16-bit)", "Half-float color; plenty for training "
             "and half the size"),
            ("FLOAT", "Float (32-bit)", "Full float color"),
        ],
        default="HALF",
    )
    appearance_codec: EnumProperty(
        name="Appearance Compression",
        description="EXR compression of the color passes. Data passes are "
                    "always stored losslessly",
        items=[
            ("ZIP", "ZIP (lossless)", "Lossless, good all-round"),
            ("PIZ", "PIZ (lossless)", "Lossless, better on noisy renders"),
            ("DWAA", "DWAA (lossy)", "Much smaller color passes with "
             "visually lossless compression"),
        ],
        default="ZIP",
    )
    data_codec: EnumProperty(
        name="Data Compression",
        description="EXR compression of geometry, material and denoising data "
                    "passes. Depth and ids are always stored exactly",
        items=[
            ("ZIP", "ZIP (lossless)", "Exact float32 values"),
            ("PXR24", "PXR24 (near-lossless)", "Float32 rounded to 24 bits "
             "(about 5 significant digits) before lossless compression; "
             "roughly half the size for position, UV and object coordinates"),
        ],
        default="ZIP",
    )
    # -- auxiliary renders -------------------------------------------------
    aux_samples: IntProperty(
        name="Auxiliary Samples",
        description="Cycles samples for clay, world and probe renders. "
                    "0 uses the scene's own sample count",
        default=64, min=0, max=65536,
    )
    world_resolution: IntProperty(
        name="World Height",
        description="Height of the equirectangular world render; width is twice this",
        default=1024, min=16, max=16384,
    )
    probe_resolution: IntProperty(
        name="Probe Height",
        description="Height of each equirectangular probe capture; width is twice this",
        default=256, min=16, max=8192,
    )
    probe_count: IntProperty(
        name="Automatic Probes",
        description="Probes placed automatically in free space when the scene "
                    "has no Light Probe Sphere objects of its own",
        default=1, min=0, max=64,
    )
    voxel_resolution: IntProperty(
        name="Voxel Resolution",
        description="Voxels along the longest side of the scene bounds",
        default=128, min=16, max=512,
    )


def settings(scene):
    return getattr(scene, "splatgen_raw", None)


def enabled(scene):
    raw = settings(scene)
    return bool(raw is not None and raw.enabled)


def group_enabled(raw, group):
    if group == "always":
        return True
    return bool(getattr(raw, {
        "lighting": "lighting",
        "clay": "clay",
        "geometry": "geometry",
        "material": "material",
        "ids": "ids",
        "motion": "motion",
    }[group]))


def register():
    bpy.utils.register_class(SplatGenRawSettings)
    bpy.types.Scene.splatgen_raw = bpy.props.PointerProperty(
        type=SplatGenRawSettings
    )


def unregister():
    if hasattr(bpy.types.Scene, "splatgen_raw"):
        del bpy.types.Scene.splatgen_raw
    try:
        bpy.utils.unregister_class(SplatGenRawSettings)
    except RuntimeError:
        pass
