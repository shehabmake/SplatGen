# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Raw export settings, drawn inside the Dataset settings popover, and the
Export Raw Data operator for running the stage on an existing build."""

import bpy
from bpy.types import Operator

from .. import theme
from . import layout, properties, stage


def _target_build(cfg):
    """The build Export Raw Data works on: the active one, else the newest."""
    from .. import sceneray_splat
    from ..building_data import paths as bd_paths

    active = sceneray_splat._sr_effective_output_path(cfg)
    if active is not None and bd_paths.has_stage1_output(active):
        return active
    root = sceneray_splat._sr_output_base_path(cfg)
    latest = bd_paths.latest_build(root) if root is not None else None
    if latest is not None and bd_paths.has_stage1_output(latest):
        return latest
    return None


class SPLATGEN_OT_export_raw_data(Operator):
    """Write or complete Dataset(Raw) for the current build without touching
    its legacy files. Renders only what is missing"""

    bl_idname = "splatgen.export_raw_data"
    bl_label = "Export Raw Data"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        from ..building_data import stages

        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and properties.enabled(context.scene)
                and not stages.anything_running(context)
                and not stage.is_running())

    def execute(self, context):
        from .. import progress

        cfg = context.scene.SCENERAY_SPLAT
        build = _target_build(cfg)
        if build is None:
            self.report({"ERROR"}, "Build the dataset first: Export Raw Data needs "
                                   "rendered images and cameras.txt/images.txt.")
            return {"CANCELLED"}
        progress.begin(stage.PHASE, (stage.PHASE,))
        if not stage.start(context, build):
            progress.end("Nothing to export")
            self.report({"ERROR"}, f"No rendered views found in {build.name}.")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exporting raw data into {build.name}/{layout.RAW_FOLDER}")
        return {"FINISHED"}


def draw_settings(layout_, context, busy):
    raw = properties.settings(context.scene)
    if raw is None:
        return
    theme.heading(layout_, "Raw data (trainer input)", custom="section_dataset")
    body = theme.form(layout_)
    body.enabled = not busy
    body.prop(raw, "enabled")
    if not raw.enabled:
        return
    grid = body.grid_flow(columns=2, even_columns=True, align=True)
    for name in ("lighting", "clay", "geometry", "material", "ids", "motion",
                 "scene_mesh", "collision_voxels", "world", "probes"):
        grid.prop(raw, name, toggle=True)
    body.separator(factor=0.5)
    column = body.column(align=True)
    column.use_property_split = True
    column.use_property_decorate = False
    column.prop(raw, "appearance_precision", text="Color precision")
    column.prop(raw, "appearance_codec", text="Color compression")
    column.prop(raw, "aux_samples")
    if raw.world:
        column.prop(raw, "world_resolution")
    if raw.probes:
        column.prop(raw, "probe_resolution")
        column.prop(raw, "probe_count")
    if raw.collision_voxels:
        column.prop(raw, "voxel_resolution")
    theme.message(body, "Written to Dataset(Raw) beside the unchanged legacy dataset.")
    row = body.row()
    theme.operator(row, SPLATGEN_OT_export_raw_data.bl_idname,
                   text="Export raw data for this build", icon="EXPORT")


CLASSES = (SPLATGEN_OT_export_raw_data,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
