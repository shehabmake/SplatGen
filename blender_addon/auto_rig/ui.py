# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Smart camera rig card: systems first, then one Calculate button.

Reading top to bottom follows the workflow: add a system, set it up in the
list, calculate. Calculate stays disabled until at least one system is ready,
so the first thing to press is always the one at the top.
"""

import bpy

from .. import theme
from . import operators
from .properties import KIND_ICONS

ADD_SYSTEM = operators.SPLATGEN_OT_auto_rig_add_system.bl_idname


class SPLATGEN_UL_rig_systems(bpy.types.UIList):
    """One line per system: use, type, name, and what it still needs."""
    bl_idname = "SPLATGEN_UL_rig_systems"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.label(text="", icon=KIND_ICONS[item.kind])
        name = row.row()
        name.active = item.enabled
        name.prop(item, "name", text="", emboss=False)
        problem = operators.system_problem(context, item)
        tail = row.row(align=True)
        tail.alignment = 'RIGHT'
        if problem:
            tail.alert = True
            tail.label(text=problem, icon='ERROR')
        else:
            if not item.max_cameras:
                cams = "auto"
            elif item.stop_when_covered:
                cams = f"≤ {item.max_cameras} cams"
            else:
                cams = f"{item.max_cameras} cams"
            tail.label(text=f"{item.views} views · {cams}" if item.stop_when_covered else cams)


def _draw_system(layout, context, settings, index):
    """The selected system's own settings, directly under the list."""
    system = settings.systems[index]
    box = layout.box()
    head = box.row(align=True)
    head.label(text=system.name, icon=KIND_ICONS[system.kind])
    theme.operator(head, operators.SPLATGEN_OT_auto_rig_remove_system.bl_idname,
                   text="", icon='X').index = index
    kinds = box.row(align=True)
    for kind, text in (('INTERIOR', "Interior"), ('EXTERIOR', "Exterior"), ('OBJECT', "Object")):
        kinds.prop_enum(system, "kind", kind, text=text, icon=KIND_ICONS[kind])
    box.separator(factor=.5)
    location = box.column()
    theme.heading(location, "Capture area", fallback='SPHERE')
    if system.kind in {'INTERIOR', 'EXTERIOR'}:
        row = location.row(align=True)
        row.scale_y = 1.2
        if operators.blob_alive(context, system):
            op = theme.operator(row, operators.SPLATGEN_OT_auto_rig_place_blob.bl_idname,
                                text="Select blob", icon='RESTRICT_SELECT_OFF')
            op.index, op.to_cursor = index, False
            op = theme.operator(row, operators.SPLATGEN_OT_auto_rig_place_blob.bl_idname,
                                text="To cursor", icon='PIVOT_CURSOR')
            op.index, op.to_cursor = index, True
        else:
            row.alert = True
            op = theme.operator(row, operators.SPLATGEN_OT_auto_rig_place_blob.bl_idname,
                                text="Blob missing: create", icon='SPHERE')
            op.index, op.to_cursor = index, True
        # Toggle rows span the full width: a split form would turn their
        # text into a side label and leave a bare icon button.
        form = theme.form(location)
        toggle = form.row(align=True)
        toggle.use_property_split = False
        toggle.prop(system, "bounded", text="Limit radius")
        if system.bounded and operators.blob_alive(context, system):
            form.prop(system.blob, "empty_display_size", text="Radius")
        if system.kind == 'EXTERIOR':
            reach = form.row(align=True)
            reach.use_property_split = False
            reach.prop_enum(system, "reach", 'NEARBY', text="Nearby")
            reach.prop_enum(system, "reach", 'SCENE', text="Whole scene")
    else:
        form = theme.form(location)
        source = form.row(align=True)
        source.use_property_split = False
        source.prop(system, "source", expand=True)
        if system.source == 'OBJECT':
            form.prop(system, "target_object", text="")
        else:
            form.prop(system, "target_collection", text="")
    box.separator(factor=.7)
    theme.heading(box, "Camera coverage", custom='stage_cameras')
    counts = theme.form(box)
    counts.prop(system, "max_cameras", text="Cameras")
    stop = counts.row()
    stop.use_property_split = False
    stop.prop(system, "stop_when_covered", text="Stop when covered")
    if system.stop_when_covered:
        counts.prop(system, "views", text="Views")

    # Placement settings that may differ per system. Everything else is
    # global: one lens for every camera, one analysis quality, one total.
    box.separator(factor=.7)
    toggle = box.row()
    toggle.use_property_split = False
    toggle.prop(system, "use_custom", text="Custom placement")
    if not system.use_custom:
        return
    custom = theme.form(box)
    custom.prop(system, "clearance")
    if system.kind in {'INTERIOR', 'EXTERIOR'}:
        custom.prop(system, "seal_size")
        custom.prop(system, "limit_height")
        if system.limit_height:
            heights = custom.column(align=True)
            heights.prop(system, "height_min")
            heights.prop(system, "height_max")
    if system.kind == 'EXTERIOR':
        custom.prop(system, "allow_below")


def draw_panel(layout, context):
    """The Smart camera rig card in Prepare."""
    from ..building_data import stages
    scene = context.scene
    settings = scene.splatgen_auto_rig
    cfg = scene.SCENERAY_SPLAT
    busy = stages.anything_running(context)
    box = layout.column()

    body = box.column()
    body.enabled = not busy
    add = body.row(align=True)
    add.scale_y = 1.4
    theme.accent(add, not len(settings.systems))
    add.operator_menu_enum(ADD_SYSTEM, "kind", text="Add system", icon='ADD')

    if settings.systems:
        rows = min(5, max(2, len(settings.systems)))
        body.template_list("SPLATGEN_UL_rig_systems", "", settings, "systems",
                           settings, "active_system", rows=rows, maxrows=5)
        index = min(max(0, settings.active_system), len(settings.systems) - 1)
        _draw_system(body, context, settings, index)

    asked = operators.cameras_asked(context)
    if asked > settings.max_cameras:
        # Counted systems cannot all get their cameras: say so before the run.
        warn = body.box().column(align=True)
        warn.alert = True
        warn.label(text=f"Systems ask for {asked} cameras", icon='ERROR')
        warn.prop(settings, "max_cameras")

    body.separator(factor=0.5)
    run = body.row(align=True)
    run.scale_y = 1.4
    main = run.row(align=True)
    ready = operators.ready_systems(context)
    main.enabled = bool(ready)
    theme.accent(main, bool(ready) and not len(cfg.camera_queue))
    label = f"Calculate cameras ({len(ready)})" if ready else "Calculate cameras"
    theme.operator(main, operators.SPLATGEN_OT_auto_rig_generate.bl_idname,
                   text=label, custom="action_calculate", icon='AUTO')
    run.popover(panel="SPLATGEN_PT_auto_rig", text="", **theme.icon_args('PREFERENCES'))
    if operators.auto_rig_cameras(scene):
        theme.operator(run, operators.SPLATGEN_OT_auto_rig_clear.bl_idname, text="", icon='TRASH')

def draw_settings(layout, context):
    """Calculation settings, in two kinds.

    *All systems* are genuinely global - one lens for every camera, one
    analysis quality, one total - and exist only here. *System defaults*
    apply to every system that does not set its own under Custom Placement.
    """
    from ..building_data import stages
    scene = context.scene
    settings = scene.splatgen_auto_rig
    cfg = scene.SCENERAY_SPLAT
    layout.enabled = not stages.anything_running(context)
    theme.heading(layout, "All systems", custom="method_auto_rig", fallback='AUTO')
    theme.muted(layout, "Shared by every camera and system.")
    form = theme.form(layout)
    form.row().prop(settings, "quality", expand=True)
    # The lens is the global one every queued camera uses; showing it here
    # is a reminder that a wider lens needs far fewer cameras.
    form.prop(cfg, "global_focal_length", text="Lens (all cameras)")
    total = form.row(align=True)
    asked = operators.cameras_asked(context)
    total.alert = asked > settings.max_cameras
    total.prop(settings, "max_cameras")
    if asked > settings.max_cameras:
        theme.status(form, f"Systems ask for {asked}: raise the total.", theme.STATUS_WAIT)
    elif asked:
        theme.muted(form, f"Systems ask for {asked} of them.")
    form.prop(settings, "geometry_collection")
    form.prop(settings, "replace_previous")
    form.prop(settings, "live_preview")
    form.prop(settings, "seed")

    layout.separator(factor=0.5)
    theme.heading(layout, "System defaults", custom="stage_cameras", fallback='CAMERA_DATA')
    theme.muted(layout, "Used by every system without Custom Placement.")
    form = theme.form(layout)
    form.prop(settings, "clearance")
    form.prop(settings, "seal_size")
    form.prop(settings, "limit_height")
    if settings.limit_height:
        sub = form.column(align=True)
        sub.prop(settings, "height_min")
        sub.prop(settings, "height_max")
    form.prop(settings, "allow_below")


class SPLATGEN_PT_auto_rig(bpy.types.Panel):
    bl_idname = "SPLATGEN_PT_auto_rig"
    bl_label = "Smart camera rig settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_ui_units_x = 17

    def draw(self, context):
        draw_settings(self.layout, context)


CLASSES = (SPLATGEN_UL_rig_systems, SPLATGEN_PT_auto_rig)
