# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""The Building Data panel.

Two sections, one per operation:

    Building Data
    ├── Rendering Images   camera list, render settings, build, status
    └── Point Cloud        estimate, quality preset, build, status

Everything that produces dataset files lives here. As with the other workflow
sections this class is never registered: the host panel calls ``draw`` through
a proxy layout, so ``self`` is not a panel instance and the drawing work lives
in module-level functions.
"""

from bpy.types import Panel

from .. import theme
from . import manifest as bd_manifest
from . import paths
from . import stages


def _section(layout, cfg, property_name, label, icon, *, custom=None,
             detail=None, state=None):
    from .. import sceneray_splat

    return sceneray_splat._sr_pipeline_section(
        layout, cfg, property_name, label, icon,
        custom=custom, detail=detail, state=state,
    )


# The per-section progress bars are gone. There is exactly one bar, drawn by
# progress.draw_active directly beneath whichever button started the work.


# --------------------------------------------------------------------------
# Rendering Images
# --------------------------------------------------------------------------

def draw_image_settings(layout, cfg, busy=False):
    """Formats first; PNG compression is shared by RGB and validity masks."""
    layout.enabled = not busy
    layout.prop(cfg, 'dataset_image_format')
    layout.prop(cfg, 'dataset_mask_format')
    if cfg.dataset_image_format == 'JPEG':
        layout.prop(cfg, 'jpeg_quality')
    if 'PNG' in (cfg.dataset_image_format, cfg.dataset_mask_format):
        layout.prop(cfg, 'png_compression')


def _draw_render_controls(layout, cfg, busy):
    box = theme.form(layout)
    box.enabled = not busy
    box.use_property_split = True
    box.use_property_decorate = False
    theme.heading(
        box,
        "RENDER OUTPUT",
        fallback="OUTPUT",
    )
    draw_image_settings(box, cfg, busy)


def _outdated_cameras(context, cfg):
    """Cameras whose settings no longer match the image already rendered."""
    from .. import sceneray_splat

    # Same rule as the queue itself: a camera can only be out of date with
    # respect to work that already exists.
    if not sceneray_splat._sr_continuing_existing_dataset(cfg):
        return {}
    output_dir = sceneray_splat._sr_effective_output_path(cfg)
    if output_dir is None:
        return {}
    data = (
        sceneray_splat.read_render_manifest(output_dir)
        or sceneray_splat.read_completed_camera_data(output_dir)
        or {}
    )
    entries = {
        entry.get("name"): entry
        for entry in data.get("cameras", ())
        if isinstance(entry, dict) and entry.get("name")
    }
    if not entries:
        return {}
    try:
        return bd_manifest.outdated_cameras(cfg, entries)
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        return {}


def _draw_rendering_images(layout, context, cfg, busy):
    box = _section(
        layout, cfg, "show_render_section", "Render Images", "RENDER_STILL",
        custom="stage_dataset",
    )
    if box is None:
        return
    _draw_render_controls(box, cfg, busy)


def _draw_camera_section(layout, context, cfg):
    """Step one: which cameras the dataset is built from."""
    box = _section(
        layout, cfg, "show_camera_list_section", "Camera List",
        "OUTLINER_OB_CAMERA",
        custom="stage_cameras",
    )
    if box is None:
        return
    from .. import sceneray_splat

    sceneray_splat.SCENERAY_SPLAT_PT_cameras.draw(
        sceneray_splat._SplatGenDrawProxy(box), context
    )


# --------------------------------------------------------------------------
# Point Cloud
# --------------------------------------------------------------------------

def _draw_point_estimate(layout, cfg):
    """What the current settings will produce, as figures rather than prose."""
    from .. import sceneray_splat

    queued = sum(1 for item in cfg.camera_queue if item.camera is not None)
    if not queued:
        return
    estimate = sceneray_splat._sr_estimate_point_cloud(cfg, queued)
    retention = f"{estimate['retained_fraction'] * 100.0:.0f}%"
    if estimate["retention_learned"]:
        retention += " (from last run)"
    rows = [
        ("Cameras Used",
         f"{estimate['cameras_used']:,} of {estimate['cameras_total']:,}"),
        ("Rays Per Camera", f"{estimate['rays_per_camera']:,}"),
        ("Merge Retention", retention),
        ("Estimated Points", f"~{estimate['final_points']:,}"),
        ("Warning Above", f"{estimate['warning_points']:,}"),
    ]
    if cfg.point_last_count:
        rows.append(("Last Generated", f"{cfg.point_last_count:,}"))
    theme.stat(layout, rows)
    _draw_point_size_warning(layout, cfg, estimate)


def _draw_point_size_warning(layout, cfg, estimate=None, short=False):
    """The over-2,000,000-points warning. Warns only; nothing is capped."""
    from .. import sceneray_splat

    if estimate is None:
        queued = sum(1 for item in cfg.camera_queue if item.camera is not None)
        if not queued:
            return
        estimate = sceneray_splat._sr_estimate_point_cloud(cfg, queued)
    limit = estimate["warning_points"]
    if estimate["exceeds_warning"]:
        text = (
            f"Point cloud estimate ~{estimate['final_points']:,} exceeds "
            f"{limit:,} points - too much for the trainer. See Point Cloud "
            "settings."
            if short else
            sceneray_splat._sr_large_point_cloud_message(
                estimate["final_points"], estimated=True)
        )
        theme.status(layout, text, state=theme.STATUS_ERROR, alert=True)
    elif not short and cfg.point_last_count > limit:
        theme.status(
            layout,
            f"The last point cloud had {cfg.point_last_count:,} points, "
            f"over {limit:,}.",
            state=theme.STATUS_ERROR, alert=True,
        )


def _draw_point_quality(layout, cfg, busy):
    box = theme.form(layout)
    box.enabled = not busy
    box.use_property_split = True
    box.use_property_decorate = False
    theme.heading(
        box,
        "POINT QUALITY",
        custom="stage_points",
        fallback="PRESET",
    )
    box.prop(cfg, "point_quality", text="Quality preset")
    if cfg.point_quality != "CUSTOM":
        return
    advanced = box.column(align=True)
    advanced.prop(cfg, "point_camera_usage")
    advanced.prop(cfg, "point_sampling_density")
    advanced.prop(cfg, "point_merging_strength")


def _draw_point_cloud(layout, context, cfg, busy):
    box = _section(
        layout,
        cfg,
        "show_pointcloud_section",
        "Point Cloud",
        "OUTLINER_OB_POINTCLOUD",
        custom="stage_points",
    )
    if box is None:
        return
    _draw_point_quality(box, cfg, busy)

    box.separator(factor=.5)
    information = box.column()
    theme.heading(information, "Sampling estimate", fallback="INFO")
    _draw_point_estimate(information, cfg)


# Camera coverage has one entry point: the Camera coverage card in Prepare
# (auto_rig.coverage_ui), directly above Build Dataset.


def _draw_build_dataset(layout, context, cfg, busy):
    import bpy
    from .. import sceneray_splat
    if not bpy.data.is_saved:
        row = layout.column()
        row.operator_context = 'INVOKE_DEFAULT'
        theme.primary(row, 'wm.save_as_mainfile', 'Save File First',
                      'action_save', enabled=not busy)
        return
    ready = not stages.anything_running(context) and sceneray_splat._sr_has_queued_camera(cfg)
    # Said before the build starts, where the button is pressed.
    _draw_point_size_warning(layout, cfg, short=True)
    report = stages.plan(context, cfg)
    state = report.get('state', 'NEW')
    if state == 'INCOMPLETE':
        theme.primary(layout, stages.SPLATRAY_OT_build_dataset.bl_idname, 'Continue dataset',
                      'action_build', enabled=ready).requested_action = 'CONTINUE'
        row = layout.row(); row.enabled = ready
        theme.operator(row, stages.SPLATRAY_OT_build_dataset.bl_idname, text='Build new dataset',
                       custom='ui_reset').requested_action = 'NEW'
    else:
        theme.primary(layout, stages.SPLATRAY_OT_build_dataset.bl_idname,
                      'Build new dataset' if state == 'COMPLETE' else 'Build dataset',
                      'action_build', enabled=ready).requested_action = 'NEW' if state == 'COMPLETE' else 'AUTO'


class SPLATRAY_PT_building_data(Panel):
    """Section 2 of the workflow: everything that produces dataset files."""

    bl_idname = "SPLATRAY_PT_building_data"
    bl_label = "2. Building Data"
    bl_parent_id = "SCENERAY_SPLAT_PT_panel"
    bl_space_type = "OUTLINER"
    bl_region_type = "WINDOW"
    bl_category = "SplatGen"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        cfg = context.scene.SCENERAY_SPLAT
        busy = stages.is_busy(cfg)
        # Cameras, then the render, then the point cloud: the same order the
        # build itself runs in, so reading the panel top to bottom describes
        # what Build Dataset is about to do.
        # The camera list draws the global settings itself, so this panel and
        # the Camera Placement one stay identical with no duplicated layout.
        _draw_camera_section(layout, context, cfg)
        _draw_rendering_images(layout, context, cfg, busy)
        _draw_point_cloud(layout, context, cfg, busy)
        _draw_build_dataset(layout, context, cfg, busy)


# The host panel keeps the camera list and Build Dataset permanently visible
# and folds the two settings groups away, so it reaches these three pieces
# individually rather than drawing the section as a block.

def draw_build_dataset(layout, context, cfg):
    _draw_build_dataset(layout, context, cfg, stages.is_busy(cfg))


def draw_render_settings(layout, context, cfg):
    _draw_rendering_images(layout, context, cfg, stages.is_busy(cfg))


def draw_pointcloud_settings(layout, context, cfg):
    _draw_point_cloud(layout, context, cfg, stages.is_busy(cfg))
