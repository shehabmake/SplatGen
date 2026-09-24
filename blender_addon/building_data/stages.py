# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""The two independent Building Data operations.

Stage 1 renders the images and writes ``cameras.txt`` and ``images.txt``.
Stage 2 reads those and writes the point cloud. Neither triggers the other:
they are separate buttons so point-cloud work can grow without touching the
rendering path.

This module never imports ``sceneray_splat`` at module level - that module
imports ``building_data.paths``, and importing back would close the loop.
"""

import time

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator

from .. import theme


def _cfg(context):
    return getattr(context.scene, "SCENERAY_SPLAT", None)


def is_busy(cfg):
    return bool(
        cfg is not None and (cfg.is_rendering or cfg.is_generating_points)
    )


def anything_running(context=None):
    """Whether any major operation owns the machine right now.

    One at a time: rendering, the point cloud and coverage validation
    compete for the same CPU and GPU, and running two at once
    produces conflicts rather than results. Every long operation reports
    through ``progress``, so asking it covers all of them at once.
    """
    from .. import progress

    if progress.is_active():
        return True
    context = context or bpy.context
    scene = getattr(context, "scene", None)
    if is_busy(getattr(scene, "SCENERAY_SPLAT", None)):
        return True
    return False


def plan(context, cfg):
    """What pressing Build Images would do right now, without doing it."""
    from .. import sceneray_splat

    report = sceneray_splat._sr_inspect_dataset_state(context, cfg)
    state = report["state"]
    report["will_start_new"] = state == "COMPLETE"
    report["will_rotate_recovery"] = state == "INCOMPLETE"
    return report


def _summarise(report):
    """One line describing what is already on disk, for the confirm dialog."""
    name = report.get("previous_name") or "the previous build"
    rendered = int(report.get("rendered", 0))
    pending = int(report.get("pending", 0))
    if report["state"] == "COMPLETE":
        return f"{name} — {rendered:,} camera(s) rendered, dataset complete."
    missing = report.get("missing_files") or []
    detail = f"{name} — {rendered:,} rendered, {pending:,} still to render"
    if missing:
        detail += f"; missing {', '.join(missing)}"
    return detail + "."




class _RenderStage:
    """Shared implementation of the two render-driven workflows.

    A plain mixin on purpose. Subclassing a *registered* Operator makes the
    subclass inherit its ``bl_rna``, which unregisters the parent's polling
    behaviour and leaves its button permanently greyed out.
    """

    #: 'RENDER' stops after the metadata; 'FULL' carries on into the point
    #: cloud. The render pipeline reads this to decide what to announce.
    workflow = 'RENDER'

    # Blender turns annotations into RNA when a class is created and does walk
    # the bases, so both operators get their own independent copy of these.
    confirm_new: BoolProperty(
        name="Create a new dataset for this scene",
        description=(
            "Write a second dataset folder alongside the finished one. "
            "The existing dataset is never modified or deleted"
        ),
        default=False,
        options={'SKIP_SAVE'},
    )
    continue_previous: BoolProperty(
        name="Continue the unfinished build",
        description=(
            "Carry the images that already rendered forward and render only "
            "what is missing. Switch off to render every camera again"
        ),
        default=True,
        options={'SKIP_SAVE'},
    )
    #: Set by ``invoke`` so ``draw`` and ``execute`` agree on what was found
    #: without inspecting the disk again on every redraw. Empty when the
    #: operator is called from a script, which is what keeps it unprompted.
    pending_state: StringProperty(
        description="What was found on disk when the button was pressed",
        default="",
        options={'SKIP_SAVE', 'HIDDEN'},
    )
    pending_summary: StringProperty(
        description="Human-readable form of what was found on disk",
        default="",
        options={'SKIP_SAVE', 'HIDDEN'},
    )
    requested_action: StringProperty(
        description="Action selected by the dynamic Build Dataset controls",
        default="AUTO",
        options={'SKIP_SAVE', 'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        from .. import sceneray_splat

        cfg = _cfg(context)
        return (
            cfg is not None
            and not is_busy(cfg)
            and sceneray_splat._sr_has_queued_camera(cfg)
        )

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------

    def invoke(self, context, event):
        """Ask before writing, whenever there is something already there.

        A build always writes into a brand new folder. That is safe, but it is
        not obvious: pressing the button twice quietly leaves two datasets on
        disk. So a finished dataset asks for confirmation before a second one
        is made, and unfinished work asks whether to carry on with it.

        A first build has nothing to warn about and runs straight away - a
        dialog there would be noise on the one press that cannot surprise
        anyone.
        """
        cfg = _cfg(context)
        if cfg is None:
            return self.execute(context)
        report = plan(context, cfg)
        self.pending_state = report["state"]
        self.pending_summary = _summarise(report)
        if report["state"] == "NEW":
            return self.execute(context)
        if report["state"] == "INCOMPLETE" and self.requested_action == "CONTINUE":
            self.continue_previous = True
            return self.execute(context)
        # A new dataset is never created without being asked for; continuing
        # unfinished work is the safe default, so that one starts ticked.
        self.confirm_new = False
        self.continue_previous = self.requested_action != "NEW"
        return context.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, context):
        layout = self.layout
        if self.requested_action == "NEW":
            self._draw_new_confirmation(
                layout, incomplete=self.pending_state == "INCOMPLETE"
            )
        elif self.pending_state == "COMPLETE":
            self._draw_complete(layout)
        elif self.pending_state == "INCOMPLETE":
            self._draw_incomplete(layout)

    def _draw_new_confirmation(self, layout, *, incomplete=False):
        box = layout.box()
        box.alert = incomplete
        if incomplete:
            box.label(text="This dataset build is incomplete.", icon=theme.STATUS_WAIT)
            if self.pending_summary:
                theme.message(box, self.pending_summary)
            box.label(
                text="Starting over will not continue from the rendered images above.",
                icon="ERROR",
            )
            question = "Start a separate dataset instead of continuing this build?"
        else:
            box.label(text="The dataset has already been completely built.",
                      icon=theme.STATUS_OK)
            if self.pending_summary:
                theme.message(box, self.pending_summary)
            question = "Do you want to build a new dataset from scratch?"

        layout.separator()
        theme.heading(layout, "New dataset", fallback="FILE_REFRESH")
        theme.message(layout, question)
        note = layout.column(align=True)
        theme.message(note, text="Every camera will render again; no images are reused.")
        theme.message(note, text="The existing files are not overwritten.")
        layout.separator()
        row = layout.row()
        row.scale_y = 1.3
        row.prop(self, "confirm_new", text="Yes, build a new dataset")
        if not self.confirm_new:
            warn = layout.row()
            warn.alert = True
            warn.label(text="Confirm above, or press Cancel.",
                       icon=theme.STATUS_WAIT)

    def _draw_complete(self, layout):
        box = layout.box()
        box.label(text="This scene already contains a generated dataset.",
                  icon=theme.STATUS_OK)
        if self.pending_summary:
            theme.message(box, self.pending_summary)

        layout.separator()
        layout.label(text="Do you want to rebuild the dataset?",
                     icon="FILE_REFRESH")
        note = layout.column(align=True)
        theme.message(note, text="Every image is rendered again before the new dataset "
                        "is generated.")
        theme.message(note, text="Nothing is reused — the scene may have changed since "
                        "the last one.")
        theme.message(note, text="The existing dataset is kept in its own folder.")

        layout.separator()
        row = layout.row()
        row.scale_y = 1.3
        row.prop(self, "confirm_new")
        if not self.confirm_new:
            warn = layout.row()
            warn.alert = True
            warn.label(text="Tick the box above, or press Cancel.",
                       icon=theme.STATUS_WAIT)

    def _draw_incomplete(self, layout):
        box = layout.box()
        box.label(text="Unfinished work found from an earlier build.",
                  icon=theme.STATUS_WAIT)
        if self.pending_summary:
            theme.message(box, self.pending_summary)

        layout.separator()
        row = layout.row()
        row.scale_y = 1.3
        row.prop(self, "continue_previous")

        outcome = layout.column(align=True)
        if self.continue_previous:
            theme.message(outcome, text="Images already rendered are carried forward; "
                               "only the missing ones render.",
                          icon=theme.STATUS_OK)
        else:
            theme.message(outcome, text="Every camera renders again from scratch.",
                          icon=theme.STATUS_WAIT)
        theme.message(outcome, text="Either way the unfinished folder is kept as a "
                           "single recovery copy.")

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def execute(self, context):
        from .. import sceneray_splat

        if not bpy.data.is_saved:
            self.report({'ERROR'}, "Save the .blend file before building the dataset.")
            return {'CANCELLED'}
        cfg = context.scene.SCENERAY_SPLAT
        # ``pending_state`` is empty when execute() is reached directly - from
        # a script, or EXEC_DEFAULT - which keeps the operator scriptable and
        # unprompted while the button still asks.
        requires_confirmation = (
            self.pending_state == "COMPLETE"
            or (
                self.pending_state == "INCOMPLETE"
                and self.requested_action == "NEW"
            )
        )
        if requires_confirmation and not self.confirm_new:
            self.report({"INFO"},
                        "Cancelled — the existing dataset was left alone.")
            return {"CANCELLED"}
        if self.pending_state == "COMPLETE" or self.requested_action == "NEW":
            # A rebuild renders everything again. The scene may have moved,
            # been relit or re-shaded since the finished dataset was made, and
            # a carried-forward image would silently mix the two.
            reuse_images = False
        elif self.pending_state == "INCOMPLETE":
            reuse_images = self.continue_previous
        else:
            reuse_images = True

        cfg.build_workflow = self.workflow
        report = plan(context, cfg)
        state = report["state"]
        previous = report["previous"]
        root = sceneray_splat._sr_output_base_path(cfg)

        try:
            # Images that are still valid are carried forward unless the user
            # asked for a clean start: re-rendering an unchanged camera costs
            # time and changes nothing.
            target, previous, reused, pending = (
                sceneray_splat._sr_prepare_versioned_dataset(
                    context, cfg, reuse_images=reuse_images
                )
            )
        except (OSError, TypeError, ValueError) as exc:
            cfg.build_workflow = 'NONE'
            self.report({"ERROR"}, f"Could not start the build: {exc}")
            return {"CANCELLED"}

        recovery, removed = None, []
        if state != "COMPLETE" and previous is not None:
            # Anything short of a finished build is a leftover once its work
            # has been copied forward. It becomes the single recovery copy and
            # any older one is dropped, so unfinished folders never pile up.
            # A finished build is never touched.
            recovery, removed = sceneray_splat._sr_retire_previous_version(
                root, previous
            )
            if recovery is not None:
                sceneray_splat._sr_repoint_dataset_references(
                    context.scene, previous, recovery
                )

        cfg.build_started_at = time.time()
        cfg.write_metadata_after_render = True

        if state == "COMPLETE" or self.requested_action == "NEW":
            message = (
                f"{target.name}: rebuilding from scratch; "
                f"{pending:,} camera(s) to render."
            )
        elif pending == 0:
            message = (
                f"{target.name}: reused {reused:,} image(s); rewriting "
                "cameras.txt and images.txt without rendering."
            )
        else:
            message = (
                f"{target.name}: {reused:,} image(s) carried forward, "
                f"{pending:,} camera(s) to render."
            )
        if recovery is not None:
            message += f" Recovery copy: {recovery.name}."
        if removed:
            message += f" Removed older recovery: {', '.join(removed)}."
        self.report({"INFO"}, message)

        # Rendering always comes first. The point cloud takes its colour from
        # the rendered images, so building it earlier produced a colourless
        # cloud; the render stage hands on to the metadata and then, for the
        # full workflow, to the point cloud.
        #
        # With nothing pending the render step finishes immediately and hands
        # straight on to the metadata, so one path covers both cases.
        return bpy.ops.sceneray_splat.render_images("INVOKE_DEFAULT")


class SPLATRAY_OT_build_images(_RenderStage, Operator):
    """Render every outdated camera, then write cameras.txt and images.txt"""

    bl_idname = "splatray.build_images"
    bl_label = "Render Images"
    bl_options = {"REGISTER"}

    workflow = 'RENDER'



class SPLATRAY_OT_build_dataset(_RenderStage, Operator):
    """Render images, write camera data, then generate the point cloud"""

    bl_idname = "splatray.build_dataset"
    bl_label = "Build Dataset"
    bl_options = {"REGISTER"}

    #: The only difference from Render Images: this one keeps going.
    workflow = 'FULL'



class SPLATRAY_OT_build_pointcloud(Operator):
    """Back-project the coloured point cloud from rendered RGB-D images.

    The points take their colour from the renders, so the images must exist
    first. To check coverage before rendering, use Check coverage
    """

    bl_idname = "splatray.build_pointcloud"
    bl_label = "Generate Point Cloud"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        cfg = _cfg(context)
        return cfg is not None and not is_busy(cfg)

    def execute(self, context):
        from .. import sceneray_splat

        cfg = context.scene.SCENERAY_SPLAT
        output_dir = sceneray_splat._sr_effective_output_path(cfg)
        if output_dir is None:
            # No build folder yet: make one, so the cloud can be inspected
            # before any image is rendered.
            try:
                sceneray_splat._sr_prepare_versioned_dataset(
                    context, cfg, reuse_images=True
                )
            except Exception as exc:
                self.report({"ERROR"}, f"Could not start the build: {exc}")
                return {"CANCELLED"}
        if not sceneray_splat._sr_has_queued_camera(cfg):
            self.report(
                {"ERROR"},
                "Add cameras to the queue before generating the point cloud.",
            )
            return {"CANCELLED"}
        # The points take their colour from the rendered images, so every one
        # of them has to exist and be readable first. Refusing here is what
        # stops a colourless cloud being produced quietly.
        missing = sceneray_splat._sr_unreadable_render_images(cfg)
        if missing:
            names = ", ".join(missing[:3])
            if len(missing) > 3:
                names += f" and {len(missing) - 3} more"
            self.report(
                {"ERROR"},
                f"Render the images first — missing or unreadable: {names}. "
                "To check coverage before rendering, use Check "
                "coverage.",
            )
            return {"CANCELLED"}
        cfg.build_workflow = 'POINTS'
        return bpy.ops.sceneray_splat.generate_points3d("INVOKE_DEFAULT")


# Rendering and the point cloud are one workflow again: Build Dataset runs
# them in order and does not start a step until the one before it succeeded.
# The individual operators remain registered so a script can still drive one
# stage, but the panel offers the single path.
CLASSES = (
    SPLATRAY_OT_build_dataset,
    SPLATRAY_OT_build_images,
    SPLATRAY_OT_build_pointcloud,
)
