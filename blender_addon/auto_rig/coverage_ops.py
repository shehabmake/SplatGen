# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Camera coverage: settings, operators, the viewport overlay and its card.

Check coverage grades every surface the queued cameras should capture (see
``coverage``) and paints the result onto the scene as small tiles lying on
the surfaces: green for good coverage, yellow for fair coverage, orange
for weak coverage and red where no camera looks. Views, Angles
and Detail show the three measurements behind that verdict in bands, with
each band's share of the surface in the viewport legend. Problem areas get
numbered badges in the viewport and a row each in the card - frame it, or
let the add-on place cameras for it - and cameras that add nothing needed
can be selected in one click.

The overlay draws onto the surfaces themselves, hidden by what is in front
of them (or through everything with X-ray). It is not scene data: nothing is
saved in the .blend and nothing can render.
"""

import threading

import bpy
import gpu
import numpy as np
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       IntProperty, StringProperty)
from bpy.types import PropertyGroup
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from .. import theme, palette
from . import coverage, planner

OPERATION = "Camera Coverage"
STAGES = (coverage.STAGE_SCENE, coverage.STAGE_CAMERAS, coverage.STAGE_SUMMARY)
FIX_TAG = "splatgen_coverage_fix"
FIX_COLLECTION = "SplatGen Coverage Fixes"

RED = palette.rgb(palette.RED)
ORANGE = palette.rgb(palette.ORANGE)
YELLOW = palette.rgb(palette.YELLOW)
GREEN = palette.rgb(palette.GREEN)
BLUE = palette.rgb(palette.BLUE)
# Existing numeric bands stay intact; surplus-view bands share brand blue.
COLOURS = {coverage.GOOD: GREEN, coverage.FAIR: YELLOW, coverage.WEAK: ORANGE,
           coverage.UNSEEN: RED}
TITLES = {
    'GRADE': ("Coverage", "Good · fair · weak · unseen, from views, angles and detail"),
    'VIEWS': ("Views per surface", "How many cameras see each surface"),
    'DETAIL': ("Detail", "Size of one image pixel on the surface, from its sharpest view"),
}

_cancel = {"requested": False}
_state = {"result": None, "signature": None, "handles": [], "cache": {}}


def request_cancel():
    _cancel["requested"] = True


def result():
    return _state["result"]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _redraw_all(self=None, context=None):
    _state["cache"].clear()
    for window in getattr(bpy.context.window_manager, "windows", ()):
        for area in window.screen.areas:
            if area.type in {'VIEW_3D', 'PROPERTIES'}:
                area.tag_redraw()


def _show_changed(self, context):
    if self.show and _state["result"] is not None:
        _add_handlers()
    else:
        _remove_handlers()
    _redraw_all()


def _target_changed(self, context):
    """Re-grade the last check for the new target - no new check needed."""
    found = _state["result"]
    if found is not None:
        coverage.grade(found, self.target_views)
    sync_areas(context.scene)
    _redraw_all()


def _area_picked(self, context):
    """Picking a problem area in the list frames it in the viewport."""
    frame_area(context, self.active_area)


def sync_areas(scene):
    """Copy the last check's problem areas into the list the card shows."""
    try:
        settings = scene.splatgen_coverage
    except (AttributeError, ReferenceError):
        return
    found = _state["result"]
    settings.areas.clear()
    for number, area in enumerate(found.areas if found is not None else (), 1):
        item = settings.areas.add()
        item.number = number
        item.issue = area["issue"].capitalize()
        item.share = area["share"]
        item.unseen = area["issue"] == "never seen"


class SplatGenProblemArea(PropertyGroup):
    """One row of the problem-area list (a copy; the check result is not saved)."""

    number: IntProperty()
    issue: StringProperty()
    share: FloatProperty()
    unseen: BoolProperty()


class SplatGenCoverageSettings(PropertyGroup):
    display: EnumProperty(
        name="Show",
        items=[
            ('GRADE', "Grade", "The verdict for every surface: good, fair, weak or unseen",
             'CHECKMARK', 0),
            ('VIEWS', "Views", "How many cameras see each surface, in bands from none to far "
             "more than needed - brand blue identifies excess views",
             'CAMERA_DATA', 1),
            ('DETAIL', "Detail", "How large one image pixel is on each surface in its "
             "sharpest view, compared with the rest of the capture", 'IMAGE_DATA', 3),
        ],
        default='GRADE', update=_redraw_all)
    problems_only: BoolProperty(
        name="Problems Only",
        description="Hide well-covered surfaces so only what needs attention remains",
        default=False, update=_redraw_all)
    xray: BoolProperty(
        name="X-Ray",
        description="Show the result through walls and objects",
        default=False, update=_redraw_all)
    show: BoolProperty(
        name="Show Coverage", default=True,
        description="Show the coverage result in the viewport", update=_show_changed)
    show_display_settings: BoolProperty(name="Display settings", default=False)
    show_redundant: BoolProperty(
        name="Mark Redundant Cameras",
        description="Mark the cameras that add nothing needed with blue badges in the viewport",
        default=False, update=_redraw_all)
    dot_size: FloatProperty(
        name="Tile Size",
        description="Size of the coloured tiles on the surfaces. 1 closes the gaps between them",
        default=1.0, min=0.2, max=3.0, update=_redraw_all)
    target_views: IntProperty(
        name="Views Needed",
        description=(
            "Views from different directions a surface needs to count as good. 3 trains "
            "clean Splats. Changing it re-grades the last check at once"
        ),
        default=3, min=1, max=12, update=_target_changed)
    quality: EnumProperty(
        name="Detail",
        items=[('DRAFT', "Draft", "Fewer surface points, a few seconds"),
               ('STANDARD', "Standard", "Balanced"),
               ('HIGH', "High", "Dense surface points and camera images. Slower")],
        default='STANDARD')
    areas: CollectionProperty(type=SplatGenProblemArea, options={'SKIP_SAVE'})
    active_area: IntProperty(
        name="Problem Area", default=0, min=0, options={'SKIP_SAVE'}, update=_area_picked,
        description="Pick a problem area to frame it in the 3D Viewport")


# ---------------------------------------------------------------------------
# Reading the queue (main thread)
# ---------------------------------------------------------------------------

def queued_cameras(context):
    """The queued cameras as coverage.Camera, each pose once.

    Returns (cameras, skipped non-perspective names, duplicate names).
    """
    from .. import sceneray_splat as sr
    scene = context.scene
    cfg = scene.SCENERAY_SPLAT
    width, height = sr.effective_resolution(scene.render)
    cameras, seen, skipped, duplicates = [], set(), [], []
    for item in cfg.camera_queue:
        camera = item.camera
        if camera is None or camera.type != 'CAMERA':
            continue
        if camera.data.type != 'PERSP':
            skipped.append(camera.name)
            continue
        pose, _scaled = sr.camera_pose_world(camera)
        matrix = np.array(pose, dtype=np.float64)
        key = tuple(np.round(matrix[:3, :].ravel(), 5))
        if key in seen:
            duplicates.append(camera.name)
            continue
        seen.add(key)
        fx, fy, cx, cy, _angle = sr.compute_intrinsics(camera.data, scene.render)
        cameras.append(coverage.Camera(camera.name, matrix, fx, fy, cx, cy, width, height,
                                       camera.data.clip_start, camera.data.clip_end))
    return cameras, skipped, duplicates


def queue_signature(context):
    cfg = context.scene.SCENERAY_SPLAT
    parts = []
    for item in cfg.camera_queue:
        camera = item.camera
        if camera is None:
            continue
        try:
            parts.append((camera.name, tuple(round(v, 4) for v in camera.matrix_world.translation),
                          round(camera.data.lens, 3)))
        except (ReferenceError, AttributeError):
            continue
    return hash(tuple(parts))


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class SPLATGEN_OT_coverage_check(bpy.types.Operator):
    """Check how well the queued cameras cover the scene: every surface they should
    capture is graded by how many cameras see it, from how many directions and in how
    much detail, and weak places are grouped into problem areas to fix"""
    bl_idname = "sceneray_splat.coverage_check"
    bl_label = "Check Coverage"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        from .. import sceneray_splat as sr
        from ..building_data import stages
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and sr._sr_has_queued_camera(cfg)
                and not stages.anything_running(context))

    def _survey(self, context):
        from . import scene_proxy
        cameras, skipped, duplicates = queued_cameras(context)
        if not cameras:
            raise planner.PlanError("No perspective cameras in the queue to check.")
        settings = context.scene.splatgen_coverage
        rig = context.scene.splatgen_auto_rig
        blobs = {s.blob.name for s in rig.systems if s.blob is not None}
        geometry = scene_proxy.extract(context, rig.geometry_collection, exclude=blobs)
        self.skipped = skipped
        self.signature = queue_signature(context)
        self.job = dict(geometry=geometry, cameras=cameras, quality=settings.quality,
                        target_views=settings.target_views, seal_size=rig.seal_size,
                        duplicates=duplicates)
        return geometry, cameras

    def _run(self):
        state = self.state
        job = self.job
        try:
            state["result"] = coverage.analyse(
                job["geometry"], job["cameras"], quality=job["quality"],
                target_views=job["target_views"], seal_size=job["seal_size"],
                duplicates=job["duplicates"],
                progress=lambda stage, fraction, message: state.update(
                    stage=stage,
                    fraction=state["fraction"] if fraction is None else fraction,
                    message=message or state["message"]),
                cancelled=lambda: _cancel["requested"])
        except planner.Cancelled:
            state["error"] = "CANCELLED"
        except planner.PlanError as exc:
            state["error"] = str(exc)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            state["error"] = f"Coverage check failed: {exc}"
        finally:
            state["done"] = True

    def _start(self, context):
        from .. import progress
        _cancel["requested"] = False
        self.skipped = []
        self.state = {"stage": coverage.STAGE_SCENE, "fraction": 0.0, "message": "",
                      "done": False, "result": None, "error": None}
        progress.begin(OPERATION, STAGES)
        progress.update(message="Reading render-visible geometry")

    def _finish(self, context):
        from .. import progress
        found = self.state["result"]
        _state["result"] = found
        _state["signature"] = self.signature
        _state["cache"].clear()
        settings = context.scene.splatgen_coverage
        sync_areas(context.scene)
        settings.show = True
        _add_handlers()
        _redraw_all()
        areas = len(found.areas)
        summary = (f"Coverage checked: {found.stats['cameras']} cameras, "
                   f"{areas} problem area{'s' if areas != 1 else ''}, "
                   f"{len(found.redundant)} redundant camera{'s' if len(found.redundant) != 1 else ''}")
        progress.end(summary)
        for warning in found.warnings:
            self.report({'WARNING'}, warning)
        if self.skipped:
            self.report({'WARNING'}, f"Skipped non-perspective cameras: {', '.join(self.skipped[:5])}")
        self.report({'INFO'}, summary)
        return {'FINISHED'}

    def _fail(self, context, message):
        from .. import progress
        cancelled = message == "CANCELLED"
        text = "Coverage check cancelled." if cancelled else message
        progress.end(text, outcome="CANCELLED" if cancelled else "FAILED")
        self.report({'WARNING'} if cancelled else {'ERROR'}, text)
        return {'CANCELLED'}

    def execute(self, context):
        self._start(context)
        try:
            self._survey(context)
        except planner.PlanError as exc:
            return self._fail(context, str(exc))
        self._run()
        if self.state["error"]:
            return self._fail(context, self.state["error"])
        return self._finish(context)

    def invoke(self, context, event):
        if bpy.app.background:
            return self.execute(context)
        self._start(context)
        self.phase = 'SURVEY'
        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        from .. import progress
        if event.type == 'ESC' and event.value == 'PRESS':
            request_cancel()
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        try:
            if self.phase == 'SURVEY':
                if _cancel["requested"]:
                    self._cleanup(context)
                    return self._fail(context, "CANCELLED")
                _geometry, cameras = self._survey(context)
                progress.update(fraction=1.0, message=f"{len(cameras)} distinct cameras")
                self.phase = 'RUN'
                threading.Thread(target=self._run, daemon=True).start()
                return {'RUNNING_MODAL'}
            state = self.state
            if state["stage"] != progress.current()["stage"] and state["stage"] in STAGES:
                progress.set_stage(state["stage"])
            progress.update(fraction=state["fraction"], message=state["message"])
            if not state["done"]:
                return {'RUNNING_MODAL'}
            self._cleanup(context)
            if state["error"]:
                return self._fail(context, state["error"])
            return self._finish(context)
        except planner.PlanError as exc:
            self._cleanup(context)
            return self._fail(context, str(exc))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._cleanup(context)
            return self._fail(context, f"Coverage check failed: {exc}")

    def _cleanup(self, context):
        timer = getattr(self, "_timer", None)
        if timer is not None:
            context.window_manager.event_timer_remove(timer)
            self._timer = None


class SPLATGEN_OT_coverage_clear(bpy.types.Operator):
    """Remove the coverage result from the viewport"""
    bl_idname = "sceneray_splat.coverage_clear"
    bl_label = "Clear Coverage"
    bl_options = {'REGISTER'}

    def execute(self, context):
        clear()
        return {'FINISHED'}


def _view3d(context):
    screen = context.screen or (context.window.screen if context.window else None)
    areas = [a for a in getattr(screen, "areas", ()) if a.type == 'VIEW_3D']
    if not areas:
        return None
    return max(areas, key=lambda a: a.width * a.height)


class SPLATGEN_OT_coverage_frame(bpy.types.Operator):
    """Point the 3D Viewport at this problem area"""
    bl_idname = "sceneray_splat.coverage_frame"
    bl_label = "Frame Problem Area"
    bl_options = {'REGISTER'}

    index: IntProperty(min=0)

    def execute(self, context):
        return {'FINISHED'} if frame_area(context, self.index) else {'CANCELLED'}


def frame_area(context, index):
    """Point the largest 3D Viewport at problem area ``index``."""
    found = _state["result"]
    area = _view3d(context)
    if found is None or area is None or not 0 <= index < len(found.areas):
        return False
    problem = found.areas[index]
    view = area.spaces.active.region_3d
    centre = np.asarray(problem["centre"], dtype=np.float64)
    distance = max(problem["size"] * 2.5, found.view_distance * 0.8)
    normal = np.asarray(problem["normal"], dtype=np.float64)
    if problem["normal_confidence"] > 0.3 and found.grid is not None:
        # Look at the area from its open side, a little from above, and no
        # farther than the open space in front of it: from behind a wall
        # the viewport would show the wall instead.
        out = normal + np.array([0.0, 0.0, 0.35])
        out /= np.linalg.norm(out)
        start = centre + out * (2.0 * found.voxel)
        free = coverage._march_from(found.field, found.grid, start, out[None], np.array([distance]))[0]
        if np.isfinite(free):
            distance = max(min(distance, 0.9 * free), 4.0 * found.voxel)
        view.view_rotation = Vector(-out).to_track_quat('-Z', 'Y')
    view.view_location = tuple(centre)
    view.view_distance = distance
    area.tag_redraw()
    return True


class SPLATGEN_UL_problem_areas(bpy.types.UIList):
    """Numbered like the viewport badges; picking one frames it."""
    bl_idname = "SPLATGEN_UL_problem_areas"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.label(text=f"{item.number}   {item.issue}",
                  **theme.icon_args('ERROR' if item.unseen else 'TIME'))
        op = theme.operator(row, SPLATGEN_OT_coverage_frame.bl_idname, text="",
                            icon='ZOOM_SELECTED', emboss=False)
        op.index = index


def _add_fix_camera(context, name, position, forward, up):
    from .. import sceneray_splat as sr
    scene = context.scene
    cfg = scene.SCENERAY_SPLAT
    collection = sr._sr_get_or_create_collection(scene, FIX_COLLECTION)
    camera = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    collection.objects.link(camera)
    sr._sr_apply_global_camera_settings(cfg, camera)
    right = np.cross(forward, up)
    camera.matrix_world = Matrix((
        (right[0], up[0], -forward[0], position[0]),
        (right[1], up[1], -forward[1], position[1]),
        (right[2], up[2], -forward[2], position[2]),
        (0.0, 0.0, 0.0, 1.0)))
    camera[FIX_TAG] = True
    sr._sr_queue_add(cfg, camera, apply_settings=False)
    return camera


class SPLATGEN_OT_coverage_fix(bpy.types.Operator):
    """Place cameras that look at weak areas from clear, varied directions, add them to
    the queue, and check coverage again"""
    bl_idname = "sceneray_splat.coverage_fix"
    bl_label = "Add Cameras for Problem Areas"
    # Kept for a later release, but not offered: hidden from F3 search too.
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    index: IntProperty(default=-1, description="Problem area to fix; -1 fixes all of them")

    @classmethod
    def poll(cls, context):
        found = _state["result"]
        return found is not None and bool(found.areas) and found.grid is not None

    def execute(self, context):
        from .. import sceneray_splat as sr
        found = _state["result"]
        targets = (range(len(found.areas)) if self.index < 0
                   else [self.index] if self.index < len(found.areas) else [])
        prefix = sr._MANAGED_CAMERA_PREFIX
        number, added, unreachable = 1, 0, 0
        sr._sr_bulk["busy"] = True
        try:
            for k in targets:
                area = found.areas[k]
                want = int(np.clip(found.target_views - round(area["views"]), 1, 3))
                poses = coverage.fix_poses(found, area, want)
                if not poses:
                    unreachable += 1
                for position, forward, up in poses:
                    while f"{prefix}_{number:04d}" in bpy.data.objects:
                        number += 1
                    _add_fix_camera(context, f"{prefix}_{number:04d}", position, forward, up)
                    number += 1
                    added += 1
        finally:
            sr._sr_bulk["busy"] = False
        sr._sr_mark_scene_cache_dirty(context.scene)
        if not added:
            self.report({'WARNING'}, "No clear viewpoint was found for those areas; "
                                     "place a camera there by hand.")
            return {'CANCELLED'}
        message = f"Added {added} camera(s) for weak areas."
        if unreachable:
            message += f" {unreachable} area(s) had no clear viewpoint."
        self.report({'INFO'}, message)
        # Show the effect straight away.
        if bpy.app.background:
            bpy.ops.sceneray_splat.coverage_check()
        else:
            bpy.app.timers.register(_recheck, first_interval=0.1)
        return {'FINISHED'}


def _recheck():
    window = bpy.context.window_manager.windows[0] if bpy.context.window_manager.windows else None
    if window is None:
        return None
    with bpy.context.temp_override(window=window):
        if bpy.ops.sceneray_splat.coverage_check.poll():
            bpy.ops.sceneray_splat.coverage_check('INVOKE_DEFAULT')
    return None


class SPLATGEN_OT_coverage_select_redundant(bpy.types.Operator):
    """Select the cameras that can go without any surface losing views it needs or
    angle between views. They were chosen together, so deleting all of them is safe"""
    bl_idname = "sceneray_splat.coverage_select_redundant"
    bl_label = "Select Redundant Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        found = _state["result"]
        return found is not None and bool(found.redundant)

    def execute(self, context):
        names = set(_state["result"].redundant)
        for obj in context.selected_objects:
            obj.select_set(False)
        chosen = 0
        for obj in context.scene.objects:
            if obj.name in names and obj.name in context.view_layer.objects:
                try:
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    chosen += 1
                except RuntimeError:
                    pass
        self.report({'INFO'}, f"Selected {chosen} redundant camera(s). Press X to delete them.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Colour bands
# ---------------------------------------------------------------------------

def _length(metres):
    if metres < 0.01:
        return f"{metres * 1000:.1f} mm"
    if metres < 1.0:
        return f"{metres * 100:.1f} cm"
    return f"{metres:.2f} m"


def bands(found, display):
    """(band index per point, [(label, colour)]) for one display mode."""
    views = found.views
    if display == 'VIEWS':
        k = max(1, found.target_views)
        rows = [(0, "none", RED)]
        if k > 1:
            rows.append((1, "1", ORANGE))
        if k > 2:
            rows.append((2, "2" if k == 3 else f"2–{k - 1}", YELLOW))
        for low, colour in ((k, GREEN), (2 * k, BLUE), (4 * k, BLUE)):
            high = 2 * low - 1
            rows.append((low, str(low) if high == low else f"{low}–{high}", colour))
        rows.append((8 * k, f"{8 * k}+", BLUE))
        lows = np.array([row[0] for row in rows])
        index = np.searchsorted(lows, views, side='right') - 1
        return index, [(label, colour) for _low, label, colour in rows]
    if display == 'DETAIL':
        median = found.stats.get("median_detail") or 1.0
        ratio = np.where(np.isfinite(found.detail), found.detail / median, np.inf)
        steps = (1.0, 1.5, 2.0, 3.0)
        index = np.select([ratio <= s for s in steps] + [np.isfinite(ratio)], [0, 1, 2, 3, 4], 5)
        labels = [f"≤ {_length(median * s)}" for s in steps] + [f"> {_length(median * 3.0)}"]
        return index, list(zip(labels, (BLUE, GREEN, YELLOW, ORANGE, RED))) + [("unseen", RED)]
    order = [coverage.GOOD, coverage.FAIR, coverage.WEAK, coverage.UNSEEN]
    lookup = np.zeros(4, dtype=np.int64)
    lookup[order] = np.arange(4)
    return lookup[found.grade.astype(np.int64)], [(coverage.CLASS_NAMES[c], COLOURS[c]) for c in order]


def band_shares(found, display):
    index, rows = bands(found, display)
    shares = np.bincount(index, weights=found.weights, minlength=len(rows))
    total = float(found.weights.sum()) or 1.0
    return [(label, colour, float(share) / total) for (label, colour), share in zip(rows, shares)]


# ---------------------------------------------------------------------------
# Viewport overlay
# ---------------------------------------------------------------------------

def _linear(rgb):
    """Display colour to the linear colour Blender's overlays expect.

    Without this the viewport's view transform brightens every colour and the
    greens and oranges wash out to pastels on light surfaces.
    """
    return np.power(np.clip(rgb, 0.0, 1.0), 2.2)


def _shader(name, fallback):
    try:
        return gpu.shader.from_builtin(name)
    except (ValueError, SystemError):
        return gpu.shader.from_builtin(fallback)


def _tiles(found, keep, colours, size):
    """Small squares lying on the surfaces, one per checked point.

    Sized to the spacing between points, so together they paint the
    surfaces; lifted a little off the surface on the side that was
    photographed, so the scene's own depth hides them correctly.
    """
    points = found.points[keep].astype(np.float64)
    normals = found.normals[keep].astype(np.float64)
    helper = np.where(np.abs(normals[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    across = np.cross(normals, helper)
    across /= np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-12)
    along = np.cross(normals, across)
    cell = found.cells[keep][:, None].astype(np.float64) if found.cells is not None else found.cell
    half = 0.5 * cell * size
    # Each point is the only one in its cell of a world grid: centring its
    # tile on that cell (on the surface's plane) paints walls and floors as
    # a clean mosaic instead of scattered, overlapping squares.
    lattice = (np.floor(points / cell) + 0.5) * cell
    lattice -= normals * np.einsum('ij,ij->i', lattice - points, normals)[:, None]
    centre = lattice + normals * (0.3 * found.voxel + 0.002)
    corners = np.stack([centre + (-across - along) * half, centre + (across - along) * half,
                        centre + (across + along) * half, centre + (-across + along) * half], axis=1)
    rgba = np.concatenate([_linear(colours[keep]), np.ones((int(keep.sum()), 1))], axis=1)
    first = np.arange(len(points), dtype=np.int32)[:, None, None] * 4
    indices = (first + np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)[None]).reshape(-1, 3)
    return (corners.reshape(-1, 3).astype(np.float32),
            np.repeat(rgba, 4, axis=0).astype(np.float32), indices)


def _hidden_owners(context, found):
    """Objects of the result that are hidden in the viewport right now.

    Their tiles hide with them - hide a ceiling to look into a room.
    """
    objects = context.view_layer.objects
    hidden = []
    for index in _state["cache"].setdefault("present", np.unique(found.owners).tolist()):
        if index >= len(found.owner_names):
            continue
        obj = objects.get(found.owner_names[index])
        if obj is not None and not obj.visible_get():
            hidden.append(index)
    return tuple(hidden)


def _batch(context):
    found = _state["result"]
    settings = context.scene.splatgen_coverage
    hidden = _hidden_owners(context, found)
    key = (id(found), settings.display, settings.problems_only, round(settings.dot_size, 3),
           found.target_views, hidden)
    cached = _state["cache"].get("tiles")
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    index, rows = bands(found, settings.display)
    table = np.array([colour for _label, colour in rows], dtype=np.float64)
    colours = table[index]
    keep = np.ones(len(found.points), dtype=bool)
    if settings.problems_only:
        keep = found.grade < coverage.GOOD
    if hidden:
        keep &= ~np.isin(found.owners, hidden)
    shader = _shader('SMOOTH_COLOR', 'FLAT_COLOR')
    pos, rgba, indices = _tiles(found, keep, colours, settings.dot_size)
    batch = batch_for_shader(shader, 'TRIS', {"pos": pos, "color": rgba}, indices=indices)
    _state["cache"]["tiles"] = (key, shader, batch)
    return shader, batch


def _on_prepare_page(context):
    """The overlay belongs to Prepare; on Train and View it only gets in the way."""
    cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
    return cfg is None or getattr(cfg, "workspace_step", 'PREPARE') == 'PREPARE'


def _draw_view():
    context = bpy.context
    found = _state["result"]
    settings = getattr(context.scene, "splatgen_coverage", None)
    if found is None or settings is None or not settings.show or not len(found.points):
        return
    if not _on_prepare_page(context):
        return
    shader, batch = _batch(context)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE' if settings.xray else 'LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    gpu.state.face_culling_set('NONE')
    batch.draw(shader)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def _disc(shader, x, y, radius, colour, alpha):
    ring = [(x + radius * np.cos(a), y + radius * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24)]
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": [(x, y)] + ring})
    shader.uniform_float("color", (*_linear(np.array(colour)), alpha))
    batch.draw(shader)


def _draw_pixel():
    import blf
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    context = bpy.context
    found = _state["result"]
    settings = getattr(context.scene, "splatgen_coverage", None)
    if found is None or settings is None or not settings.show:
        return
    if not _on_prepare_page(context):
        return
    region, view = context.region, context.region_data
    scale = context.preferences.system.ui_scale
    font = 0
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    # Redundant cameras, on request: grey minus badges.
    if settings.show_redundant and found.redundant:
        wanted = set(found.redundant)
        for name, position in zip(found.camera_names, found.camera_positions):
            if name not in wanted:
                continue
            spot = location_3d_to_region_2d(region, view, position)
            if spot is None:
                continue
            _disc(shader, spot.x, spot.y, 7 * scale, BLUE, 0.9)
            _disc(shader, spot.x, spot.y, 5 * scale, (0.15, 0.15, 0.17), 0.9)
            blf.size(font, 11 * scale)
            blf.color(font, 0.85, 0.85, 0.9, 1)
            width, height = blf.dimensions(font, "–")
            blf.position(font, spot.x - width / 2, spot.y - height / 2 + 1, 0)
            blf.draw(font, "–")
    # Numbered badges on the problem areas.
    for number, area in enumerate(found.areas, 1):
        spot = location_3d_to_region_2d(region, view, area["centre"])
        if spot is None:
            continue
        colour = RED if area["issue"] == "never seen" else ORANGE
        # A white rim keeps the badge readable over tiles of its own colour.
        _disc(shader, spot.x, spot.y, 13 * scale, (1.0, 1.0, 1.0), 0.95)
        _disc(shader, spot.x, spot.y, 11 * scale, colour, 1.0)
        blf.size(font, 12 * scale)
        text = str(number)
        width, height = blf.dimensions(font, text)
        blf.color(font, 1, 1, 1, 1)
        blf.position(font, spot.x - width / 2, spot.y - height / 2 + 1, 0)
        blf.draw(font, text)
    _draw_legend(context, found, settings, scale)
    gpu.state.blend_set('NONE')


def _draw_legend(context, found, settings, scale):
    """Bottom-left: what the colours mean and how much surface each covers."""
    import blf
    font = 0
    region = context.region
    title, hint = TITLES[settings.display]
    if settings.display in {'GRADE', 'VIEWS'}:
        hint += f" · {found.target_views} needed"
    entries = [(f"{label} {share:.0%}", colour)
               for label, colour, share in band_shares(found, settings.display)
               if share >= 0.0005]
    blf.size(font, 11 * scale)
    lines, line, width = [], [], 0.0
    room = region.width - 40 * scale
    for text, colour in entries:
        w = blf.dimensions(font, f"● {text}")[0] + 14 * scale
        if line and width + w > room:
            lines.append(line)
            line, width = [], 0.0
        line.append((text, colour))
        width += w
    if line:
        lines.append(line)
    x = 18 * scale
    y = 26 * scale + 17 * scale * len(lines)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 5, 0, 0, 0, 0.8)
    blf.size(font, 13 * scale)
    blf.color(font, 1, 1, 1, 1)
    blf.position(font, x, y + 20 * scale, 0)
    blf.draw(font, f"Coverage · {title}" if settings.display != 'GRADE' else title)
    blf.size(font, 10 * scale)
    blf.color(font, 0.8, 0.8, 0.82, 1)
    blf.position(font, x, y + 5 * scale, 0)
    blf.draw(font, hint)
    blf.size(font, 11 * scale)
    for row, items in enumerate(lines):
        offset = x
        yy = y - 12 * scale - 17 * scale * row
        for text, colour in items:
            blf.color(font, *_linear(np.array(colour)), 1.0)
            blf.position(font, offset, yy, 0)
            blf.draw(font, "●")
            dot = blf.dimensions(font, "● ")[0]
            blf.color(font, 0.92, 0.92, 0.94, 1.0)
            blf.position(font, offset + dot, yy, 0)
            blf.draw(font, text)
            offset += dot + blf.dimensions(font, text)[0] + 14 * scale
    blf.disable(font, blf.SHADOW)


def _add_handlers():
    if _state["handles"]:
        return
    _state["handles"] = [
        (bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), 'WINDOW', 'POST_VIEW'), 'WINDOW'),
        (bpy.types.SpaceView3D.draw_handler_add(_draw_pixel, (), 'WINDOW', 'POST_PIXEL'), 'WINDOW'),
    ]


def _remove_handlers():
    for handle, region in _state["handles"]:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, region)
        except (ValueError, RuntimeError):
            pass
    _state["handles"] = []


def clear():
    _remove_handlers()
    _state["result"] = None
    _state["signature"] = None
    _state["cache"].clear()
    try:
        sync_areas(bpy.context.scene)
    except (AttributeError, RuntimeError):
        pass            # restricted context while loading or unregistering
    _redraw_all()


@bpy.app.handlers.persistent
def _on_load(*_args):
    clear()


# ---------------------------------------------------------------------------
# The Camera coverage card
# ---------------------------------------------------------------------------

def draw_card(layout, context):
    """Camera coverage in Prepare: one button, then the result.

    Viewing options appear only once there is a result to view. Placing
    cameras for problem areas (``SPLATGEN_OT_coverage_fix``) stays
    registered but is not offered: fill gaps by recalculating the Smart
    camera rig or by adding cameras by hand.
    """
    from .. import progress
    from ..building_data import stages
    scene = context.scene
    settings = scene.splatgen_coverage
    found = _state["result"]
    box = layout.box()
    header = box.row(align=True)
    theme.icon_label(header, "2  ·  Check coverage", custom="section_coverage", fallback='VIEWZOOM')
    if found is not None:
        tools = header.row(align=True)
        tools.alignment = 'RIGHT'
        tools.prop(settings, "show", text="", emboss=False,
                   **theme.icon_args('HIDE_OFF' if settings.show else 'HIDE_ON',
                                     'action_hide' if settings.show else 'stage_viewer'))
        theme.operator(tools, SPLATGEN_OT_coverage_clear.bl_idname, text="", icon='X')

    stale = found is not None and _state["signature"] != queue_signature(context)
    run = box.row(align=True)
    run.scale_y = 1.5
    run.enabled = not stages.anything_running(context) and any(item.camera for item in scene.SCENERAY_SPLAT.camera_queue)
    theme.accent(run, found is None or stale)
    theme.operator(run, SPLATGEN_OT_coverage_check.bl_idname,
                   text="Check again" if stale else "Check coverage", custom="action_coverage")
    if progress.is_owner((OPERATION,)):
        return
    if found is None:
        return

    # Summary.
    summary = box.box().column(align=True)
    if stale:
        theme.status(summary, "The camera queue changed since this check.", theme.STATUS_WAIT)
    theme.icon_label(summary, f"{found.stats['cameras']} cameras checked", custom="stage_cameras")
    good = found.fractions.get('Good', 0.0)
    progress._draw_bar(summary, good, f'{good:.0%} good coverage')
    legend = summary.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
    for label, icon in (('Good', 'status_ok'), ('Fair', 'coverage_fair'),
                        ('Weak', 'coverage_weak'), ('Unseen', 'status_error')):
        theme.icon_label(legend, f"{label}  {found.fractions.get(label, 0.0):.0%}", custom=icon)
    count = len(found.areas)
    if count:
        theme.status(summary, f"{count} problem area{'s' if count != 1 else ''} to look at",
                     theme.STATUS_ERROR)
    else:
        theme.status(summary, "No problem areas", theme.STATUS_OK)

    # How to look at it.
    view = box.column(align=True)
    theme.heading(view, "View", custom="stage_viewer", fallback='HIDE_OFF')
    modes = view.row(align=True)
    modes.scale_y = 1.2
    modes.prop(settings, "display", expand=True)
    toggles = view.row(align=True)
    toggles.prop(settings, "problems_only", toggle=True, icon='ERROR')
    toggles.prop(settings, "xray", toggle=True, icon='XRAY')
    form = theme.disclosure(box, settings, 'show_display_settings', 'Display settings')
    if form is not None:
        form.prop(settings, "target_views", text="Views needed")
        form.prop(settings, "dot_size", text="Tile size", slider=True)
        theme.muted(form, f"Scope: {found.scope}.")

    # Problem areas, numbered like the viewport badges.
    if count:
        areas = box.column(align=True)
        theme.heading(areas, f"Problem areas · {count}", custom="status_error", fallback='ERROR')
        areas.template_list("SPLATGEN_UL_problem_areas", "", settings, "areas",
                            settings, "active_area", rows=min(6, max(3, count)), maxrows=8)


    # Cameras that can go.
    if found.redundant:
        spare = box.column(align=True)
        n = len(found.redundant)
        theme.heading(spare, f"Redundant cameras · {n}", fallback='INFO')
        theme.muted(spare, "Removing all of them together costs no surface a view it needs.")
        row = spare.row(align=True)
        row.scale_y = 1.15
        row.prop(settings, "show_redundant", text="Mark", toggle=True,
                 **theme.icon_args('HIDE_OFF' if settings.show_redundant else 'HIDE_ON',
                                   'action_hide' if settings.show_redundant else 'stage_viewer'))
        theme.operator(row, SPLATGEN_OT_coverage_select_redundant.bl_idname,
                       text="Select", icon='RESTRICT_SELECT_OFF')


CLASSES = (
    SplatGenProblemArea,
    SplatGenCoverageSettings,
    SPLATGEN_UL_problem_areas,
    SPLATGEN_OT_coverage_check,
    SPLATGEN_OT_coverage_clear,
    SPLATGEN_OT_coverage_frame,
    SPLATGEN_OT_coverage_fix,
    SPLATGEN_OT_coverage_select_redundant,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.splatgen_coverage = bpy.props.PointerProperty(type=SplatGenCoverageSettings)
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)


def unregister():
    request_cancel()
    clear()
    while _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    if hasattr(bpy.types.Scene, "splatgen_coverage"):
        del bpy.types.Scene.splatgen_coverage
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
