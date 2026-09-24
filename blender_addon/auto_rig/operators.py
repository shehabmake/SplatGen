# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Operators: manage rig systems, calculate the cameras, remove them.

The calculation reads the scene on the main thread (Blender data is not
thread safe), plans on a worker thread (pure numpy, which releases the GIL
for its heavy work), then creates the cameras back on the main thread.
Blender stays responsive throughout; the shared job monitor shows each phase
with the usual Stop button, and the live preview draws the work in the
viewport as it happens.
"""

import threading
import time
import types
import uuid

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty
from mathutils import Matrix

from . import planner
from . import preview
from .properties import KIND_LABELS, SYSTEM_KINDS

#: Operation name in the shared progress report; ``progress.OWNER_AUTO_RIG``.
OPERATION = "Smart Camera Rig"
STAGE_SURVEY = "Scene survey"
STAGE_CREATE = "Create cameras"
STAGES = (STAGE_SURVEY, planner.STAGE_SPACE, planner.STAGE_VISIBILITY,
          planner.STAGE_SELECTION, STAGE_CREATE)

RIG_COLLECTION = "SplatGen Auto Rig"
BLOB_COLLECTION = "SplatGen Scan Blobs"
#: Custom property that identifies cameras made by the Smart camera rig.
RIG_TAG = "splatgen_auto_rig"

_cancel = {"requested": False}
#: True while a system is being built, so its property updates stay quiet.
_building = {"active": False}


def request_cancel():
    _cancel["requested"] = True


def is_running():
    from .. import progress
    return progress.is_owner((OPERATION,))


# ---------------------------------------------------------------------------
# Blobs: the scene objects behind Interior / Exterior systems
# ---------------------------------------------------------------------------

def _find_layer_collection(layer, collection):
    if layer.collection == collection:
        return layer
    for child in layer.children:
        found = _find_layer_collection(child, collection)
        if found is not None:
            return found
    return None


def _usable_collection(context):
    """The blob collection, guaranteed to be part of this View Layer.

    A blob in a collection that is excluded or hidden in the current View
    Layer - or that lives in another scene - exists but cannot be selected,
    which is what raised "cannot be selected because it is not in View
    Layer". The collection is linked here and made visible in this layer.
    """
    from .. import sceneray_splat as sr
    collection = sr._sr_get_or_create_collection(context.scene, BLOB_COLLECTION)
    collection.hide_viewport = False
    collection.hide_select = False
    layer = _find_layer_collection(context.view_layer.layer_collection, collection)
    if layer is not None:
        layer.exclude = False
        layer.hide_viewport = False
    return collection


def _in_view_layer(context, obj):
    try:
        return obj is not None and context.view_layer.objects.get(obj.name) == obj
    except ReferenceError:
        return False


def _select_only(context, obj):
    """Make ``obj`` the only selected, active object - quietly, if it can't be."""
    if not _in_view_layer(context, obj):
        return False
    try:
        for other in context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
    except RuntimeError:
        return False
    return True


def blob_alive(context, system):
    blob = system.blob
    try:
        return blob is not None and context.scene.objects.get(blob.name) == blob
    except ReferenceError:
        return False


def blob_radius(obj):
    scale = obj.matrix_world.to_scale()
    return float(obj.empty_display_size) * max(abs(scale.x), abs(scale.y), abs(scale.z))


def create_blob(context, system):
    """A sphere Empty at the 3D Cursor for this system, selected and ready to move."""
    scene = context.scene
    collection = _usable_collection(context)
    blob = bpy.data.objects.new(system.name, None)
    collection.objects.link(blob)
    # A freshly linked object joins the View Layer's list on its next update.
    context.view_layer.update()
    if not _in_view_layer(context, blob):
        # The collection could not be made part of this layer (for example a
        # library override): fall back to the scene's own collection.
        collection.objects.unlink(blob)
        scene.collection.objects.link(blob)
    blob.empty_display_type = 'SPHERE'
    blob.empty_display_size = 0.5
    blob.show_name = True
    # Drawn like any object, so walls and furniture hide it: placing a blob
    # inside a room is judged against what is really in front of it.
    blob.show_in_front = False
    blob.hide_render = True
    blob.location = scene.cursor.location
    blob.splatgen_blob.is_blob = True
    blob.splatgen_blob.mode = system.kind
    system.blob = blob
    _select_only(context, blob)
    return blob


def delete_blob(context, system):
    blob = system.blob
    system.blob = None
    if blob is None:
        return
    try:
        still_used = any(other.blob == blob for other in context.scene.splatgen_auto_rig.systems)
        if not still_used and blob.splatgen_blob.is_blob:
            from .. import sceneray_splat as sr
            sr._sr_remove_objects([blob])
    except ReferenceError:
        pass


def sync_system_blob(context, system, rename=False):
    """Give Interior/Exterior systems a live blob; take it away from Objects."""
    if _building["active"] or context is None or context.scene is None:
        return
    if system.kind in {'INTERIOR', 'EXTERIOR'}:
        if not blob_alive(context, system):
            create_blob(context, system)
        else:
            system.blob.splatgen_blob.mode = system.kind
            if rename and system.blob.name != system.name:
                system.blob.name = system.name
    elif system.blob is not None:
        delete_blob(context, system)


def select_active_system(context):
    """Selecting a system in the list selects what it captures in the viewport."""
    settings = context.scene.splatgen_auto_rig
    if not (0 <= settings.active_system < len(settings.systems)):
        return
    system = settings.systems[settings.active_system]
    if system.kind in {'INTERIOR', 'EXTERIOR'} and blob_alive(context, system):
        _select_only(context, system.blob)
    elif system.kind == 'OBJECT' and system.source == 'OBJECT' and system.target_object:
        _select_only(context, system.target_object)


def _next_name(settings, kind):
    base = "Object" if kind == 'OBJECT' else KIND_LABELS[kind]
    used = {system.name for system in settings.systems}
    number = 1
    while f"{base} {number}" in used:
        number += 1
    return f"{base} {number}"


def system_problem(context, system):
    """Why a system cannot be calculated, or '' when it is ready."""
    if system.kind in {'INTERIOR', 'EXTERIOR'}:
        return "" if blob_alive(context, system) else "Blob missing"
    if system.source == 'OBJECT':
        return "" if system.target_object is not None else "Pick an object"
    return "" if system.target_collection is not None else "Pick a collection"


def ready_systems(context):
    settings = context.scene.splatgen_auto_rig
    return [s for s in settings.systems if s.enabled and not system_problem(context, s)]


def cameras_asked(context):
    """Cameras the ready systems ask for by count (automatic ones ask for none)."""
    return sum(s.max_cameras for s in ready_systems(context))


def field_of_view(scene):
    """Half-angle tangents of the dataset cameras, from the global settings.

    Planning must see exactly what the rendered images will: the same lens,
    sensor and fit as ``_sr_apply_global_camera_settings`` writes to every
    queued camera, at the effective render resolution.
    """
    from .. import sceneray_splat as sr
    cfg = scene.SCENERAY_SPLAT
    lens = types.SimpleNamespace(
        lens=cfg.global_focal_length, sensor_width=cfg.global_sensor_width,
        sensor_height=cfg.global_sensor_height, sensor_fit=cfg.global_sensor_fit,
        shift_x=0.0, shift_y=0.0)
    fx, fy, _cx, _cy, _angle = sr.compute_intrinsics(lens, scene.render)
    w, h = sr.effective_resolution(scene.render)
    return w / (2.0 * fx), h / (2.0 * fy)


def _systems_to_plan(context, geometry):
    """The enabled, complete systems as planner blobs and object groups."""
    blobs, objects, skipped = [], [], []
    names = geometry.owner_names
    for system in ready_systems(context):
        if system.kind in {'INTERIOR', 'EXTERIOR'}:
            blob = system.blob
            custom = system.use_custom
            blobs.append(planner.Blob(
                system.name, tuple(blob.matrix_world.translation), blob_radius(blob),
                mode=system.kind, bounded=system.bounded, views=system.views,
                reach=system.reach, max_cameras=system.max_cameras,
                stop_when_covered=system.stop_when_covered,
                clearance=system.clearance if custom else None,
                seal=system.seal_size if custom else None,
                limit_height=system.limit_height if custom else None,
                height_min=system.height_min if custom else None,
                height_max=system.height_max if custom else None,
                allow_below=system.allow_below if custom else None))
            continue
        if system.source == 'OBJECT':
            wanted = {system.target_object.name}
        else:
            wanted = {obj.name for obj in system.target_collection.all_objects}
        owners = np.array([name in wanted for name in names], dtype=bool)
        if owners.any():
            objects.append(planner.ObjectGroup(
                system.name, owners, views=system.views, max_cameras=system.max_cameras,
                stop_when_covered=system.stop_when_covered,
                clearance=system.clearance if system.use_custom else None))
        else:
            skipped.append(system.name)
    if not blobs and not objects:
        raise planner.PlanError(
            "No system is ready: give each system a blob, an object or a collection "
            "with render-visible geometry.")
    return blobs, objects, skipped


SCENE_TAG = "splatgen_rig_scene"


def scene_uid(scene):
    """A lasting id for a scene - names change, pointers do not survive saving."""
    uid = scene.get(SCENE_TAG)
    if not uid:
        uid = scene[SCENE_TAG] = uuid.uuid4().hex
    return uid


def auto_rig_cameras(scene):
    """The calculated cameras that belong to this scene.

    Only the scene that calculated a camera replaces it. A camera from a
    file saved before cameras carried their scene counts as this scene's
    only when no other scene shows it - older files could share the rig
    collection between scenes.
    """
    # Read only: this runs in button polls while panels draw, where writing
    # to the scene is not allowed. A scene without an id tagged no camera.
    uid = scene.get(SCENE_TAG)
    mine = []
    for obj in scene.objects:
        if obj.type != 'CAMERA' or obj.get(RIG_TAG) is None:
            continue
        owner = obj.get(SCENE_TAG)
        if (owner is not None and owner == uid) or (
                owner is None and all(s == scene for s in obj.users_scene)):
            mine.append(obj)
    return mine


# ---------------------------------------------------------------------------
# Creating and removing rig cameras
# ---------------------------------------------------------------------------

def clear_auto_rig(context):
    """Delete every Auto-rig camera and take it out of the render queue."""
    from .. import sceneray_splat as sr
    scene = context.scene
    cfg = scene.SCENERAY_SPLAT
    cameras = auto_rig_cameras(scene)
    if not cameras:
        return 0
    doomed = {camera.as_pointer() for camera in cameras}
    names = {camera.name for camera in cameras}
    sr._sr_bulk["busy"] = True
    try:
        for index in range(len(cfg.camera_queue) - 1, -1, -1):
            camera = cfg.camera_queue[index].camera
            if camera is not None and camera.as_pointer() in doomed:
                cfg.camera_queue.remove(index)
        cfg.active_camera_index = min(cfg.active_camera_index,
                                      max(0, len(cfg.camera_queue) - 1))
        sr._sr_remove_objects(cameras)
    finally:
        sr._sr_bulk["busy"] = False
    sr._sr_prune_camera_from_manifests(cfg, names)
    sr._sr_mark_scene_cache_dirty(scene)
    return len(cameras)


def create_cameras(context, result, settings):
    """Turn a plan into queued cameras. Returns (created, removed)."""
    from .. import sceneray_splat as sr
    scene = context.scene
    cfg = scene.SCENERAY_SPLAT
    removed = clear_auto_rig(context) if settings.replace_previous else 0
    collection = sr._sr_get_or_create_collection(scene, RIG_COLLECTION)
    run = uuid.uuid4().hex[:8]
    first_index = len(cfg.camera_queue)
    number = 1
    prefix = sr._MANAGED_CAMERA_PREFIX
    sr._sr_bulk["busy"] = True
    created = []
    try:
        for position, forward, up in zip(result.positions, result.forwards, result.ups):
            while f"{prefix}_{number:04d}" in bpy.data.objects:
                number += 1
            name = f"{prefix}_{number:04d}"
            number += 1
            camera = bpy.data.objects.new(name, bpy.data.cameras.new(name))
            collection.objects.link(camera)
            sr._sr_apply_global_camera_settings(cfg, camera)
            right = np.cross(forward, up)
            # Blender cameras look down -Z with +Y up.
            camera.matrix_world = Matrix((
                (right[0], up[0], -forward[0], position[0]),
                (right[1], up[1], -forward[1], position[1]),
                (right[2], up[2], -forward[2], position[2]),
                (0.0, 0.0, 0.0, 1.0)))
            camera[RIG_TAG] = run
            camera[SCENE_TAG] = scene_uid(scene)
            sr._sr_queue_add(cfg, camera, apply_settings=False)
            created.append(camera)
    finally:
        sr._sr_bulk["busy"] = False
    if created:
        cfg.active_camera_index = first_index if first_index < len(cfg.camera_queue) else 0
    sr._sr_mark_scene_cache_dirty(scene)
    return len(created), removed


def _summary(result, removed):
    s = result.stats
    kinds = ", ".join(sorted({r["kind"].title() for r in result.regions}))
    text = (f"{s['cameras']} cameras · {s['covered']:.0%} of visible surfaces well covered"
            f" · {kinds} · {s['seconds']:.1f}s")
    if removed:
        text += f" · replaced {removed}"
    return text


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class SPLATGEN_OT_auto_rig_add_system(bpy.types.Operator):
    """Add a system to the Smart camera rig. Interior and Exterior place a blob at
    the 3D Cursor to move into the space; Object / Collection orbits chosen objects"""
    bl_idname = "sceneray_splat.auto_rig_add_system"
    bl_label = "Add System"
    bl_options = {'REGISTER', 'UNDO'}

    kind: EnumProperty(name="Type", items=SYSTEM_KINDS, default='INTERIOR')

    @classmethod
    def poll(cls, context):
        return context.scene is not None and context.mode == 'OBJECT'

    def execute(self, context):
        settings = context.scene.splatgen_auto_rig
        _building["active"] = True
        try:
            system = settings.systems.add()
            system.name = _next_name(settings, self.kind)
            system.kind = self.kind
            if self.kind == 'OBJECT':
                selected = context.active_object
                if selected is not None and selected.type not in {'EMPTY', 'CAMERA', 'LIGHT'}:
                    system.target_object = selected
        finally:
            _building["active"] = False
        if self.kind in {'INTERIOR', 'EXTERIOR'}:
            create_blob(context, system)
        index = len(settings.systems) - 1
        _building["active"] = True
        try:
            settings.active_system = index
        finally:
            _building["active"] = False
        if self.kind == 'OBJECT':
            self.report({'INFO'}, f"{system.name} added. Pick the object or collection to capture.")
        else:
            where = "inside the room" if self.kind == 'INTERIOR' else "outside the building"
            self.report({'INFO'}, f"{system.name} added at the 3D Cursor. Move its blob {where}.")
        return {'FINISHED'}


class SPLATGEN_OT_auto_rig_remove_system(bpy.types.Operator):
    """Remove this system, and its blob from the scene"""
    bl_idname = "sceneray_splat.auto_rig_remove_system"
    bl_label = "Remove System"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(min=0)

    @classmethod
    def poll(cls, context):
        return bool(context.scene.splatgen_auto_rig.systems)

    def execute(self, context):
        settings = context.scene.splatgen_auto_rig
        if self.index >= len(settings.systems):
            return {'CANCELLED'}
        system = settings.systems[self.index]
        name = system.name
        delete_blob(context, system)
        settings.systems.remove(self.index)
        _building["active"] = True
        try:
            settings.active_system = min(settings.active_system, max(0, len(settings.systems) - 1))
        finally:
            _building["active"] = False
        self.report({'INFO'}, f"Removed {name}.")
        return {'FINISHED'}


class SPLATGEN_OT_auto_rig_place_blob(bpy.types.Operator):
    """Select this system's blob, creating it again if it was deleted. With Shift,
    move it to the 3D Cursor"""
    bl_idname = "sceneray_splat.auto_rig_place_blob"
    bl_label = "Select Blob"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(min=0)
    to_cursor: bpy.props.BoolProperty(name="To 3D Cursor", default=False)

    def invoke(self, context, event):
        self.to_cursor = self.to_cursor or event.shift
        return self.execute(context)

    def execute(self, context):
        settings = context.scene.splatgen_auto_rig
        if self.index >= len(settings.systems):
            return {'CANCELLED'}
        system = settings.systems[self.index]
        if system.kind not in {'INTERIOR', 'EXTERIOR'}:
            return {'CANCELLED'}
        if not blob_alive(context, system):
            create_blob(context, system)
            self.report({'INFO'}, f"New blob for {system.name} at the 3D Cursor.")
        elif self.to_cursor:
            system.blob.location = context.scene.cursor.location
        if not _select_only(context, system.blob):
            self.report({'WARNING'}, f"{system.blob.name} is hidden in this View Layer.")
        return {'FINISHED'}


class SPLATGEN_OT_auto_rig_generate(bpy.types.Operator):
    """Calculate the camera rig for every enabled system together: rooms, exteriors
    and objects. Every surface gets several good, distinct views, with occlusion,
    distance and viewing angle taken into account"""
    bl_idname = "sceneray_splat.auto_rig_generate"
    bl_label = "Calculate Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from ..building_data import stages
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        settings = getattr(context.scene, "splatgen_auto_rig", None)
        return (cfg is not None and settings is not None and context.mode == 'OBJECT'
                and bool(ready_systems(context)) and not stages.anything_running(context))

    # ---- the three phases ------------------------------------------------
    def _survey(self, context):
        from . import scene_proxy
        settings = context.scene.splatgen_auto_rig
        blobs = {s.blob.name for s in settings.systems if s.blob is not None}
        geometry = scene_proxy.extract(context, settings.geometry_collection, exclude=blobs)
        if geometry.is_empty:
            raise planner.PlanError("No render-visible geometry was found to scan.")
        plan_blobs, plan_objects, skipped = _systems_to_plan(context, geometry)
        self.skipped = skipped
        tan_h, tan_v = field_of_view(context.scene)
        self.request = planner.Request(
            geometry, blobs=plan_blobs, objects=plan_objects,
            tan_h=tan_h, tan_v=tan_v, quality=settings.quality,
            max_cameras=settings.max_cameras,
            clearance=settings.clearance, seal_size=settings.seal_size,
            limit_height=settings.limit_height, height_min=settings.height_min,
            height_max=settings.height_max, allow_below=settings.allow_below,
            seed=settings.seed)
        return geometry

    def _plan(self):
        state = self.state
        try:
            state["result"] = planner.plan(
                self.request,
                progress=lambda stage, fraction, message: state.update(
                    stage=stage,
                    fraction=state["fraction"] if fraction is None else fraction,
                    message=message or state["message"]),
                cancelled=lambda: _cancel["requested"],
                visual=preview.feed if self.live else None)
        except planner.Cancelled:
            state["error"] = "CANCELLED"
        except planner.PlanError as exc:
            state["error"] = str(exc)
        except Exception as exc:     # a bug must end the job, not hang it
            import traceback
            traceback.print_exc()
            state["error"] = f"Planning failed: {exc}"
        finally:
            state["done"] = True

    def _finish(self, context):
        from .. import progress
        settings = context.scene.splatgen_auto_rig
        result = self.state["result"]
        progress.set_stage(STAGE_CREATE, message=f"Creating {len(result.positions)} cameras")
        created, removed = create_cameras(context, result, settings)
        summary = _summary(result, removed)
        warnings = list(result.warnings)
        if self.skipped:
            warnings.insert(0, "Skipped (no render-visible geometry): " + ", ".join(self.skipped))
        settings.last_summary = summary
        settings.last_warning = " ".join(warnings[:2])
        progress.end(summary)
        preview.finish()
        for warning in warnings:
            self.report({'WARNING'}, warning)
        self.report({'INFO'}, f"Smart camera rig: {summary}")
        print(f"[SplatGen] Smart camera rig: {summary} | stats {result.stats}")
        return {'FINISHED'}

    def _fail(self, context, message):
        from .. import progress
        cancelled = message == "CANCELLED"
        text = "Camera calculation cancelled." if cancelled else message
        context.scene.splatgen_auto_rig.last_warning = "" if cancelled else text
        progress.end(text, outcome="CANCELLED" if cancelled else "FAILED")
        preview.clear()
        self.report({'WARNING'} if cancelled else {'ERROR'}, text)
        return {'CANCELLED'}

    # ---- running ---------------------------------------------------------
    def _start(self, context):
        from .. import progress
        _cancel["requested"] = False
        self.skipped = []
        self.live = bool(context.scene.splatgen_auto_rig.live_preview) and not bpy.app.background
        self.state = {"stage": STAGE_SURVEY, "fraction": 0.0, "message": "",
                      "done": False, "result": None, "error": None}
        self.started = time.time()
        progress.begin(OPERATION, STAGES)
        progress.update(message="Reading render-visible geometry")
        if self.live:
            preview.begin()

    def execute(self, context):
        """Synchronous path, used in background mode and from scripts."""
        self._start(context)
        try:
            self._survey(context)
        except planner.PlanError as exc:
            return self._fail(context, str(exc))
        self._plan()
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
                # Deferred one tick so the job monitor is drawn before the
                # (main-thread) geometry read begins.
                if _cancel["requested"]:
                    self._cleanup(context)
                    return self._fail(context, "CANCELLED")
                geometry = self._survey(context)
                progress.update(fraction=1.0, message=(
                    f"{len(geometry.owner_names):,} objects · "
                    f"{geometry.triangles:,} triangles"))
                self.phase = 'PLAN'
                self.worker = threading.Thread(target=self._plan, daemon=True)
                self.worker.start()
                return {'RUNNING_MODAL'}
            state = self.state
            if state["stage"] != progress.current()["stage"] and state["stage"] in STAGES:
                progress.set_stage(state["stage"])
            elapsed = time.time() - self.started
            detail = f"{elapsed:.0f}s elapsed"
            cameras = preview.camera_count()
            if cameras:
                detail = f"{cameras} cameras placed · " + detail
            progress.update(fraction=state["fraction"], message=state["message"], detail=detail)
            preview.redraw()
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
            return self._fail(context, f"Camera calculation failed: {exc}")

    def _cleanup(self, context):
        timer = getattr(self, "_timer", None)
        if timer is not None:
            context.window_manager.event_timer_remove(timer)
            self._timer = None


class SPLATGEN_OT_auto_rig_clear(bpy.types.Operator):
    """Delete the cameras created by the Smart camera rig and remove them from the
    render queue. Cameras you placed yourself are kept"""
    bl_idname = "sceneray_splat.auto_rig_clear"
    bl_label = "Remove Calculated Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from ..building_data import stages
        return (context.scene is not None and not stages.anything_running(context)
                and bool(auto_rig_cameras(context.scene)))

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        removed = clear_auto_rig(context)
        context.scene.splatgen_auto_rig.last_summary = ""
        self.report({'INFO'}, f"Removed {removed} calculated camera(s).")
        return {'FINISHED'}


CLASSES = (
    SPLATGEN_OT_auto_rig_add_system,
    SPLATGEN_OT_auto_rig_remove_system,
    SPLATGEN_OT_auto_rig_place_blob,
    SPLATGEN_OT_auto_rig_generate,
    SPLATGEN_OT_auto_rig_clear,
)
