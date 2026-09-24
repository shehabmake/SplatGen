# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""The raw export stage: everything the legacy render did not produce.

It runs after the legacy build (or on its own, from Export Raw Data) as a
list of steps::

    beauty   render any view whose raw beauty passes are missing
    clay     the clay render of every view
    cameras  cameras.json and a copy of the COLMAP files
    scene    scene_description.json
    mesh     scene_mesh.ply
    voxels   collision_voxels.npz
    world    the world panorama and source HDRI
    probes   the reflection probe panoramas
    manifest manifest.json, written last

In the interface each render is started with INVOKE_DEFAULT and polled from
a timer, so Blender stays responsive, the same way the legacy batch works.
In background mode the same steps run straight through with EXEC_DEFAULT.
A failing step is recorded in the manifest and the stage carries on; the
legacy dataset is never touched.
"""

import shutil
import time
import traceback
from pathlib import Path

import bpy
from bpy.app.handlers import persistent

from . import geometry, layout, metadata, properties, sessions

PHASE = "Export Raw Data"

_runtime = {"stage": None, "timer": False}
_job = {"waiting": False, "started": False, "complete": False,
        "cancelled": False, "launched": 0.0, "idle_since": None}


def is_running():
    return _runtime["stage"] is not None


def _log(message):
    print(f"[splatgen raw] {message}")


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

class _RenderStep:
    """A group of renders sharing one session (one setup, one restore)."""

    kind = "render"

    def __init__(self, name, label):
        self.name = name
        self.label = label

    def setup(self, stage):
        """Return (session, items); each item is a dict with a 'stem'."""
        raise NotImplementedError

    def prepare(self, stage, session, item):
        pass

    def finish(self, stage, item, published, missing):
        if missing:
            stage.issues.append(f"{self.name} {item['stem']}: missing {', '.join(missing)}")

    def teardown(self, stage):
        pass

    def render_scene(self, stage):
        return stage.scene


class _ViewStep(_RenderStep):
    """beauty / clay: one render per dataset view that lacks its files."""

    def __init__(self, name, label, builder):
        super().__init__(name, label)
        self.builder = builder

    def setup(self, stage):
        session = self.builder(stage)
        keys = list(session.outputs)
        items = []
        for frame in stage.frames:
            if layout.view_complete(stage.root, frame["index"], keys):
                continue
            if not metadata.camera_matches(frame, stage.cfg):
                stage.notes.append(
                    f"{self.name}: '{frame['camera_name']}' changed or is gone since "
                    f"{frame['stem']} was rendered; not re-rendered so passes stay aligned"
                )
                continue
            items.append({"stem": frame["stem"], "frame": frame,
                          "camera": bpy.data.objects[frame["camera_name"]]})
        if items:
            scene, original = stage.scene, stage.scene.camera
            session.on_restore(lambda: setattr(scene, "camera", original))
        return session, items

    def prepare(self, stage, session, item):
        stage.scene.camera = item["camera"]


class _WorldStep(_RenderStep):
    def setup(self, stage):
        if stage.scene.world is None:
            stage.notes.append("world: the scene has no world")
            return None, []
        session = sessions.begin_world(stage.scene, stage.root)
        stage.world_scene = session.scene
        return session, [{"stem": "world"}]

    def render_scene(self, stage):
        return stage.world_scene

    def finish(self, stage, item, published, missing):
        super().finish(stage, item, published, missing)
        raw = stage.raw
        folder = stage.root / layout.WORLD_FOLDER
        copied = metadata.copy_environment_images(stage.scene.world, folder)
        info = {
            "render": "world_equirect.exr" if "world" in published else None,
            "resolution": [int(raw.world_resolution) * 2, int(raw.world_resolution)],
            "projection": metadata.CONVENTIONS["equirect"],
            "engine": "CYCLES",
            "samples": int(raw.aux_samples) or "scene",
            "source_images": copied,
            "world": metadata._world_info(stage.scene.world),
            "note": "The world alone: no objects or lights. Linear scene radiance.",
        }
        sessions.write_json(stage.root / layout.WORLD_INFO, info)
        stage.files["world"] = layout.WORLD_INFO


class _ProbeStep(_RenderStep):
    def setup(self, stage):
        probes = []
        for obj in stage.scene.objects:
            data = getattr(obj, "data", None)
            if obj.type == "LIGHT_PROBE" and str(getattr(data, "type", "")) in {"SPHERE", "CUBEMAP"}:
                probes.append({"source": "scene", "name": obj.name_full,
                               "position": [float(v) for v in obj.matrix_world.translation],
                               "clip_start": float(getattr(data, "clip_start", 0.001))})
        if not probes and stage.raw.probe_count > 0:
            grid = stage.voxel_grid()
            positions = geometry.probe_positions(
                grid, int(stage.raw.probe_count), stage.camera_positions()
            ) if grid is not None else []
            for position in positions:
                probes.append({"source": "automatic", "name": "",
                               "position": position, "clip_start": 0.001})
        if not probes:
            stage.notes.append("probes: no probe position found")
            return None, []
        session = sessions.begin_probes(stage.scene, stage.view_layer, stage.root)
        items = []
        for index, probe in enumerate(probes):
            items.append({"stem": f"probe_{index:03d}", "probe": probe})
        stage.probe_records = []
        return session, items

    def prepare(self, stage, session, item):
        camera = session.probe_camera
        camera.location = item["probe"]["position"]
        camera.data.clip_start = max(1e-4, float(item["probe"]["clip_start"]))
        camera.data.clip_end = 1.0e6

    def finish(self, stage, item, published, missing):
        super().finish(stage, item, published, missing)
        record = dict(item["probe"])
        record["id"] = item["stem"]
        record["files"] = {key: f"{item['stem']}/{Path(path).name}"
                           for key, path in published.items()}
        stage.probe_records.append(record)

    def teardown(self, stage):
        raw = stage.raw
        sessions.write_json(stage.root / layout.PROBES_INFO, {
            "format": "SplatGen reflection probes",
            "resolution": [int(raw.probe_resolution) * 2, int(raw.probe_resolution)],
            "projection": metadata.CONVENTIONS["equirect"],
            "coordinate_frame": metadata.CONVENTIONS["world_frame"],
            "radiance": "linear RGB of the whole scene seen from the probe centre",
            "distance": "Cycles Depth pass of the panoramic camera (distance from the probe centre)",
            "engine": "CYCLES",
            "samples": int(raw.aux_samples) or "scene",
            "probes": getattr(stage, "probe_records", []),
        })
        stage.files["probes"] = layout.PROBES_INFO


class _ComputeStep:
    kind = "compute"

    def __init__(self, name, label, function):
        self.name = name
        self.label = label
        self.function = function


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------

class RawStage:
    def __init__(self, scene, build, finished_message=""):
        self.scene = scene
        self.view_layer = bpy.context.view_layer
        if getattr(self.view_layer, "id_data", scene) != scene:
            self.view_layer = scene.view_layers[0]
        self.cfg = scene.SCENERAY_SPLAT
        self.raw = properties.settings(scene)
        self.build = Path(build)
        self.root = layout.raw_root(build)
        self.finished_message = finished_message
        self.frames, self.resolution, self.color_management = metadata.rendered_frames(build)
        self.issues, self.notes, self.files = [], [], {}
        self.mesh = None
        self._grid = None
        self._grid_done = False
        self.world_scene = None
        self.window_pointer = None
        self.session = None
        self.items = []
        self.item_index = 0
        self.step_index = 0
        self.finished = False
        self.render_display = None
        self.steps = self._plan()

    # -- plan ----------------------------------------------------------------

    def _plan(self):
        raw = self.raw
        steps = [
            _ViewStep("beauty", "Rendering missing beauty passes",
                      lambda stage: sessions.begin_beauty(
                          stage.scene, stage.view_layer, stage.root,
                          own_tree=True, resolution=stage.resolution)),
        ]
        if raw.clay:
            steps.append(_ViewStep("clay", "Rendering clay views",
                                   lambda stage: sessions.begin_clay(
                                       stage.scene, stage.view_layer, stage.root,
                                       stage.resolution)))
        steps.append(_ComputeStep("cameras", "Writing cameras", self._cameras))
        steps.append(_ComputeStep("scene", "Describing the scene", self._describe))
        if raw.scene_mesh:
            steps.append(_ComputeStep("mesh", "Exporting the scene mesh", self._mesh))
        if raw.collision_voxels:
            steps.append(_ComputeStep("voxels", "Building collision voxels", self._voxels))
        if raw.world:
            steps.append(_WorldStep("world", "Rendering the world"))
        if raw.probes:
            steps.append(_ProbeStep("probes", "Rendering reflection probes"))
        steps.append(_ComputeStep("manifest", "Writing the manifest", self._manifest))
        return steps

    # -- shared data ---------------------------------------------------------

    def id_map(self):
        return sessions.read_json(self.root / layout.ID_MAP) or {}

    def camera_positions(self):
        positions = []
        for frame in self.frames:
            camera = bpy.data.objects.get(frame["camera_name"])
            if camera is not None:
                positions.append([float(v) for v in camera.matrix_world.translation])
        return positions

    def scene_mesh(self):
        if self.mesh is None:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            self.mesh = geometry.collect_scene_mesh(
                self.scene, self.view_layer, depsgraph, self.id_map()) or False
        return self.mesh or None

    def voxel_grid(self):
        if not self._grid_done:
            self._grid_done = True
            mesh = self.scene_mesh()
            if mesh is not None:
                resolution = int(self.raw.voxel_resolution) if self.raw.collision_voxels else 64
                self._grid = geometry.voxelize(mesh, self.camera_positions(), resolution)
        return self._grid

    # -- compute steps -------------------------------------------------------

    def _cameras(self):
        metadata.write_cameras(self.scene, self.build, self.root,
                               self.frames, self.resolution)
        self.files["cameras"] = layout.CAMERAS_JSON
        self.files["colmap"] = layout.COLMAP_FOLDER

    def _describe(self):
        capture_info = sessions.read_json(
            layout.work_dir(self.root) / f"capture_{layout.BEAUTY}.json") or {}
        previous = sessions.read_json(self.root / layout.MANIFEST) or {}
        reports = capture_info.get("materials") or previous.get("materials") or {}
        description = metadata.scene_description(self.scene, self.id_map(), reports)
        sessions.write_json(self.root / layout.SCENE_DESCRIPTION, description)
        self.files["scene_description"] = layout.SCENE_DESCRIPTION
        if (self.root / layout.ID_MAP).is_file():
            self.files["id_map"] = layout.ID_MAP

    def _mesh(self):
        mesh = self.scene_mesh()
        if mesh is None:
            self.notes.append("scene mesh: no render-visible geometry")
            return
        geometry.write_ply(mesh, self.root / layout.SCENE_MESH)
        lo, hi = geometry.mesh_bounds(mesh)
        sessions.write_json(self.root / layout.SCENE_MESH_INFO, {
            "format": "SplatGen scene mesh",
            "file": Path(layout.SCENE_MESH).name,
            "coordinate_frame": metadata.CONVENTIONS["world_frame"],
            "evaluation": "dependency-graph evaluated (modifiers, geometry nodes, "
                          "instances) at viewport levels, render-visible objects only",
            "vertex_properties": ["x", "y", "z", "nx", "ny", "nz"],
            "face_properties": ["vertex_indices", "object_id", "material_id"],
            "ids": "ids/id_map.json",
            "vertex_count": int(len(mesh["vertices"])),
            "triangle_count": int(len(mesh["triangles"])),
            "bounds": [lo, hi],
            "objects": mesh["objects"],
        })
        self.files["scene_mesh"] = layout.SCENE_MESH
        self.files["scene_mesh_info"] = layout.SCENE_MESH_INFO

    def _voxels(self):
        grid = self.voxel_grid()
        if grid is None:
            self.notes.append("collision voxels: no render-visible geometry")
            return
        geometry.write_voxels(grid, self.root / layout.VOXELS,
                              self.root / layout.VOXELS_INFO,
                              {"requested_resolution": int(self.raw.voxel_resolution)})
        self.files["collision_voxels"] = layout.VOXELS
        self.files["collision_voxels_info"] = layout.VOXELS_INFO

    def _manifest(self):
        metadata.write_readme(self.root)
        manifest = metadata.write_manifest(
            self.scene, self.build, self.root, self.frames, self.resolution,
            self.color_management, self.files, self.notes, self.issues,
        )
        self.manifest_status = manifest["status"]
        work = layout.work_dir(self.root)
        if work.is_dir():
            shutil.rmtree(work, ignore_errors=True)

    # -- driving -----------------------------------------------------------

    def fraction(self):
        total = max(1, len(self.steps))
        within = 0.0
        if self.items:
            within = self.item_index / max(1, len(self.items))
        return min(1.0, (self.step_index + within) / total)

    def _report(self, message, detail=""):
        from .. import progress

        self.cfg.render_status = message
        progress.update(fraction=self.fraction(), message=message, detail=detail)

    def cancel_requested(self):
        from .. import sceneray_splat

        return bool(sceneray_splat._cancel_flag.get("requested"))

    def tick(self, background=False):
        """Advance; returns a delay for the next tick, or None when done."""
        if self.finished:
            return None
        if _job["waiting"]:
            state = _poll_render(self)
            if state == "wait":
                return 0.1
            self._complete_item(state == "complete")
            return 0.05
        if self.cancel_requested():
            self._end(cancelled=True)
            return None
        if self.step_index >= len(self.steps):
            self._end()
            return None
        step = self.steps[self.step_index]
        if step.kind == "compute":
            self._report(step.label)
            try:
                step.function()
            except Exception as exc:
                traceback.print_exc()
                self.issues.append(f"{step.name}: {exc}")
            self.step_index += 1
            return 0.05
        if self.session is None and not self.items:
            self._report(step.label)
            try:
                self.session, self.items = step.setup(self)
            except Exception as exc:
                traceback.print_exc()
                self.issues.append(f"{step.name}: could not prepare ({exc})")
                self.session, self.items = None, []
            self.item_index = 0
            if not self.items:
                self._teardown(step)
                return 0.05
        if self.item_index < len(self.items):
            item = self.items[self.item_index]
            self._report(step.label, f"{self.item_index + 1} of {len(self.items)}")
            try:
                step.prepare(self, self.session, item)
                self.session.prepare_view(item["stem"])
                _launch(self, step.render_scene(self), background)
            except Exception as exc:
                traceback.print_exc()
                self.issues.append(f"{step.name} {item['stem']}: {exc}")
                _job["waiting"] = False
                self.item_index += 1
            return 0.1
        self._teardown(step)
        return 0.05

    def _complete_item(self, rendered):
        step = self.steps[self.step_index]
        item = self.items[self.item_index]
        _job["waiting"] = False
        if rendered:
            try:
                published, missing = self.session.finish_view(item["stem"])
                step.finish(self, item, published, missing)
            except Exception as exc:
                traceback.print_exc()
                self.issues.append(f"{step.name} {item['stem']}: {exc}")
        else:
            self.issues.append(f"{step.name} {item['stem']}: render was cancelled")
        self.item_index += 1

    def _teardown(self, step):
        try:
            step.teardown(self)
        except Exception as exc:
            traceback.print_exc()
            self.issues.append(f"{step.name}: {exc}")
        if self.session is not None:
            self.session.restore()
        self.session = None
        self.items = []
        self.item_index = 0
        self.world_scene = None
        self.step_index += 1

    def _end(self, cancelled=False, error=""):
        from .. import progress

        if self.finished:
            return
        self.finished = True
        if self.session is not None:
            self.session.restore()
            self.session = None
        _restore_render_display(self)
        self.cfg.is_rendering = False
        self.cfg.build_workflow = 'NONE'
        if error:
            message = f"Raw export stopped: {error}"
            self.cfg.render_status = message
            progress.fail(message)
        elif cancelled:
            message = "Raw export cancelled - run Export Raw Data to finish it."
            self.cfg.render_status = message
            progress.end(message)
        else:
            status = getattr(self, "manifest_status", "incomplete")
            message = (f"Raw dataset {status} ({len(self.frames)} views)"
                       + (f"; {len(self.issues)} issue(s) - see manifest.json"
                          if self.issues else ""))
            if self.finished_message:
                message = f"{self.finished_message} · {message}"
            self.cfg.render_status = message
            progress.end(message)
        _log(message)
        for issue in self.issues:
            _log(f"  issue: {issue}")
        _runtime["stage"] = None
        try:
            from .. import sceneray_splat
            sceneray_splat._cancel_flag["requested"] = False
            sceneray_splat._tag_redraw_sceneray_splat(bpy.context)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _launch(stage, scene, background):
    if background:
        result = bpy.ops.render.render("EXEC_DEFAULT", write_still=False,
                                       scene=scene.name)
        _job.update(waiting=False)
        stage._complete_item("FINISHED" in result)
        return
    from .. import sceneray_splat

    window, area, region = sceneray_splat._sr_live_window_context(stage.window_pointer)
    if window is None:
        raise RuntimeError("no Blender window is available to render in")
    override = {"window": window, "screen": window.screen, "scene": scene}
    if area is not None:
        override["area"] = area
    if region is not None:
        override["region"] = region
    _job.update(waiting=True, started=False, complete=False, cancelled=False,
                launched=time.time(), idle_since=None)
    with bpy.context.temp_override(**override):
        result = bpy.ops.render.render("INVOKE_DEFAULT", write_still=False,
                                       scene=scene.name)
    if "FINISHED" in result:
        _job.update(started=True, complete=True)
    elif "CANCELLED" in result:
        _job["cancelled"] = True


def _poll_render(stage):
    from .. import sceneray_splat

    now = time.time()
    if sceneray_splat._sr_render_job_running():
        _job["started"] = True
        _job["idle_since"] = None
        return "wait"
    if not (_job["complete"] or _job["cancelled"]):
        if not _job["started"] and now - _job["launched"] < 3.0:
            return "wait"
        # The job ended without a handler call reaching us; trust the files.
        stem = stage.items[stage.item_index]["stem"]
        written = any(any(output["folder"].glob(f".{stem}.pending_*.exr"))
                      for output in stage.session.outputs.values())
        _job["complete" if written else "cancelled"] = True
    if _job["idle_since"] is None:
        _job["idle_since"] = now
        return "wait"
    if now - _job["idle_since"] < 0.35:
        return "wait"
    return "complete" if _job["complete"] and not _job["cancelled"] else "cancelled"


@persistent
def _on_render_init(*_args):
    if _job["waiting"]:
        _job["started"] = True


@persistent
def _on_render_complete(*_args):
    if _job["waiting"]:
        _job["complete"] = True


@persistent
def _on_render_cancel(*_args):
    if _job["waiting"]:
        _job["cancelled"] = True


def _hide_render_display(stage):
    try:
        view = bpy.context.preferences.view
        stage.render_display = view.render_display_type
        view.render_display_type = "NONE"
    except (AttributeError, TypeError):
        stage.render_display = None


def _restore_render_display(stage):
    if stage.render_display is None:
        return
    try:
        bpy.context.preferences.view.render_display_type = stage.render_display
    except (AttributeError, TypeError):
        pass
    stage.render_display = None


def _timer():
    stage = _runtime["stage"]
    if stage is None:
        _runtime["timer"] = False
        return None
    try:
        delay = stage.tick(background=False)
    except Exception as exc:
        traceback.print_exc()
        stage._end(error=str(exc))
        delay = None
    if delay is None:
        _runtime["timer"] = False
    return delay


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def should_run(scene, workflow):
    return properties.enabled(scene) and workflow in {"RENDER", "FULL", "RAW"}


def start(context, build, finished_message=""):
    """Run the raw stage on ``build``. False if it cannot start."""
    from .. import progress

    if is_running():
        return False
    scene = context.scene
    if not properties.enabled(scene) or build is None:
        return False
    stage = RawStage(scene, build, finished_message)
    if not stage.frames:
        _log("no rendered views in this build; raw export skipped")
        return False
    if not progress.is_active():
        progress.begin(PHASE, (PHASE,))
    progress.set_stage(PHASE, message="Preparing the raw dataset")
    stage.root.mkdir(parents=True, exist_ok=True)
    stage.cfg.is_rendering = True
    _runtime["stage"] = stage
    if bpy.app.background:
        while True:
            try:
                if stage.tick(background=True) is None:
                    break
            except Exception as exc:
                traceback.print_exc()
                stage._end(error=str(exc))
                break
        return True
    try:
        stage.window_pointer = context.window.as_pointer()
    except (AttributeError, ReferenceError):
        stage.window_pointer = None
    _hide_render_display(stage)
    if not _runtime["timer"]:
        _runtime["timer"] = True
        bpy.app.timers.register(_timer, first_interval=0.05)
    return True


def abort(reason="Raw export stopped."):
    stage = _runtime["stage"]
    if stage is None:
        return
    try:
        stage._end(error=reason)
    except Exception:
        traceback.print_exc()
    _runtime["stage"] = None
    _job["waiting"] = False


@persistent
def _on_load_post(*_args):
    # The scene the stage was working on is gone; drop it without touching data.
    stage = _runtime["stage"]
    _runtime["stage"] = None
    _job["waiting"] = False
    if stage is not None:
        _restore_render_display(stage)
    from . import capture
    try:
        capture.purge_leftovers()
    except Exception:
        pass


_HANDLERS = (
    ("render_init", _on_render_init),
    ("render_complete", _on_render_complete),
    ("render_cancel", _on_render_cancel),
    ("load_post", _on_load_post),
)


def register():
    for name, handler in _HANDLERS:
        collection = getattr(bpy.app.handlers, name)
        if handler not in collection:
            collection.append(handler)


def unregister():
    abort("SplatGen add-on disabled.")
    try:
        if bpy.app.timers.is_registered(_timer):
            bpy.app.timers.unregister(_timer)
    except (ReferenceError, RuntimeError, ValueError):
        pass
    _runtime["timer"] = False
    for name, handler in _HANDLERS:
        collection = getattr(bpy.app.handlers, name)
        try:
            collection.remove(handler)
        except ValueError:
            pass
