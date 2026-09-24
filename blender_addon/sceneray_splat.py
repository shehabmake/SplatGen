"""
SplatGen — Blender → 3D Gaussian Splatting / COLMAP dataset exporter.

Renders exactly the Camera objects you place in the scene (one image per
camera), writes their true poses and intrinsics, and reconstructs a coloured
point cloud by back-projecting the rendered RGB and metric-depth maps. No
camera or render setting is ever modified.


THE COORDINATE CONTRACT  (read this before changing anything)
=============================================================
Everything this dataset exporter writes lives in ONE frame, the "export frame".
There is exactly one transform from Blender world space to it:

    X_export = M_export_from_world @ X_world

`M_export_from_world` is a single 4x4 matrix. It is computed ONCE, by
`export_transform_from_cameras()`, at the moment the dataset text is
written (step 2), and it is PERSISTED into the dataset manifest. Step 3
does not recompute it, does not measure it, does not infer it — it reads
it back and applies the identical matrix to the point cloud.

Every exported entity goes through that one matrix:

    camera poses   c2w_export = M @ c2w_world          (step 2)
    point cloud    P_export   = M @ P_world            (step 3)

That is the whole coordinate story for the WORLD frame. There is no
handedness change, no Z-up→Y-up swap and no scale change anywhere:
Blender world space and COLMAP world space are both right-handed metric
spaces. COLMAP requires cameras and points to agree; it does not prescribe
a world up-axis. Viewer-specific Gaussian PLY/SOG conversions belong at the
model export boundary in coordinates.py, never in this dataset pipeline.

The ONLY other convention change is per-camera and purely local:

    blender_to_opencv()  rebases a camera's own axes from Blender's
    (+X right, +Y up, −Z forward) to OpenCV's (+X right, +Y down,
    +Z forward) by RIGHT-multiplying with diag(1, −1, −1, 1).

Right-multiplication touches only the rotation columns; the translation
column — the camera's position in the export frame — is bit-identical
before and after. So this flip can never move a camera relative to the
point cloud. COLMAP images.txt stores the OpenCV world-to-camera form derived from
the same `c2w_export`.

Camera object scale is stripped in exactly one place
(`camera_pose_world()`), because the renderer ignores camera scale. If it
were left in, images.txt would carry a non-orthonormal "rotation" while
RGB-D back-projection used the scale-free one — the two would disagree.


DATASET PASS BOUNDARY
=====================
The portable ``Dataset(Default)`` always contains RGB, geometry-derived binary
masks, metric camera-Z depth, and canonical COLMAP files. Optional Blender
normal and archival render passes are owned exclusively by
``building_data.dataset_export`` and live below ``sg_metadata``. Point-cloud
generation back-projects the saved RGB-D views and never depends on a live
mesh intersection to decide whether a rendered surface exists.
"""

import bpy
import errno
import hashlib
import json
import math
import os
import shutil
import threading
import time
import uuid
from bpy.app.handlers import persistent
from mathutils import Matrix, Quaternion, Vector
from pathlib import Path

from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

# Leaf modules only. Nothing under building_data may import this module at
# import time, which is what keeps this direction of the dependency safe.
from . import edition
from . import progress
from . import theme
from . import camera_rigs
from .building_data import manifest as bd_manifest
from .building_data import paths as bd_paths
from .building_data import dataset_export as bd_export
from .building_data import pointcloud as bd_pointcloud
from .raw_export import hooks as raw_hooks
from .raw_export import stage as raw_stage

#: The stages each Building Data workflow reports, in order.
#: RGB and depth must be rendered before the point cloud can be reconstructed.
# The three phases of a full build, in the order they run. Each gets its own
# progress bar and status, so the whole workflow is visible from the start
# rather than one phase at a time.
PHASE_RENDER = "Render Camera Images"
PHASE_FILES = "Generate Dataset Files"
PHASE_POINTS = "Generate Point Cloud"

BUILD_WORKFLOW_STAGES = {
    'RENDER': (PHASE_RENDER, PHASE_FILES),
    'POINTS': (PHASE_POINTS,),
    'FULL': (PHASE_RENDER, PHASE_FILES, PHASE_POINTS),
}
BUILD_WORKFLOW_LABELS = {
    'RENDER': "Render Images",
    'POINTS': "Generate Point Cloud",
    'FULL': "Build Dataset",
}


def _sr_workflow_stages(scene, workflow, fallback):
    """The phases a workflow will announce, including the raw export."""
    stages = tuple(BUILD_WORKFLOW_STAGES.get(workflow, fallback))
    if raw_stage.should_run(scene, workflow):
        stages += (raw_stage.PHASE,)
    return stages


# ═══════════════════════════════════════════════════════════════════════
#  1. THE COORDINATE CONTRACT
#     One transform. Everything exported passes through it.
# ═══════════════════════════════════════════════════════════════════════


def camera_pose_world(cam_obj):
    """The camera's true camera-to-world matrix in Blender world space,
    with object scale stripped.

    `matrix_world` may carry object scale (including negative/mirrored
    scale). The renderer ignores it — a scaled camera renders exactly as an
    unscaled one — so the exported pose must ignore it too. Decomposing and
    rebuilding from (location, rotation) alone yields a proper rigid
    transform: an orthonormal rotation with det = +1, which is what both
    `Matrix.inverted()` → quaternion (images.txt) and ray-direction maths
    (point cloud) require.

    Returns (Matrix c2w, bool had_scale).
    """
    loc, rot, scl = cam_obj.matrix_world.decompose()
    had_scale = any(abs(s - 1.0) > 1e-4 for s in (scl.x, scl.y, scl.z))
    m = rot.to_matrix().to_4x4()
    m.translation = loc
    return m, had_scale


def export_transform_from_cameras(world_poses, recenter=False):
    """THE world→export transform. Computed once, persisted, reused.

    Datasets are exported in Blender world space, so this is the identity and
    a Splat trained from one lands exactly on the scene it came from. The
    recentering branch is retained only so a dataset manifest written by an
    older build still resolves to the matrix it was created with.

    A pure translation is used on purpose: it commutes with everything, it
    cannot introduce a rotation or scale mismatch, and applying it to a
    camera pose is provably the same operation as applying it to a point.
    """
    if not recenter or not world_poses:
        return Matrix.Identity(4)
    locs = [m.to_translation() for m in world_poses]
    centre = Vector((
        (min(v.x for v in locs) + max(v.x for v in locs)) / 2.0,
        (min(v.y for v in locs) + max(v.y for v in locs)) / 2.0,
        (min(v.z for v in locs) + max(v.z for v in locs)) / 2.0,
    ))
    return Matrix.Translation(-centre)


def apply_export_transform_to_pose(M, c2w_world):
    """Camera-to-world pose, world frame → export frame."""
    return M @ c2w_world


def apply_export_transform_to_points(M, xyz):
    """Nx3 point array, world frame → export frame. Same M, same maths."""
    import numpy as np
    A = np.array([[M[r][c] for c in range(4)] for r in range(4)],
                 dtype=np.float64)
    return (xyz @ A[:3, :3].T) + A[:3, 3]


# Camera-local axis rebasing. Right-multiply ⇒ the translation column is
# untouched ⇒ this can never move a camera relative to the point cloud.
_OPENCV_FLIP = Matrix.Diagonal(Vector((1.0, -1.0, -1.0, 1.0)))


def blender_to_opencv(c2w):
    """Blender camera axes (−Z fwd, +Y up) → OpenCV (+Z fwd, +Y down)."""
    return c2w @ _OPENCV_FLIP


def colmap_pose_from_c2w(c2w_export):
    """COLMAP images.txt stores WORLD-TO-CAMERA in OpenCV convention.
    Returns (Quaternion qw,qx,qy,qz, Vector t)."""
    w2c = blender_to_opencv(c2w_export).inverted()
    return w2c.to_quaternion(), w2c.to_translation()


def camera_centre_from_colmap(q, t):
    """Inverse of the above, for verification: C = -R^T t."""
    R = Quaternion(q).to_matrix()
    return -(R.transposed() @ Vector(t))


def matrix_to_list(m):
    return [[m[r][c] for c in range(4)] for r in range(4)]


def list_to_matrix(rows):
    return Matrix([[float(v) for v in row] for row in rows])


def matrix_to_flat_list(matrix):
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def flat_list_to_matrix(values):
    if len(values) != 16:
        raise ValueError("A 4x4 matrix needs 16 values")
    return Matrix([[float(values[row * 4 + column]) for column in range(4)]
                   for row in range(4)])


# ═══════════════════════════════════════════════════════════════════════
#  2. INTRINSICS
#     Derived from Blender's own viewplane maths (BKE_camera_params_*),
#     so the exported pinhole model matches the pixels that were rendered.
# ═══════════════════════════════════════════════════════════════════════


def effective_resolution(render):
    """The pixel size Blender will actually render at, honouring
    resolution_percentage. We never change the user's render settings, so
    intrinsics and the dataset must both use this effective size."""
    pct = getattr(render, "resolution_percentage", 100) / 100.0
    return (max(1, int(render.resolution_x * pct)),
            max(1, int(render.resolution_y * pct)))


def compute_intrinsics(cam_data, render):
    """Return (fx, fy, cx, cy, camera_angle_x) in pixels for one camera.

    Read-only: inspects lens/sensor/shift, never writes.

    Blender fits the sensor to one image axis and derives the other, and
    the choice depends on sensor_fit AND the aspect-corrected resolution:

        fit    = HORIZONTAL if pixel_aspect_x*w >= pixel_aspect_y*h  (AUTO)
        sensor = sensor_height if sensor_fit == 'VERTICAL' else sensor_width
        viewfac = w if fit is HORIZONTAL else (par_y/par_x) * h
        fx = lens * viewfac / sensor
        fy = fx * par_x / par_y
        cx = w/2 - shift_x * viewfac
        cy = h/2 + shift_y * viewfac * par_x / par_y

    The previous implementation assumed horizontal fit and a centred
    principal point, which silently produced wrong fx for portrait renders
    and ignored lens shift entirely. Both cases render fine and then fail
    to train, because the exported model no longer describes the images.
    """
    w, h = effective_resolution(render)
    par_x = getattr(render, "pixel_aspect_x", 1.0) or 1.0
    par_y = getattr(render, "pixel_aspect_y", 1.0) or 1.0
    ycor = par_y / par_x

    fit = cam_data.sensor_fit
    if fit == 'AUTO':
        resolved = 'HORIZONTAL' if (par_x * w) >= (par_y * h) else 'VERTICAL'
    else:
        resolved = fit
    # BKE_camera_sensor_size(): sensor_y only when the PROPERTY is VERTICAL.
    sensor = cam_data.sensor_height if fit == 'VERTICAL' else cam_data.sensor_width
    viewfac = w if resolved == 'HORIZONTAL' else ycor * h

    focal_mm = max(1e-6, cam_data.lens)
    fx = focal_mm * viewfac / max(1e-9, sensor)
    fy = fx / ycor
    cx = w / 2.0 - cam_data.shift_x * viewfac
    cy = h / 2.0 + cam_data.shift_y * viewfac / ycor
    camera_angle_x = 2.0 * math.atan(w / (2.0 * fx))
    return fx, fy, cx, cy, camera_angle_x


# ═══════════════════════════════════════════════════════════════════════
#  3. SETTINGS
# ═══════════════════════════════════════════════════════════════════════


def _sr_apply_global_camera_settings(cfg, camera):
    """Apply the dataset-wide camera settings to one managed camera."""
    if camera is None or camera.type != 'CAMERA':
        return
    data = camera.data
    data.type = 'PERSP'
    data.lens_unit = 'MILLIMETERS'
    data.lens = cfg.global_focal_length
    data.sensor_width = cfg.global_sensor_width
    data.sensor_height = cfg.global_sensor_height
    data.sensor_fit = cfg.global_sensor_fit
    data.shift_x = cfg.global_shift_x
    data.shift_y = cfg.global_shift_y
    # Global clipping is real camera state, not a render-time override. This
    # makes the selected camera view match the dataset at all times and also
    # means newly created/imported queued cameras inherit it immediately.
    if cfg.use_global_clipping:
        data.clip_start = cfg.global_clip_start
        data.clip_end = cfg.global_clip_end
    data.display_size = cfg.global_display_size
    data.passepartout_alpha = (cfg.global_passepartout_opacity
                               if cfg.global_use_passepartout else 0.0)


def _sr_global_camera_settings_update(self, context):
    """Keep every managed camera identical as soon as a setting changes."""
    if _sr_bulk["busy"]:
        return
    changed = False
    for item in self.camera_queue:
        _sr_apply_global_camera_settings(self, item.camera)
        if item.render_state == 'RENDERED':
            item.render_state = 'PENDING'
            changed = True
    if changed:
        _sr_mark_cameras_pending(self, [item.camera for item in self.camera_queue])
        self.render_status = "Camera settings changed - rendered cameras are pending."
    _sr_mark_scene_cache_dirty(getattr(self, "id_data", None))
    _tag_redraw_sceneray_splat(context)


def _sr_output_dir_update(self, context):
    """Refresh the selected dataset without a trainer dependency."""
    self.active_dataset_dir = ""
    self.latest_dataset_dir = ""
    base = _sr_output_base_path(self)
    if base is not None:
        latest = _sr_latest_completed_dataset(base)
        if latest is not None:
            self.active_dataset_dir = str(latest)
            self.latest_dataset_dir = str(latest)
    _sr_request_render_status_sync(self, force=True)
    _tag_redraw_sceneray_splat(context)



_SR_DATASET_IMAGE_FORMATS = {
    "JPEG": ("JPEG", ".jpg"),
    "PNG": ("PNG", ".png"),
}


def _sr_dataset_image_extension(cfg):
    """Filename suffix belonging to the user's dataset image format."""
    selected = str(getattr(cfg, "dataset_image_format", "PNG"))
    return _SR_DATASET_IMAGE_FORMATS.get(selected, ("PNG", ".png"))[1]


def _sr_dataset_master_relative_path(cfg, frame_index, output_dir=None):
    """The one rendered image for this frame.

    PNG and JPEG are written as color-only images. Geometry validity is
    carried exclusively by the same-name mask image.
    """
    extension = _sr_dataset_image_extension(cfg)
    folder = bd_paths.IMAGES_FOLDER
    if output_dir is not None:
        try:
            folder = (
                bd_paths.data_dir(output_dir)
                .relative_to(bd_paths.build_root(output_dir))
                .as_posix()
            )
        except ValueError:
            folder = bd_paths.IMAGES_FOLDER
    return f"./{folder}/frame_{int(frame_index):04d}{extension}"


def _sr_dataset_training_relative_path(cfg, frame_index, master_path=""):
    """The image consumed by training - the same file that was rendered."""
    return str(master_path or _sr_dataset_master_relative_path(cfg, frame_index))


def _sr_color_management_snapshot(scene, cfg=None):
    """Record the output transform Blender actually uses without changing it."""
    image_settings = scene.render.image_settings
    output_mode = str(image_settings.color_management)
    view = (
        image_settings.view_settings
        if output_mode == "OVERRIDE"
        else scene.view_settings
    )
    return {
        "working_space": str(bpy.data.colorspace.working_space),
        "working_space_interop_id": str(bpy.data.colorspace.working_space_interop_id),
        "display_device": str(scene.display_settings.display_device),
        "view_transform": str(view.view_transform),
        "look": str(view.look),
        "exposure": float(view.exposure),
        "gamma": float(view.gamma),
        "use_curve_mapping": bool(view.use_curve_mapping),
        "use_white_balance": bool(
            getattr(view, "use_white_balance", False)
        ),
        "white_balance_temperature": float(
            getattr(view, "white_balance_temperature", 6500.0)
        ),
        "white_balance_tint": float(
            getattr(view, "white_balance_tint", 10.0)
        ),
        "output_color_management": output_mode,
        "master_format": str(
            getattr(cfg, "dataset_image_format", "PNG")
        ),
        "training_color_space": (
            "Stored RGB bytes written by Blender; no add-on color transform"
        ),
    }


def _sr_dataset_image_format_update(self, context):
    """A format change requires new files even when old frames still exist."""
    changed = False
    for item in self.camera_queue:
        if item.camera is not None and item.render_state == "RENDERED":
            item.render_state = "PENDING"
            changed = True
    if changed:
        self.render_status = "Image format changed - rendered cameras are pending."
    _tag_redraw_sceneray_splat(context)


def _sr_set_viewport_wire_colors(use_object_colors):
    """Point every 3D View's wireframe colour at the object, or back at theme.

    Blender colours a camera's wireframe from ``Object.color`` only when the
    viewport's Wireframe Color is set to Object, so the switch that turns
    status colours on has to set it - otherwise the colours are stored and
    never seen. Turning the feature off puts it back to Theme.
    """
    wanted = 'OBJECT' if use_object_colors else 'THEME'
    window_manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                shading = getattr(space, "shading", None)
                if shading is None:
                    continue
                try:
                    shading.wireframe_color_type = wanted
                except (AttributeError, TypeError):
                    pass
            area.tag_redraw()


def _sr_apply_camera_status_colors(cfg):
    """Colour every queued camera by whether its image exists yet.

    Purple until it is rendered, blue once it is - the same two colours the
    Camera List shows, so the 3D View and the list never disagree.
    """
    if cfg is None or not getattr(cfg, "camera_status_colors", False):
        return
    from . import camera_overlay

    for item in cfg.camera_queue:
        camera = item.camera
        if camera is None:
            continue
        wanted = theme.camera_rgba(item.render_state == 'RENDERED')
        try:
            if tuple(round(c, 4) for c in camera.color) != wanted:
                camera.color = wanted
        except (AttributeError, ReferenceError, TypeError):
            continue
    # Object.color is set for the viewport modes that do honour it; the
    # overlay is what guarantees the colour is actually seen.
    camera_overlay.tag_redraw()


def _sr_camera_status_colors_update(self, context):
    from . import camera_overlay

    _sr_set_viewport_wire_colors(self.camera_status_colors)
    if self.camera_status_colors:
        _sr_apply_camera_status_colors(self)
    # The overlay is what actually makes the colours visible: Blender draws
    # cameras as extras, so Object.color alone does not recolour them.
    camera_overlay.sync(self)


class SceneRaySplatProperties(PropertyGroup):
    # There is no viewport-overlay switch any more. SplatGen draws no buttons
    # into the 3D View: every Viewer control lives in the add-on panel, where
    # Blender owns the layout and nothing competes with the viewport's own
    # gizmos, tools or navigation.
    show_import_section: BoolProperty(
        name="Viewer",
        default=True,
    )
    show_project_section: BoolProperty(
        name="Project",
        default=False,
    )
    show_camera_section: BoolProperty(
        name="Camera Placement",
        default=False,
    )
    show_dataset_section: BoolProperty(
        name="Building Dataset",
        default=False,
    )
    show_render_section: BoolProperty(
        name="Rendering Images",
        default=False,
    )
    show_pointcloud_section: BoolProperty(
        name="Point Cloud",
        default=False,
    )
    show_camera_rigs: BoolProperty(
        name="Camera Creation & Rigs",
        default=False,
    )
    show_camera_list: BoolProperty(
        name="Camera List",
        default=False,
    )
    show_prepare_queue: BoolProperty(name="Camera queue", default=True)
    show_camera_list_section: BoolProperty(
        name="Camera List",
        default=False,
    )
    show_global_camera_settings: BoolProperty(
        name="Global Camera Settings",
        default=False,
    )
    # Global Camera Settings is a stack of collapsed groups. Each one is a
    # handful of properties that are set once and rarely revisited, so none of
    # them earns permanent vertical space in the panel.
    camera_status_colors: BoolProperty(
        name="Camera Status Colors",
        description=(
            "Colour queued cameras in the 3D View by their state - orange "
            "until rendered, blue once rendered. "
            "Sets each viewport's Wireframe Color to Object, which is what "
            "makes object colours visible"
        ),
        default=False,
        update=_sr_camera_status_colors_update,
    )
    show_camera_placement_section: BoolProperty(
        name="Camera Placement",
        default=False,
    )
    show_clipping_settings: BoolProperty(
        name="Clipping Settings",
        default=False,
    )
    show_lens_settings: BoolProperty(
        name="Lens & Sensor",
        default=False,
    )
    show_display_settings: BoolProperty(
        name="Viewport Display",
        default=False,
    )
    output_dir: StringProperty(
        name="Output Dir",
        description="Parent folder for generated datasets. Leave empty to create SplatGen_<project> beside the saved .blend file. Save the file before building",
        default="",
        subtype='DIR_PATH',
        update=_sr_output_dir_update,
    )
    dataset_image_format: EnumProperty(
        name="Image Format",
        description="File format used for the rendered training images",
        items=[
            (
                "PNG",
                "PNG",
                "Lossless 8-bit RGB color; geometry validity is stored in the mask image",
            ),
            ("JPEG", "JPEG", "Small 8-bit files with adjustable quality"),
        ],
        default="PNG",
        update=_sr_dataset_image_format_update,
    )
    jpeg_quality: IntProperty(
        name="JPEG Quality",
        description=(
            "Quality of the smaller JPEG training images. 90 keeps strong "
            "visual detail while substantially reducing dataset size"
        ),
        default=90,
        min=70,
        max=100,
    )
    png_compression: IntProperty(
        name="PNG Compression",
        description="Shared compression for PNG training images and PNG masks; higher is smaller but slower, with no loss of image quality",
        default=15,
        min=0,
        max=100,
        subtype="PERCENTAGE",
    )
    dataset_mask_format: EnumProperty(
        name="Mask Format",
        description=(
            "File type for the one-channel validity masks. PNG is lossless "
            "and recommended; JPEG is smaller but can soften mask edges"
        ),
        items=[
            ("PNG", "PNG", "Recommended lossless binary validity masks"),
            ("JPEG", "JPEG", "Smaller masks with possible edge compression"),
        ],
        default="PNG",
        update=_sr_dataset_image_format_update,
    )
    export_ground_truth_indices: BoolProperty(
        name="Export Object and Material Indices",
        description=(
            "Script-accessible future pass option. Requires a render engine "
            "that exposes Object Index and Material Index, such as Cycles"
        ),
        default=False,
        options={"HIDDEN"},
        update=_sr_dataset_image_format_update,
    )
    # ---- live UI state (not saved into the .blend) ----------------------
    build_workflow: EnumProperty(
        name="Workflow",
        description="Which Building Data operation is currently running",
        items=[
            ('NONE', "None", "Nothing is running"),
            ('RENDER', "Render Images", "Images and camera data only"),
            ('POINTS', "Generate Point Cloud", "Point cloud only"),
            ('FULL', "Build Dataset", "Images, camera data and point cloud"),
        ],
        default='NONE',
        options={'SKIP_SAVE'},
    )
    is_rendering: BoolProperty(
        name="Rendering", default=False, options={'SKIP_SAVE'},
        description="A SplatGen dataset render is currently in progress",
    )
    render_progress: IntProperty(name="Render Progress", default=0,
                                 options={'SKIP_SAVE'})
    render_total: IntProperty(name="Render Total", default=0,
                              options={'SKIP_SAVE'})
    render_progress_fac: FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0,
        subtype='FACTOR', options={'SKIP_SAVE'},
    )
    render_eta: StringProperty(name="ETA", default="", options={'SKIP_SAVE'})
    render_status: StringProperty(name="Render Status", default="",
                                  options={'SKIP_SAVE'})
    build_started_at: FloatProperty(
        name="Dataset Build Start",
        default=0.0,
        options={'SKIP_SAVE', 'HIDDEN'},
    )
    # ---- dataset-wide camera settings -----------------------------------
    global_focal_length: FloatProperty(
        name="Focal Length", default=35.0, min=1.0, max=5000.0,
        description="Focal length shared by every managed camera",
        update=_sr_global_camera_settings_update)
    global_display_size: FloatProperty(
        name="Camera Display Size", default=0.5, min=0.01, max=1000.0,
        description="Viewport display size shared by every managed camera",
        update=_sr_global_camera_settings_update)
    use_global_clipping: BoolProperty(
        name="Use Global Camera Clipping",
        description=(
            "Write the clipping below directly to every managed camera so "
            "camera preview and dataset rendering always match"
        ),
        default=False,
        update=_sr_global_camera_settings_update)
    global_clip_start: FloatProperty(
        name="Clip Start", default=0.1, min=0.0001, max=100000.0,
        subtype='DISTANCE', unit='LENGTH',
        description="Near clipping written immediately to every managed camera",
        update=_sr_global_camera_settings_update)
    global_clip_end: FloatProperty(
        name="Clip End", default=1000.0, min=0.01, max=10000000.0,
        subtype='DISTANCE', unit='LENGTH',
        description="Far clipping written immediately to every managed camera",
        update=_sr_global_camera_settings_update)
    global_sensor_width: FloatProperty(
        name="Sensor Width", default=36.0, min=1.0, max=1000.0,
        description="Sensor width shared by every managed camera",
        update=_sr_global_camera_settings_update)
    global_sensor_height: FloatProperty(
        name="Sensor Height", default=24.0, min=1.0, max=1000.0,
        description="Sensor height shared by every managed camera",
        update=_sr_global_camera_settings_update)
    global_sensor_fit: EnumProperty(
        name="Sensor Fit", items=[
            ('AUTO', "Auto", "Fit the sensor automatically"),
            ('HORIZONTAL', "Horizontal", "Fit horizontally"),
            ('VERTICAL', "Vertical", "Fit vertically"),
        ], default='AUTO', update=_sr_global_camera_settings_update)
    global_shift_x: FloatProperty(
        name="Shift X", default=0.0, min=-10.0, max=10.0,
        update=_sr_global_camera_settings_update)
    global_shift_y: FloatProperty(
        name="Shift Y", default=0.0, min=-10.0, max=10.0,
        update=_sr_global_camera_settings_update)
    global_use_passepartout: BoolProperty(
        name="Passepartout", default=True,
        description="Show the global camera passepartout while reviewing",
        update=_sr_global_camera_settings_update)
    global_passepartout_opacity: FloatProperty(
        name="Passepartout Opacity", default=1.0, min=0.0, max=1.0,
        subtype='FACTOR',
        description="Darkening opacity outside every managed camera frame",
        update=_sr_global_camera_settings_update)


# ═══════════════════════════════════════════════════════════════════════
#  4. MANIFESTS
#     Two small JSON handoffs. The dataset manifest is the authoritative
#     record of the export frame — step 3 reads M from here, never guesses.
# ═══════════════════════════════════════════════════════════════════════

_WORK_DIR = "_SceneRaySplat_Temporary"
_RENDER_MANIFEST = "render_manifest.json"
_DATASET_MANIFEST = "dataset_manifest.json"
_CAMERA_MAP_PREFIX = "# Blender camera: "
_COLOR_MAP_PREFIX = "# SplatGen color management: "
#: Written by releases before the rename; datasets built then must still read.
_LEGACY_COLOR_MAP_PREFIXES = ("# SplatRay color management: ",)
_MANAGED_CAMERA_PREFIX = "SceneRaySplatCam"
RENDER_MANIFEST_VERSION = 3
DATASET_MANIFEST_VERSION = 3


class SceneRaySplatStorageError(RuntimeError):
    """A fatal output-storage failure that must stop the current batch."""


def _sr_output_base_path(cfg):
    """Project root for every dataset version.

    Resolves through _sr_project_root_path so a dataset version can never be
    used as the base, which would nest the next version inside it.
    """
    return _sr_project_root_path(cfg)


def _sr_effective_output_path(cfg):
    """Current version folder, falling back to the selected dataset root."""
    base = _sr_output_base_path(cfg)
    active_raw = (getattr(cfg, "active_dataset_dir", "") or "").strip()
    if base is None or not active_raw:
        return base
    active = Path(bpy.path.abspath(active_raw))
    try:
        if active == base or base in active.parents:
            return active
    except (OSError, RuntimeError):
        pass
    return base


# The on-disk shape lives in building_data.paths. These thin names are kept
# because they read naturally at the call sites throughout this module.
_sr_is_timestamp_folder = bd_paths.is_timestamp_folder
_sr_is_complete_dataset = bd_paths.is_complete_build
_sr_has_reusable_content = bd_paths.has_reusable_content
_sr_is_dataset_version = bd_paths.has_reusable_content
_sr_project_root_path = bd_paths.project_root
_sr_latest_completed_dataset = bd_paths.latest_build
_sr_new_version_folder = bd_paths.new_build_folder


def _sr_data_dir(output_dir):
    """The RGB-only images directory inside a build folder."""
    return bd_paths.data_dir(output_dir)


def _sr_dataset_file(output_dir, name):
    return bd_paths.dataset_file(output_dir, name)


def _sr_storage_error_message(exc, output_dir):
    if (getattr(exc, "errno", None) == errno.ENOSPC
            or getattr(exc, "winerror", None) == 112):
        return (
            f"The drive containing '{output_dir}' is full. Free some space, "
            "then run Render Images again; unfinished cameras remain Pending.")
    return (
        f"Could not update render tracking data in '{output_dir}': {exc}. "
        "Fix the output folder, then run Render Images again.")


def _sr_check_render_disk_space(output_dir, render, resolution):
    """Keep enough headroom for one render and its tracking data."""
    try:
        free_bytes = shutil.disk_usage(output_dir).free
    except OSError:
        return
    pixel_bytes = {
        'OPEN_EXR_MULTILAYER': 64,
        'OPEN_EXR': 32,
        'TIFF': 16,
    }.get(render.image_settings.file_format, 8)
    width, height = resolution
    required = max(64 * 1024 * 1024, width * height * pixel_bytes * 2)
    if free_bytes < required:
        free_mib = free_bytes / (1024 * 1024)
        required_mib = required / (1024 * 1024)
        raise SceneRaySplatStorageError(
            f"Not enough free space in '{output_dir}' to safely render the "
            f"next image ({free_mib:.1f} MiB free; about "
            f"{required_mib:.1f} MiB required). Free some space, then run "
            "Render Images again; unfinished cameras remain Pending.")


def _work_dir_path(output_dir):
    return Path(output_dir) / _WORK_DIR


def _render_manifest_path(output_dir):
    return _work_dir_path(output_dir) / _RENDER_MANIFEST


def write_render_manifest(
    output_dir,
    cameras_kept,
    resolution,
    color_management=None,
):
    """Persist every managed camera, including cameras still pending.

    ``pending`` is retained for compatibility with version-1 manifests while
    the explicit status string makes the file readable outside Blender.
    """
    entries = []
    for source in cameras_kept:
        entry = dict(source)
        if "pending" in entry:
            pending = bool(entry["pending"])
        else:
            pending = str(entry.get("status", "RENDERED")).upper() != "RENDERED"
        entry["pending"] = pending
        entry["status"] = "PENDING" if pending else "RENDERED"
        entries.append(entry)
    path = _render_manifest_path(output_dir)
    if color_management is None and path.is_file():
        try:
            with open(path, encoding="utf-8") as handle:
                color_management = json.load(handle).get("color_management")
        except (OSError, TypeError, ValueError):
            color_management = None
    data = {
        "version": RENDER_MANIFEST_VERSION,
        "resolution": list(resolution),
        "cameras": entries,
    }
    if color_management:
        data["color_management"] = dict(color_management)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    except Exception:
        # Keep the previous complete manifest authoritative and remove any
        # truncated temporary file left by an interrupted or full drive.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_render_manifest(output_dir):
    p = _render_manifest_path(output_dir)
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_completed_camera_data(output_dir):
    """Recover camera/image ownership after temporary manifests are removed."""
    root = Path(output_dir)
    images_txt = _sr_dataset_file(root, "images.txt")
    cameras_txt = _sr_dataset_file(root, "cameras.txt")
    if not images_txt.is_file():
        return None
    entries = []
    color_management = None
    try:
        with open(images_txt, encoding="utf-8") as handle:
            for line in handle:
                prefix = next(
                    (candidate
                     for candidate in (_COLOR_MAP_PREFIX,)
                     + _LEGACY_COLOR_MAP_PREFIXES
                     if line.startswith(candidate)),
                    None,
                )
                if prefix is not None:
                    try:
                        color_management = json.loads(line[len(prefix):])
                    except (TypeError, ValueError):
                        color_management = None
                    continue
                if not line.startswith(_CAMERA_MAP_PREFIX):
                    continue
                payload = json.loads(line[len(_CAMERA_MAP_PREFIX):])
                name = str(payload.get("name", ""))
                file_path = str(payload.get("file_path", ""))
                if name and file_path:
                    entry = {
                        "name": name, "file_path": file_path,
                        "training_file_path": str(
                            payload.get("training_file_path", file_path)
                        ),
                        "frame_index": int(payload.get(
                            "frame_index", len(entries))),
                        "pending": False, "status": "RENDERED",
                    }
                    signature = payload.get("signature")
                    if isinstance(signature, dict):
                        entry["signature"] = signature
                    entries.append(entry)
        resolution = (1, 1)
        if cameras_txt.is_file():
            with open(cameras_txt, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip() and not line.startswith("#"):
                        fields = line.split()
                        resolution = (int(fields[2]), int(fields[3]))
                        break
    except (OSError, ValueError, TypeError, IndexError):
        return None
    if not entries:
        return None
    data = {
        "version": RENDER_MANIFEST_VERSION,
        "resolution": list(resolution), "cameras": entries,
    }
    if color_management:
        data["color_management"] = color_management
    return data


def _sr_finalize_dataset_export(cfg, output_dir):
    """Validate the canonical dataset once every required source exists."""
    if not bd_export.enabled(cfg):
        return None
    data = read_render_manifest(output_dir) or read_completed_camera_data(output_dir)
    if not data:
        return None
    return bd_export.finalize_dataset(
        output_dir,
        data.get("cameras", ()),
        tuple(data.get("resolution", (1, 1))),
        data.get("color_management")
        or _sr_color_management_snapshot(cfg.id_data, cfg),
        cfg,
    )


_render_status_sync = {
    "pending": set(), "signatures": {}, "timer": False,
    "last_checks": {},
}
_RENDER_STATUS_CHECK_INTERVAL = 0.75


def _sr_camera_settings_fingerprint(cfg):
    """One value that changes whenever any queued camera's render inputs do.

    Folder timestamps cannot see a camera being moved or its lens changed, so
    without this the status refresh would never run for exactly the edits that
    make an image outdated.
    """
    parts = []
    for item in getattr(cfg, "camera_queue", ()):
        camera = item.camera
        if camera is None or camera.type != 'CAMERA':
            continue
        try:
            signature = bd_manifest.camera_signature(camera, cfg)
        except (AttributeError, ReferenceError, RuntimeError, TypeError):
            continue
        parts.append(f"{camera.name}:{signature}")
    if not parts:
        return 0
    return hash(tuple(parts))


def _sr_render_status_signature(cfg):
    """Cheap signature used to detect image, manifest or camera changes."""
    output_dir = _sr_effective_output_path(cfg)
    cameras = _sr_camera_settings_fingerprint(cfg)
    if output_dir is None:
        return ("", None, None, None, cameras)
    try:
        manifest = _render_manifest_path(output_dir)
        camera_data = _sr_dataset_file(output_dir, "images.txt")
        images = _sr_data_dir(output_dir)
        manifest_stamp = (manifest.stat().st_mtime_ns, manifest.stat().st_size) if manifest.is_file() else None
        camera_data_stamp = ((camera_data.stat().st_mtime_ns,
                              camera_data.stat().st_size)
                             if camera_data.is_file() else None)
        images_stamp = images.stat().st_mtime_ns if images.is_dir() else None
        return (str(output_dir), manifest_stamp, camera_data_stamp,
                images_stamp, cameras)
    except OSError:
        return (str(output_dir), None, None, None, cameras)


def _sr_continuing_existing_dataset(cfg, build=None):
    """Whether this is a resume of work that was interrupted part-way.

    Camera movement only matters while finishing a build that never finished:
    there, the images already on disk have to stay consistent with the ones
    still to come, so a camera that has ended up somewhere else needs
    re-rendering.

    A build that *did* complete is a finished thing. Moving a camera afterwards
    does not retro-actively damage it - pressing Build Dataset again makes a
    new dataset from scratch rather than repairing that one - so its cameras
    stay Rendered.

    Judged from the folder itself: a build is incomplete when it holds images
    but not the camera data that finishes it. That evidence cannot fall out of
    step with reality the way a stored flag can.
    """
    if cfg is None or cfg.is_rendering:
        return False
    if build is None:
        try:
            build = _sr_effective_output_path(cfg)
        except (OSError, TypeError, ValueError):
            return False
    if build is None:
        return False
    try:
        build = Path(build)
        if not build.is_dir():
            return False
        # Finished: nothing to keep consistent.
        if bd_paths.has_stage1_output(build):
            return False
        # Interrupted: some images exist, the dataset does not.
        return any(bd_paths.data_dir(build).glob("frame_*"))
    except (OSError, TypeError, ValueError):
        return False


def _sr_sync_render_states(scene, cfg, output_dir=None):
    """Make queue state reflect the manifest *and* its image files.

    The manifest tells us which filename belongs to a camera; the file itself
    is the final authority. This also recognizes a copied output folder when
    its manifest and images travel together.
    """
    if cfg.is_rendering:
        return False
    try:
        root = (Path(output_dir) if output_dir is not None
                else _sr_effective_output_path(cfg))
        if root is None:
            return False
    except (TypeError, ValueError):
        return False
    manifest = (read_render_manifest(root)
                or read_completed_camera_data(root) or {})
    entries = {entry.get("name"): entry
               for entry in manifest.get("cameras", [])
               if isinstance(entry, dict) and entry.get("name")}
    saved_color = manifest.get("color_management")
    color_matches = (
        not saved_color
        or saved_color == _sr_color_management_snapshot(scene, cfg)
    )
    check_movement = _sr_continuing_existing_dataset(cfg, root)
    changed = False
    manifest_changed = False
    for item in cfg.camera_queue:
        camera = item.camera
        entry = entries.get(camera.name) if camera is not None else None
        rel_path = entry.get("file_path", "") if entry else ""
        image_path = root / rel_path.lstrip("./\\") if rel_path else None
        if (
            entry
            and str(getattr(cfg, "dataset_image_format", "PNG")) == "OPEN_EXR"
            and not entry.get("training_file_path")
        ):
            training_rel_path = _sr_dataset_training_relative_path(
                cfg,
                int(entry.get("frame_index", 0)),
                rel_path,
            )
        else:
            training_rel_path = (
                str(entry.get("training_file_path", rel_path)) if entry else ""
            )
        training_path = (
            root / training_rel_path.lstrip("./\\")
            if training_rel_path
            else None
        )
        format_matches = (
            bool(rel_path)
            and Path(rel_path).suffix.lower()
            == _sr_dataset_image_extension(cfg)
        )
        if entry is None:
            entry_pending = True
        elif "pending" in entry:
            entry_pending = bool(entry["pending"])
        elif "status" in entry:
            entry_pending = str(entry["status"]).upper() != "RENDERED"
        else:
            # Version-1 entries represented successful renders only.
            entry_pending = False
        image_exists = image_path is not None and image_path.is_file()
        training_image_exists = (
            training_path is not None and training_path.is_file()
        )
        # Any tracked camera property that moved since the image was rendered
        # makes that image outdated - but only while continuing existing work.
        # During a first build there is nothing to be out of date with, and
        # flipping cameras back to Pending as the artist nudges them into
        # place is noise rather than information.
        settings_match = True
        if check_movement:
            settings_match = entry is not None and camera is not None and (
                bd_manifest.matches(
                    entry, bd_manifest.camera_signature(camera, cfg))
            )
        state = ("RENDERED" if entry is not None and not entry_pending
                 and image_exists and training_image_exists
                 and format_matches and color_matches and settings_match
                 else "PENDING")
        if item.render_state != state:
            item.render_state = state
            changed = True
        if entry is not None:
            should_pending = state == "PENDING"
            should_status = "PENDING" if should_pending else "RENDERED"
            if (bool(entry.get("pending", False)) != should_pending
                    or entry.get("status") != should_status):
                entry["pending"] = should_pending
                entry["status"] = should_status
                manifest_changed = True
    if manifest_changed:
        write_render_manifest(
            root, manifest.get("cameras", []),
            tuple(manifest.get("resolution", (1, 1))),
            color_management=manifest.get("color_management"),
        )
    # The 3D View has to agree with the list, so the colours are refreshed
    # wherever a render state is decided - not only where the list is drawn.
    if changed:
        _sr_apply_camera_status_colors(cfg)
    return changed or manifest_changed


def _sr_run_pending_render_status_sync():
    _render_status_sync["timer"] = False
    pending = set(_render_status_sync["pending"])
    _render_status_sync["pending"].clear()
    for scene in bpy.data.scenes:
        cfg = getattr(scene, "SCENERAY_SPLAT", None)
        if cfg is None:
            continue
        try:
            pointer = cfg.as_pointer()
        except ReferenceError:
            continue
        if pointer not in pending:
            continue
        try:
            _sr_sync_render_states(scene, cfg)
        except OSError as exc:
            print(f"[sceneray_splat] render-status refresh skipped: {exc}")
        _render_status_sync["signatures"][pointer] = _sr_render_status_signature(cfg)
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'OUTLINER':
                area.tag_redraw()
    return None


def _sr_request_render_status_sync(cfg, force=False):
    """Schedule safe status refreshes from property/UI draw contexts."""
    try:
        pointer = cfg.as_pointer()
    except ReferenceError:
        return
    effective = _sr_effective_output_path(cfg)
    raw_dir = str(effective) if effective is not None else ""
    now = time.monotonic()
    last = _render_status_sync["last_checks"].get(pointer)
    if (not force and last is not None and last[0] == raw_dir
            and now - last[1] < _RENDER_STATUS_CHECK_INTERVAL):
        return
    _render_status_sync["last_checks"][pointer] = (raw_dir, now)
    signature = _sr_render_status_signature(cfg)
    if not force and _render_status_sync["signatures"].get(pointer) == signature:
        return
    _render_status_sync["pending"].add(pointer)
    if not _render_status_sync["timer"]:
        _render_status_sync["timer"] = True
        bpy.app.timers.register(_sr_run_pending_render_status_sync, first_interval=0.05)

def _dataset_manifest_path(output_dir):
    return _work_dir_path(output_dir) / _DATASET_MANIFEST


def write_dataset_manifest(
    output_dir,
    frames,
    resolution,
    export_matrix,
    color_management=None,
):
    """Persist the export frame itself, plus every frame in BOTH frames.

    `world_matrix` (Blender world) is what step 3 casts rays with;
    `transform_matrix` (export frame) is what was actually written to
    images.txt comments and temporary manifests. Storing both makes the relationship
    auditable instead of implicit.
    """
    data = {
        "version": DATASET_MANIFEST_VERSION,
        "resolution": list(resolution),
        "export_from_world": matrix_to_list(export_matrix),
        "frames": frames,
    }
    if color_management:
        data["color_management"] = dict(color_management)
    path = _dataset_manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def read_dataset_manifest(output_dir):
    p = _dataset_manifest_path(output_dir)
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if data.get("version") != DATASET_MANIFEST_VERSION:
        return None
    return data


def _clean_obsolete_dataset_layout(output_dir):
    """Remove obsolete generated layouts without touching canonical output."""
    root = Path(output_dir)
    stale = ["transforms.json", "sparse_pc.ply", "README_LOADING.txt"]
    for name in stale:
        try:
            (root / name).unlink(missing_ok=True)
        except OSError:
            pass
    standard_root = bd_paths.standard_dataset_dir(root)
    # Canonical COLMAP files live only in Dataset(Default) for new builds.
    for name in bd_paths.DATASET_FILES:
        try:
            (_sr_data_dir(root) / name).unlink(missing_ok=True)
        except OSError:
            pass
        if standard_root != root:
            try:
                (root / name).unlink(missing_ok=True)
            except OSError:
                pass
    for name in ("camera.txt", "point.txt", "images_point.txt"):
        try:
            (root / name).unlink(missing_ok=True)
        except OSError:
            pass
    for generated in (root / "metadata", root / "sgdata"):
        if generated.is_dir():
            try:
                shutil.rmtree(generated)
            except OSError:
                pass
    try:
        (_sr_data_dir(root) / "sparse_pc.ply").unlink(missing_ok=True)
    except OSError:
        pass
    sparse = root / "sparse"
    if sparse.is_dir():
        try:
            shutil.rmtree(sparse)
        except OSError:
            pass
    for pattern in ("*_render_manifest.json", "*_dataset.json"):
        for path in root.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def stale_blender_temp_dirs(minimum_age_days=2):
    """Blender's own abandoned per-session temp folders.

    Blender makes one ``blender_XXXXXX`` folder per launch under the system
    temp directory and removes it on a clean exit; a crash or a force-quit
    leaves it behind. They are not this add-on's - SplatGen writes nothing to
    the system temp directory - but they are what fills the drive, so the
    add-on offers to clear them.

    The folder belonging to this Blender session, and anything modified
    recently enough that another Blender may still be using it, are never
    listed.
    """
    import tempfile

    root = Path(tempfile.gettempdir())
    mine = Path(bpy.app.tempdir).resolve() if bpy.app.tempdir else None
    cutoff = time.time() - max(0.0, float(minimum_age_days)) * 86400.0
    found = []
    try:
        children = list(root.glob("blender_*"))
    except OSError:
        return found
    for path in children:
        try:
            if not path.is_dir() or path.resolve() == mine:
                continue
            if path.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        size = 0
        try:
            for child in path.rglob("*"):
                if child.is_file():
                    size += child.stat().st_size
        except OSError:
            pass
        found.append((path, size))
    return found


class SCENERAY_SPLAT_OT_clean_temp(bpy.types.Operator):
    """Delete Blender's abandoned per-session temp folders, freeing disk space

    These are created by Blender itself, not by SplatGen, and are left behind
    when Blender does not exit cleanly. The current session's folder is never
    touched"""

    bl_idname = "sceneray_splat.clean_blender_temp"
    bl_label = "Clean Up Old Blender Temp Folders"
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        self._found = stale_blender_temp_dirs()
        if not self._found:
            self.report({'INFO'}, "No abandoned Blender temp folders found.")
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, context):
        layout = self.layout
        found = getattr(self, "_found", ())
        total = sum(size for _path, size in found)
        layout.label(
            text=f"Delete {len(found)} abandoned folder(s), "
                 f"{total / 1048576:.0f} MB?",
            icon='TRASH',
        )
        theme.muted(layout, "Remove abandoned Blender session folders. The current session stays untouched.")
        for path, size in sorted(found, key=lambda e: -e[1])[:5]:
            layout.label(text=f"    {path.name}  {size / 1048576:.0f} MB")
        if len(found) > 5:
            layout.label(text=f"    …and {len(found) - 5} more")

    def execute(self, context):
        found = getattr(self, "_found", None)
        if found is None:
            found = stale_blender_temp_dirs()
        removed, freed, failed = 0, 0, 0
        for path, size in found:
            try:
                shutil.rmtree(path)
                removed += 1
                freed += size
            except OSError:
                failed += 1
        message = f"Removed {removed} folder(s), freed {freed / 1048576:.0f} MB."
        if failed:
            message += f" {failed} still in use and were kept."
        self.report({'INFO'}, message)
        return {'FINISHED'}


def _cleanup_work_directory(output_dir):
    work = _work_dir_path(output_dir)
    if work.is_dir():
        try:
            shutil.rmtree(work)
        except OSError as exc:
            print(f"[sceneray_splat] could not remove temporary folder: {exc}")


# ═══════════════════════════════════════════════════════════════════════
#  5. DATASET WRITERS
#     Every writer below is handed frames whose `transform_matrix` is
#     ALREADY in the export frame. No writer applies a transform of its
#     own — that is what keeps the four files in one coordinate system.
# ═══════════════════════════════════════════════════════════════════════


def write_colmap_cameras_images(
    output_dir,
    frames,
    resolution,
    color_management=None,
):
    """Write the two COLMAP camera files directly in the portable dataset root.

    Model is PINHOLE (fx, fy, cx, cy), not SIMPLE_PINHOLE. SIMPLE_PINHOLE
    has a single focal length, which silently discarded fy and the true
    principal point — wrong for vertical sensor fit, non-square pixels or
    any lens shift. PINHOLE is broadly supported by compatible tools.
    """
    w, h = resolution
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    cameras_path = root / "cameras.txt"
    with open(cameras_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(frames)}\n")
        for i, fr in enumerate(frames, start=1):
            f.write(f"{i} PINHOLE {w} {h} "
                    f"{fr['fl_x']:.10g} {fr['fl_y']:.10g} "
                    f"{fr['cx']:.10g} {fr['cy']:.10g}\n")

    images_path = root / "images.txt"
    with open(images_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}\n")
        if color_management:
            f.write(
                _COLOR_MAP_PREFIX
                + json.dumps(color_management, separators=(",", ":"))
                + "\n"
            )
        for i, fr in enumerate(frames, start=1):
            mapping = {
                "name": fr["camera_name"],
                "file_path": fr["file_path"],
                "training_file_path": fr.get(
                    "training_file_path",
                    fr["file_path"],
                ),
                "frame_index": int(fr.get("frame_index", i - 1)),
            }
            # The temporary work folder is deleted once a build completes, so
            # images.txt is the only lasting record of what each image was
            # rendered with. Without it a finished build would read as
            # entirely outdated the next time Build was pressed.
            signature = fr.get("signature")
            if signature:
                mapping["signature"] = signature
            f.write(_CAMERA_MAP_PREFIX
                    + json.dumps(mapping, separators=(",", ":")) + "\n")
            q, t = colmap_pose_from_c2w(list_to_matrix(fr["transform_matrix"]))
            name = os.path.basename(
                fr.get("training_file_path", fr["file_path"])
            )
            f.write(f"{i} {q.w:.10g} {q.x:.10g} {q.y:.10g} {q.z:.10g} "
                    f"{t.x:.10g} {t.y:.10g} {t.z:.10g} {i} {name}\n")
            f.write("\n")     # empty POINTS2D line — required by the format
    return cameras_path, images_path


def read_colmap_points3D(path, max_points=None):
    """Read XYZ/RGB from the standard portable-root COLMAP text file."""
    import numpy as np
    data = np.loadtxt(path, comments="#", usecols=(1, 2, 3, 4, 5, 6),
                      max_rows=max_points)
    data = np.atleast_2d(data)
    if data.size == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8))
    xyz = data[:, :3].astype(np.float32)
    rgb = np.clip(data[:, 3:6], 0, 255).astype(np.uint8)
    return xyz, rgb


def read_colmap_images(sparse_dir):
    """images.txt → [{name, camera_id, q(w,x,y,z), t(x,y,z)}, ...]"""
    out = []
    path = Path(sparse_dir) / "images.txt"
    if not path.is_file():
        return out
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if len(p) >= 10:
            out.append({
                "name": p[9],
                "camera_id": int(p[8]),
                "q": (float(p[1]), float(p[2]), float(p[3]), float(p[4])),
                "t": (float(p[5]), float(p[6]), float(p[7])),
            })
            i += 2          # skip the POINTS2D line
        else:
            i += 1
    return out


def camera_centres_from_images_txt(sparse_dir):
    """{image_name: Vector C}, where C = -R^T t is the camera position in
    the export frame. Convention-independent: the OpenCV flip is a right-
    multiplication and never touches the translation."""
    return {os.path.basename(im["name"]): camera_centre_from_colmap(im["q"], im["t"])
            for im in read_colmap_images(sparse_dir)}


# ═══════════════════════════════════════════════════════════════════════
#  7. POINT CLOUD — RGB-D back-projection of rendered views
# ═══════════════════════════════════════════════════════════════════════


def _read_image_rgb(path):
    """Supported dataset image → top-down HxWx3 bytes for sampling."""
    import numpy as np
    img = bpy.data.images.load(str(path), check_existing=False)
    try:
        try:
            img.colorspace_settings.name = 'Non-Color'
        except Exception:
            from . import diagnostics as _diag
            _diag.swallowed("sceneray_splat.py")
        w, h = img.size
        buf = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(buf)
        rgba = buf.reshape(h, w, 4)[::-1]      # Blender stores bottom-up
        return (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    finally:
        bpy.data.images.remove(img)


def _sr_socket_value(node, names, fallback):
    """Read a constant shader socket and flag linked/procedural overrides."""
    if node is None:
        return fallback, False
    for name in names:
        socket = node.inputs.get(name)
        if socket is None:
            continue
        try:
            value = socket.default_value
            if hasattr(value, "__len__") and not isinstance(value, str):
                value = tuple(float(component) for component in value)
            else:
                value = float(value)
            return value, bool(socket.is_linked)
        except (AttributeError, TypeError, ValueError):
            continue
    return fallback, False


def _sr_principled_node(material):
    """Find the Principled node connected directly to the active output."""
    tree = getattr(material, "node_tree", None)
    if tree is None:
        return None
    for output in tree.nodes:
        if (output.bl_idname != "ShaderNodeOutputMaterial"
                or not getattr(output, "is_active_output", True)):
            continue
        surface = output.inputs.get("Surface")
        if surface is not None and surface.is_linked:
            source = surface.links[0].from_node
            if source.bl_idname == "ShaderNodeBsdfPrincipled":
                return source
    return None


def _sr_material_guidance(material):
    """Small, auditable material hints; linked inputs are marked invalid."""
    if material is None:
        return {
            "name": "<No Material>", "base_color": (0.8, 0.8, 0.8),
            "roughness": 0.5, "metallic": 0.0, "transmission": 0.0,
            "alpha": 1.0, "valid_mask": 0, "linked_inputs": (),
            "library": "",
        }
    diffuse = tuple(float(value) for value in material.diffuse_color)
    node = _sr_principled_node(material)
    base, base_linked = _sr_socket_value(
        node, ("Base Color",), diffuse[:3] + (diffuse[3],)
    )
    roughness, roughness_linked = _sr_socket_value(
        node, ("Roughness",), float(getattr(material, "roughness", 0.5))
    )
    metallic, metallic_linked = _sr_socket_value(
        node, ("Metallic",), float(getattr(material, "metallic", 0.0))
    )
    transmission, transmission_linked = _sr_socket_value(
        node, ("Transmission Weight", "Transmission"), 0.0
    )
    alpha, alpha_linked = _sr_socket_value(
        node, ("Alpha",), diffuse[3] if len(diffuse) > 3 else 1.0
    )

    def scalar(value, fallback):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    base = tuple(base) if hasattr(base, "__len__") else diffuse
    linked = [
        name for name, active in (
            ("base_color", base_linked), ("roughness", roughness_linked),
            ("metallic", metallic_linked),
            ("transmission", transmission_linked), ("alpha", alpha_linked),
        ) if active
    ]
    validity = (
        (0 if base_linked else 1) | (0 if roughness_linked else 2)
        | (0 if metallic_linked else 4) | (0 if transmission_linked else 8)
        | (0 if alpha_linked else 16)
    ) if node is not None else 0
    return {
        "name": material.name,
        "base_color": tuple(max(0.0, min(1.0, value)) for value in base[:3]),
        "roughness": max(0.0, min(1.0, scalar(roughness, 0.5))),
        "metallic": max(0.0, min(1.0, scalar(metallic, 0.0))),
        "transmission": max(0.0, min(1.0, scalar(transmission, 0.0))),
        "alpha": max(0.0, min(1.0, scalar(alpha, 1.0))),
        "valid_mask": validity,
        "linked_inputs": tuple(linked),
        "library": (
            material.library.filepath
            if getattr(material, "library", None) is not None else ""
        ),
    }


def _sr_scene_guidance_metadata(scene):
    """Describe the Blender source without copying or mutating scene data."""
    lights = []
    for obj in scene.objects:
        if obj.type != "LIGHT":
            continue
        data = obj.data
        lights.append({
            "name": obj.name, "type": data.type,
            "matrix_world": matrix_to_list(obj.matrix_world),
            "color": [float(value) for value in data.color],
            "energy": float(data.energy),
        })
    world = scene.world
    return {
        "source_blend": bpy.data.filepath or "",
        "source_scene": scene.name,
        "blender_version": bpy.app.version_string,
        "render_engine": scene.render.engine,
        "world": {
            "name": world.name if world is not None else "",
            "color": ([float(value) for value in world.color]
                      if world is not None else [0.0, 0.0, 0.0]),
            "uses_nodes": bool(world and world.use_nodes),
        },
        "lights": lights,
    }




_RENDER_GEOMETRY_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}


def _sr_ray_object_key(obj):
    """Stable key for an evaluated ray-cast object and its original object."""
    original = getattr(obj, "original", None) or obj
    try:
        return original.as_pointer()
    except ReferenceError:
        return 0


def _sr_boolean_operand_keys(scene):
    keys = set()
    for owner in scene.objects:
        for modifier in getattr(owner, "modifiers", ()):
            if modifier.type == 'BOOLEAN' and getattr(modifier, "object", None):
                keys.add(_sr_ray_object_key(modifier.object))
    return keys


def sr_renderable_mesh_objects(scene, view_layer=None):
    """Mesh objects that can contribute visible pixels to a final render.

    Read-only counterpart to ``_sr_render_visibility_state``, which has to
    mutate the scene so ray casts can see through viewport-hidden objects.
    Both apply the same rules, so the Splat seed and the sampled point cloud
    always agree about what the scene actually contains: an object excluded
    from the render is absent from the training images, and seeding Splats
    from it would add geometry the photos can never justify.
    """
    if scene is None:
        return []
    if view_layer is None:
        layers = getattr(scene, "view_layers", None)
        view_layer = layers[0] if layers else None

    render_hidden = set()

    def walk_collections(collection, inherited_hidden=False):
        hidden = inherited_hidden or bool(
            getattr(collection, "hide_render", False)
        )
        if hidden:
            render_hidden.add(collection.as_pointer())
        for child in collection.children:
            walk_collections(child, hidden)

    active_collections = set()
    holdout_collections = set()
    indirect_collections = set()

    def walk_layer(layer_collection, excluded=False, inherited_holdout=False,
                   inherited_indirect=False):
        collection = layer_collection.collection
        excluded = (
            excluded
            or layer_collection.exclude
            or bool(getattr(collection, "hide_render", False))
        )
        holdout = inherited_holdout or bool(
            getattr(layer_collection, "holdout", False)
        )
        indirect = inherited_indirect or bool(
            getattr(layer_collection, "indirect_only", False)
        )
        if not excluded:
            active_collections.add(collection.as_pointer())
            if holdout:
                holdout_collections.add(collection.as_pointer())
            if indirect:
                indirect_collections.add(collection.as_pointer())
        for child in layer_collection.children:
            walk_layer(child, excluded, holdout, indirect)

    walk_collections(scene.collection)
    if view_layer is not None:
        walk_layer(view_layer.layer_collection)
    boolean_operands = _sr_boolean_operand_keys(scene)

    renderable = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        collections = {
            collection.as_pointer() for collection in obj.users_collection
        }
        active_links = collections & active_collections
        all_render_hidden = bool(collections) and collections <= render_hidden
        all_holdout = bool(active_links) and active_links <= holdout_collections
        all_indirect = (
            bool(active_links) and active_links <= indirect_collections
        )
        try:
            object_holdout = bool(obj.is_holdout)
        except (AttributeError, RuntimeError):
            object_holdout = False
        try:
            object_indirect = bool(
                obj.indirect_only_get(view_layer=view_layer)
            ) if view_layer is not None else False
        except (AttributeError, RuntimeError, TypeError):
            object_indirect = False
        ignored = (
            (view_layer is not None and not active_links)
            or all_render_hidden
            or bool(obj.hide_render)
            or not bool(getattr(obj, "visible_camera", True))
            or object_holdout
            or all_holdout
            or object_indirect
            or all_indirect
            or _sr_ray_object_key(obj) in boolean_operands
        )
        if not ignored:
            renderable.append(obj)
    return renderable


def _sr_render_visibility_state(context, cfg):
    """Objects that cannot contribute visible pixels in this View Layer."""
    scene, view_layer = context.scene, context.view_layer
    state = {"objects": [], "layer_cols": [], "collections": [],
             "ignored_keys": set(), "revealed": [], "ignored": []}
    render_hidden, active_collections = set(), set()
    holdout_collections, indirect_collections = set(), set()

    def walk_collections(collection, inherited_hidden=False):
        hidden = inherited_hidden or bool(getattr(collection, "hide_render", False))
        if hidden:
            render_hidden.add(collection.as_pointer())
        for child in collection.children:
            walk_collections(child, hidden)

    def walk_layer(layer_collection, excluded=False, inherited_holdout=False,
                   inherited_indirect=False):
        collection = layer_collection.collection
        excluded = excluded or layer_collection.exclude or bool(getattr(collection, "hide_render", False))
        holdout = inherited_holdout or bool(getattr(layer_collection, "holdout", False))
        indirect = inherited_indirect or bool(
            getattr(layer_collection, "indirect_only", False))
        if not excluded:
            active_collections.add(collection.as_pointer())
            if holdout:
                holdout_collections.add(collection.as_pointer())
            if indirect:
                indirect_collections.add(collection.as_pointer())
            state["layer_cols"].append((layer_collection, layer_collection.hide_viewport))
            state["collections"].append((collection, collection.hide_viewport))
            if layer_collection.hide_viewport:
                layer_collection.hide_viewport = False
                state["revealed"].append(f"[collection] {collection.name}")
            if collection.hide_viewport:
                collection.hide_viewport = False
                state["revealed"].append(f"[collection] {collection.name}")
        for child in layer_collection.children:
            walk_layer(child, excluded, holdout, indirect)

    walk_collections(scene.collection)
    walk_layer(view_layer.layer_collection)
    # Point sampling must honour final render visibility, including helper
    # operands and holdouts, without asking the artist to maintain a second
    # set of exclusions.
    boolean_operands = _sr_boolean_operand_keys(scene)

    for obj in scene.objects:
        if obj.type not in _RENDER_GEOMETRY_TYPES:
            continue
        key = _sr_ray_object_key(obj)
        collections = {collection.as_pointer() for collection in obj.users_collection}
        active_links = collections & active_collections
        is_active = bool(active_links)
        all_render_hidden = bool(collections) and collections <= render_hidden
        all_holdout = bool(active_links) and active_links <= holdout_collections
        all_indirect = bool(active_links) and active_links <= indirect_collections
        try:
            object_holdout = bool(obj.is_holdout)
        except (AttributeError, RuntimeError):
            object_holdout = False
        visible_to_camera = bool(getattr(obj, "visible_camera", True))
        try:
            object_indirect = bool(obj.indirect_only_get(view_layer=view_layer))
        except (AttributeError, RuntimeError):
            object_indirect = False
        ignored = (not is_active or all_render_hidden or obj.hide_render
                   or not visible_to_camera
                   or object_holdout or all_holdout
                   or object_indirect or all_indirect
                   or key in boolean_operands)
        if ignored:
            state["ignored_keys"].add(key)
            state["ignored"].append(obj.name)
            continue
        try:
            hidden_vl = obj.hide_get(view_layer=view_layer)
        except Exception:
            hidden_vl = False
        state["objects"].append((obj, obj.hide_viewport, hidden_vl))
        if obj.hide_viewport:
            obj.hide_viewport = False
            state["revealed"].append(obj.name)
        if hidden_vl:
            try:
                obj.hide_set(False, view_layer=view_layer)
                state["revealed"].append(obj.name)
            except Exception:
                from . import diagnostics as _diag
                _diag.swallowed("sceneray_splat.py")
    try:
        view_layer.update()
    except Exception:
        from . import diagnostics as _diag
        _diag.swallowed("sceneray_splat.py")
    state["revealed"] = sorted(set(state["revealed"]))
    state["ignored"] = sorted(set(state["ignored"]))
    return state


def snapshot_and_reveal_render_geometry(context, cfg):
    """Expose viewport-hidden render geometry, while excluding helpers that
    cannot appear in the active View Layer's final render."""
    return _sr_render_visibility_state(context, cfg)


def restore_geometry_visibility(context, state):
    if not state:
        return
    for obj, hide_viewport, hidden_view_layer in state.get("objects", []):
        try:
            obj.hide_viewport = hide_viewport
        except Exception:
            from . import diagnostics as _diag
            _diag.swallowed("sceneray_splat.py")
        try:
            obj.hide_set(hidden_view_layer, view_layer=context.view_layer)
        except Exception:
            from . import diagnostics as _diag
            _diag.swallowed("sceneray_splat.py")
    for collection, hide_viewport in state.get("collections", []):
        try:
            collection.hide_viewport = hide_viewport
        except Exception:
            from . import diagnostics as _diag
            _diag.swallowed("sceneray_splat.py")
    for layer_collection, hide_viewport in state.get("layer_cols", []):
        try:
            layer_collection.hide_viewport = hide_viewport
        except Exception:
            from . import diagnostics as _diag
            _diag.swallowed("sceneray_splat.py")
    try:
        context.view_layer.update()
    except Exception:
        from . import diagnostics as _diag
        _diag.swallowed("sceneray_splat.py")

# ═══════════════════════════════════════════════════════════════════════
#  8. SCENE-STATE SNAPSHOT / RESTORE
#     Hard rule: no camera property and no render setting is ever written.
#     The only transient changes are the output filepath + still-image
#     format (so frames use the selected dataset format) and active camera.
# ═══════════════════════════════════════════════════════════════════════

_cancel_flag = {"requested": False}
_render_job_state = {
    "owner": None, "scene_pointer": None, "started": False,
    "complete": False, "cancelled": False,
}
_render_modal_wakeup = {"timer_registered": False}
_render_batch_runtime = {"batch": None, "timer_registered": False}
_render_dataset_continuation = {
    "scene_pointer": None,
    "window_pointer": None,
    "timer_registered": False,
}


def _sr_dispatch_render_modal_wakeup():
    """Wake the batch modal after Blender's nested render job ends.

    A window-bound event timer can stop delivering events when Blender changes
    the active editor or opens a render display. Render completion therefore
    sends a synthetic timer event to every Blender window.
    """
    _render_modal_wakeup["timer_registered"] = False
    if _render_job_state["owner"] is None:
        return None
    try:
        windows = tuple(bpy.context.window_manager.windows)
    except (AttributeError, ReferenceError, RuntimeError):
        windows = ()
    for window in windows:
        try:
            window.event_simulate(type='TIMER', value='NOTHING')
        except (AttributeError, ReferenceError, RuntimeError):
            pass
    return None


def _sr_schedule_render_modal_wakeup():
    if _render_modal_wakeup["timer_registered"]:
        return
    _render_modal_wakeup["timer_registered"] = True
    try:
        bpy.app.timers.register(
            _sr_dispatch_render_modal_wakeup, first_interval=0.01)
    except (ReferenceError, RuntimeError, ValueError):
        _render_modal_wakeup["timer_registered"] = False


def _sr_render_job_init(scene, *args):
    try:
        if _render_job_state["scene_pointer"] == scene.as_pointer():
            _render_job_state["started"] = True
        batch = _render_batch_runtime["batch"]
        if batch is not None and batch.matches_scene(scene):
            batch.render_started = True
    except ReferenceError:
        pass


def _sr_render_job_complete(scene, *args):
    try:
        if _render_job_state["scene_pointer"] == scene.as_pointer():
            _render_job_state["complete"] = True
            _sr_schedule_render_modal_wakeup()
        batch = _render_batch_runtime["batch"]
        if batch is not None and batch.matches_scene(scene):
            batch.render_complete = True
            _sr_schedule_render_batch_tick(0.01)
    except ReferenceError:
        pass


def _sr_render_job_cancel(scene, *args):
    try:
        if _render_job_state["scene_pointer"] == scene.as_pointer():
            _render_job_state["cancelled"] = True
            _sr_schedule_render_modal_wakeup()
        batch = _render_batch_runtime["batch"]
        if batch is not None and batch.matches_scene(scene):
            batch.render_cancelled = True
            _sr_schedule_render_batch_tick(0.01)
    except ReferenceError:
        pass


def _sr_clear_render_job(owner=None):
    if owner is not None and _render_job_state["owner"] is not owner:
        return
    _render_job_state.update({
        "owner": None, "scene_pointer": None, "started": False,
        "complete": False, "cancelled": False,
    })


def _tag_redraw_sceneray_splat(context):
    """Refresh every place that can show the dataset build monitor.

    Blender may move Render Result into a temporary window, so limiting the
    redraw to the screen that launched the job leaves that window frozen.
    """
    window_manager = getattr(context, "window_manager", None)
    if window_manager is None:
        window_manager = getattr(bpy.context, "window_manager", None)
    scene = getattr(context, "scene", None)
    cfg = getattr(scene, "SCENERAY_SPLAT", None)
    build_active = bool(
        cfg is not None and (cfg.is_rendering or cfg.is_generating_points)
    )
    for window in getattr(window_manager, "windows", ()):
        for area in window.screen.areas:
            if area.type not in {
                'OUTLINER', 'IMAGE_EDITOR', 'VIEW_3D', 'PROPERTIES'
            }:
                continue
            if area.type == 'IMAGE_EDITOR' and build_active:
                space = area.spaces.active
                image = getattr(space, "image", None)
                if image is not None and (
                    getattr(image, "type", "") == 'RENDER_RESULT'
                    or getattr(image, "name", "") == "Render Result"
                ):
                    try:
                        space.show_region_ui = True
                    except (AttributeError, TypeError):
                        pass
            area.tag_redraw()


def _fmt_eta(secs):
    secs = int(max(0, secs))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60:02d}s"
    return f"{secs // 3600}h {(secs % 3600) // 60:02d}m"


def snapshot_render_output_state(scene):
    render = scene.render
    ims = render.image_settings
    state = {
        "filepath": render.filepath,
        "frame_current": scene.frame_current,
        "file_format": ims.file_format,
        "color_mode": ims.color_mode,
        "color_depth": ims.color_depth,
        "quality": ims.quality,
        "compression": ims.compression,
        "color_management": ims.color_management,
        "film_transparent": render.film_transparent,
    }
    cfg = getattr(scene, "SCENERAY_SPLAT", None)
    selected = str(getattr(cfg, "dataset_image_format", "PNG"))
    file_format = _SR_DATASET_IMAGE_FORMATS.get(selected, ("PNG", ".png"))[0]
    ims.file_format = file_format
    # Only the file encoding is chosen here. The user's Output Properties
    # colour-management mode is never replaced: an OVERRIDE/Standard output
    # must not be silently changed to the scene's AgX transform.
    try:
        ims.color_depth = "8"
    except TypeError:
        pass
    if file_format == "PNG":
        # The dataset uses a separate geometry mask. The separate geometry mask is
        # authoritative, so PNG is encoded as color-only RGB.
        try:
            ims.color_mode = 'RGB'
        except TypeError:
            pass
        try:
            ims.compression = int(getattr(cfg, "png_compression", 15))
        except (AttributeError, TypeError, ValueError):
            pass
    else:
        # JPEG has no alpha channel; keep whatever film the scene uses.
        try:
            ims.color_mode = 'RGB'
        except TypeError:
            pass
        try:
            ims.quality = int(getattr(cfg, "jpeg_quality", 90))
        except (AttributeError, TypeError, ValueError):
            pass
    return state


#: The user's own Render Display preference, held for the render phase.
#: Only the first image is allowed to create a temporary Render window. Once
#: Blender has created it, later images update Render Result without opening
#: or restoring a window the user minimized or closed.
_render_display = {
    "saved": None,
    "window_seen": False,
    "suppressed": False,
}


def show_render_window():
    """Allow Blender's native Render window to open for the first image.

    After that window has appeared, ``settle_render_window`` changes the
    display preference to NONE for the remainder of this batch. The existing
    Render Result still refreshes, but Blender cannot pop a minimized window
    back up or recreate one the user deliberately closed.
    """
    try:
        view = bpy.context.preferences.view
        if _render_display["saved"] is None:
            _render_display["saved"] = view.render_display_type
            _render_display["window_seen"] = False
            _render_display["suppressed"] = False
        if not _render_display["suppressed"]:
            view.render_display_type = 'WINDOW'
    except (AttributeError, TypeError):
        pass


def settle_render_window(force=False):
    """Stop later images from creating or raising a Render window.

    Normally this waits until Blender has actually created the first temporary
    Image Editor window. ``force`` is the completion fallback for very fast
    first renders whose window appeared and disappeared between timer ticks.
    """
    if _render_display["saved"] is None or _render_display["suppressed"]:
        return
    seen = False
    try:
        for window in tuple(bpy.context.window_manager.windows):
            screen = getattr(window, "screen", None)
            if screen is None or not getattr(screen, "is_temporary", False):
                continue
            if any(area.type == 'IMAGE_EDITOR' for area in screen.areas):
                seen = True
                break
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    if not seen and not force:
        return
    try:
        bpy.context.preferences.view.render_display_type = 'NONE'
        _render_display["window_seen"] = bool(
            _render_display["window_seen"] or seen
        )
        _render_display["suppressed"] = True
    except (AttributeError, TypeError):
        pass


def restore_render_display():
    """Give the Render Display preference back exactly as it was."""
    saved = _render_display["saved"]
    _render_display["saved"] = None
    _render_display["window_seen"] = False
    _render_display["suppressed"] = False
    if saved is None:
        return
    try:
        bpy.context.preferences.view.render_display_type = saved
    except (AttributeError, TypeError):
        pass


def restore_render_output_state(scene, state):
    if not state:
        return
    render = scene.render
    ims = render.image_settings
    try:
        ims.file_format = state["file_format"]
    except TypeError:
        pass
    for key in (
        "color_mode",
        "color_depth",
        "quality",
        "compression",
        "color_management",
    ):
        try:
            setattr(ims, key, state[key])
        except TypeError:
            pass
    try:
        render.film_transparent = state["film_transparent"]
    except (AttributeError, KeyError, TypeError):
        pass
    render.filepath = state["filepath"]
    try:
        scene.frame_set(state["frame_current"])
    except Exception:
        scene.frame_current = state["frame_current"]


def warn_render_settings(op, scene, render):
    """Coaching only — never writes anything."""
    tips = []
    if render.engine == 'CYCLES':
        try:
            if not scene.cycles.use_denoising:
                tips.append("Denoise is OFF. 3DGS fits leftover render noise as "
                            "floaters — consider Render Properties ▸ Sampling ▸ "
                            "Denoise.")
            if scene.cycles.samples < 32:
                tips.append(f"Cycles samples = {scene.cycles.samples} (low). "
                            "Grainy frames train as fuzz; ~64+ with denoise is a "
                            "good baseline.")
        except AttributeError:
            pass
    else:
        tips.append(f"Render engine is {render.engine}. Eevee's view-dependent "
                    "shading can confuse a splat — Cycles is safer.")
    eff_w, eff_h = effective_resolution(render)
    if min(eff_w, eff_h) < 720:
        tips.append(f"Effective resolution is {eff_w}x{eff_h}. Splat detail "
                    "scales with pixels — 960x720 or higher is sharper.")
    if tips:
        print("[sceneray_splat] using your scene's render settings as-is; coaching:")
        for t in tips:
            print(f"  • {t}")
            op.report({'WARNING'}, t)


_BLANK_STD = 0.04


def _is_blank_render(path):
    """True if the render is near-uniform (empty sky/void) — advisory only,
    nothing is ever deleted."""
    try:
        img = bpy.data.images.load(str(path), check_existing=False)
        try:
            n = len(img.pixels)
            if not n:
                return True
            import numpy as np
            buf = np.empty(n, dtype=np.float32)
            img.pixels.foreach_get(buf)
            return float(buf.reshape(-1, 4)[:, :3].std()) < _BLANK_STD
        finally:
            bpy.data.images.remove(img)
    except Exception:
        return False


def _show_camera_in_viewports(context, cam):
    """Only changes which camera is active for viewing."""
    context.scene.camera = cam
    for window in context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type != 'VIEW_3D':
                    continue
                if getattr(space, "use_local_camera", False):
                    space.camera = cam
                r3d = getattr(space, "region_3d", None)
                if r3d is not None:
                    r3d.view_perspective = 'CAMERA'
            area.tag_redraw()


# ═══════════════════════════════════════════════════════════════════════
#  9. STEP 1 — RENDER IMAGES
# ═══════════════════════════════════════════════════════════════════════


class SCENERAY_SPLAT_OT_render_images(bpy.types.Operator):
    """STEP 1 — Render RGB plus one geometry-validity mask per camera.

    Renders with your scene's settings exactly as configured — engine,
    samples, denoise, resolution and transparency are never overridden.

    Incremental and resumable: cameras whose image already exists are
    skipped, progress is saved after every frame, and a camera that fails
    is reported and skipped past instead of aborting the run.

    Optional numerical passes are isolated below ``sg_metadata``. Metric
    depth is retained with the standard dataset for RGB-D point generation."""
    bl_idname = "sceneray_splat.render_images"
    bl_label = "1. Render Images"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and not cfg.is_rendering
                and len(cfg.camera_queue) > 0)

    def _setup(self, context):
        if not bpy.data.is_saved:
            raise ValueError("Save the .blend file before building the dataset.")
        scene = context.scene
        cfg = scene.SCENERAY_SPLAT
        self.scene, self.cfg = scene, cfg

        missing, duplicate_count = _sr_clean_camera_queue(
            scene, cfg, remove_duplicates=True
        )
        if missing or duplicate_count:
            self.report(
                {"INFO"},
                f"Automatic camera cleanup removed {missing} missing and "
                f"{duplicate_count} duplicate camera(s).",
            )
        cameras = gather_scene_cameras(context, cfg)
        if not cameras:
            raise ValueError("The render queue is empty - add cameras in "
                             "the Camera List panel first.")
        self.cameras = cameras
        self._orig_camera = scene.camera

        render = scene.render
        self._output_state = snapshot_render_output_state(scene)
        warn_render_settings(self, scene, render)
        self.render = render
        self.eff_res = effective_resolution(render)

        output_path = _sr_effective_output_path(cfg)
        if output_path is None:
            raise ValueError("Output Dir is empty - pick a folder to write "
                             "images/ into.")
        output_dir = Path(output_path)
        if not output_dir.is_absolute():
            raise ValueError(
                f"Output Dir resolved to a relative path ({output_dir}). Save "
                "the .blend file first, or set Output Dir to an absolute path.")
        if "#" in str(output_dir):
            raise ValueError(
                "Output path contains '#', which Blender replaces with frame "
                "numbers when writing - rename the folder.")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            probe = output_dir / ".sceneray_splat_write_test"
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            raise ValueError(
                f"Output folder is not writable: {output_dir} ({exc})")
        images_dir = _sr_data_dir(output_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir, self.images_dir = output_dir, images_dir

        # Reconcile Blender state with the tracking file and real images before
        # deciding whether a camera belongs in this batch.
        _sr_sync_render_states(scene, cfg, output_dir)
        previous = (read_render_manifest(output_dir)
                    or read_completed_camera_data(output_dir) or {})
        previous_entries = {
            entry.get("name"): entry
            for entry in previous.get("cameras", [])
            if isinstance(entry, dict) and entry.get("name")
        }
        resolution_changed = (
            bool(previous_entries)
            and tuple(previous.get("resolution", ())) != tuple(self.eff_res))
        if resolution_changed:
            self.report({"WARNING"},
                        "Render resolution changed - all cameras are pending.")
            for item in cfg.camera_queue:
                item.render_state = "PENDING"

        allocated = []
        for entry in previous_entries.values():
            try:
                allocated.append(int(entry.get("frame_index")))
            except (TypeError, ValueError):
                pass
        next_frame = max(allocated, default=-1) + 1

        self.reused, self.queue = [], []
        self.records, self.records_by_name = [], {}
        for camera in cameras:
            item = _sr_queue_item(cfg, camera)
            old = previous_entries.get(camera.name, {})
            try:
                frame_index = int(old.get("frame_index"))
            except (TypeError, ValueError):
                frame_index, next_frame = next_frame, next_frame + 1
            extension = _sr_dataset_image_extension(cfg)
            desired_path = _sr_dataset_master_relative_path(
                cfg, frame_index, output_dir
            )
            desired_training_path = _sr_dataset_training_relative_path(
                cfg,
                frame_index,
                desired_path,
            )
            rel_path = str(old.get("file_path", desired_path))
            if not rel_path:
                rel_path = desired_path
            training_rel_path = str(
                old.get("training_file_path", desired_training_path)
            )
            if not training_rel_path:
                training_rel_path = desired_training_path
            image_path = output_dir / rel_path.lstrip("./\\")
            training_image_path = (
                output_dir / training_rel_path.lstrip("./\\")
            )
            image_exists = image_path.is_file()
            training_image_exists = training_image_path.is_file()
            format_changed = Path(rel_path).suffix.lower() != extension
            signature = bd_manifest.camera_signature(camera, cfg)
            settings_changed = bool(old) and not bd_manifest.matches(
                old, signature
            )
            pending = (
                resolution_changed
                or format_changed
                or settings_changed
                or item is None
                or item.render_state == "PENDING"
                or not image_exists
                or not training_image_exists
                or (
                    bd_export.enabled(cfg)
                    and not bd_export.view_complete(
                        output_dir, frame_index, cfg
                    )
                )
            )
            if pending:
                # Re-render into the format currently selected in Build Dataset.
                rel_path = desired_path
                training_rel_path = desired_training_path
            if item is not None:
                item.render_state = "PENDING" if pending else "RENDERED"
            record = {
                "name": camera.name,
                "file_path": rel_path,
                "training_file_path": training_rel_path,
                "frame_index": frame_index,
                "pending": pending,
                "status": "PENDING" if pending else "RENDERED",
                # A reused image keeps the signature it was rendered with; a
                # pending one gets its signature when the render finishes.
                "signature": (
                    bd_manifest.signature_of(old) if not pending else None
                ),
            }
            if record["signature"] is None:
                record.pop("signature")
            self.records.append(record)
            self.records_by_name[camera.name] = record
            if pending:
                self.queue.append((camera, frame_index, rel_path))
            else:
                self.reused.append(record)

        self.kept = list(self.reused)
        self.failed = []
        self.metadata_failures = []
        self.suspect_blank = []
        self._manifest_error = ""
        self._fatal_error = ""
        _clean_obsolete_dataset_layout(output_dir)
        # Write the complete planned batch before the first expensive render.
        # If Blender closes mid-batch, remaining cameras stay explicitly pending.
        if not self._persist_manifest():
            raise ValueError(self._manifest_error)

        if self.queue:
            # Rendering anything makes the existing metadata describe a set of
            # images that no longer exists, so it is removed up front and
            # rewritten from the finished renders. The point cloud is not
            # metadata about the images and may have been built first, so it
            # is deliberately left alone.
            for name in bd_paths.STAGE1_FILES:
                try:
                    _sr_dataset_file(output_dir, name).unlink(missing_ok=True)
                except OSError:
                    pass
            dataset = _dataset_manifest_path(output_dir)
            if dataset.is_file():
                try:
                    dataset.unlink()
                except OSError:
                    pass

        self._dataset_capture = None
        if self.queue and bd_export.enabled(cfg):
            try:
                self._dataset_capture = bd_export.begin_capture(
                    scene, cfg, output_dir, self.eff_res
                )
            except Exception as exc:
                message = f"Could not prepare SplatGen metadata passes: {exc}"
                self.metadata_failures.append(("setup", message))
                print(f"[sceneray_splat] WARNING: {message}; RGB build continues")
        # Raw beauty passes ride along with this same render (best-effort).
        raw_hooks.batch_begin(self)


    def _persist_manifest(self):
        """Save once, then suppress repeated writes after a storage failure."""
        if self._manifest_error:
            return False
        try:
            write_render_manifest(
                self.output_dir,
                self.records,
                self.eff_res,
                color_management=_sr_color_management_snapshot(
                    self.scene,
                    self.cfg,
                ),
            )
            return True
        except OSError as exc:
            self._manifest_error = _sr_storage_error_message(
                exc, self.output_dir)
            print(f"[sceneray_splat] {self._manifest_error}")
            return False


    def _prepare_render_one(self, qi):
        cam_obj, frame_index, manifest_path = self.queue[qi]
        self.scene.camera = cam_obj
        _clip_start, clip_end = bd_manifest.effective_clipping(
            cam_obj.data, self.cfg
        )
        bd_export.prepare_view(
            getattr(self, "_dataset_capture", None), frame_index, clip_end
        )
        raw_hooks.batch_prepare(self, frame_index)
        _sr_check_render_disk_space(
            self.output_dir, self.render, self.eff_res)
        bpy.context.view_layer.update()
        relative = manifest_path.lstrip("./\\").replace("\\", "/")
        expected_relative = _sr_dataset_master_relative_path(
            self.cfg, frame_index, self.output_dir
        ).lstrip("./\\")
        if relative != expected_relative:
            relative = _sr_dataset_master_relative_path(
                self.cfg,
                frame_index,
                self.output_dir,
            ).lstrip("./\\")
        render_path = self.output_dir / relative
        render_path.parent.mkdir(parents=True, exist_ok=True)
        self.render.filepath = str(render_path)
        before_signature = None
        try:
            stat = render_path.stat()
            before_signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            pass
        return (
            cam_obj,
            frame_index,
            relative,
            render_path,
            before_signature,
        )

    def _finalize_render_one(self, qi, prepared, result=None):
        cam_obj, frame_index, relative, render_path, _before = prepared
        if not render_path.is_file():
            outcome = sorted(result or {'CANCELLED'})
            raise RuntimeError(
                f"render returned {outcome} but no file appeared at "
                f"{render_path} (engine: {self.render.engine})")
        if _is_blank_render(render_path):
            self.suspect_blank.append(cam_obj.name)
            print(f"[sceneray_splat] NOTE: {relative} looks blank/uniform "
                  f"(camera: {cam_obj.name}) - kept anyway.")
        try:
            bd_export.finish_view(
                getattr(self, "_dataset_capture", None), frame_index
            )
        except Exception as exc:
            # Numerical metadata is enriched training input, not permission to
            # suppress a valid RGB render or the canonical COLMAP text files.
            # Keep the view and report the exact pass failure after the batch.
            message = str(exc)
            self.metadata_failures.append((cam_obj.name, message))
            print(
                f"[sceneray_splat] WARNING: metadata for '{cam_obj.name}' "
                f"failed: {message}; canonical dataset continues"
            )
        raw_hooks.batch_finish(self, cam_obj.name, frame_index)
        training_relative = _sr_dataset_training_relative_path(
            self.cfg,
            frame_index,
            f"./{relative}",
        ).lstrip("./\\")
        record = self.records_by_name[cam_obj.name]
        record["file_path"] = f"./{relative}"
        record["training_file_path"] = f"./{training_relative}"
        record["frame_index"] = frame_index
        record["pending"] = False
        record["status"] = "RENDERED"
        # Capture what this image was rendered with, so any later change to
        # the camera shows up as a mismatch and marks it for re-rendering.
        record["signature"] = bd_manifest.camera_signature(cam_obj, self.cfg)
        record.pop("last_error", None)
        if record not in self.kept:
            self.kept.append(record)
        item = _sr_queue_item(self.cfg, cam_obj)
        if item is not None:
            item.render_state = "RENDERED"
            # Turn this camera blue in the 3D View the moment its image lands,
            # so the viewport tracks the render rather than a panel redraw.
            _sr_apply_camera_status_colors(self.cfg)
        if not self._persist_manifest():
            raise SceneRaySplatStorageError(self._manifest_error)
        print(f"[sceneray_splat] rendered {qi + 1}/{len(self.queue)} -> "
              f"{relative} (camera: {cam_obj.name})")

    def _render_one(self, qi):
        prepared = self._prepare_render_one(qi)
        try:
            result = bpy.ops.render.render(
                'EXEC_DEFAULT', write_still=True, scene=self.scene.name)
        except RuntimeError as exc:
            raise RuntimeError(
                f"bpy.ops.render.render(write_still=True) raised: {exc}")
        if 'CANCELLED' in result:
            raise RuntimeError("Render cancelled by user")
        if 'FINISHED' not in result:
            raise RuntimeError(
                f"render returned {sorted(result)} instead of FINISHED")
        self._finalize_render_one(qi, prepared, result)

    def _start_render_one_async(self, qi):
        prepared = self._prepare_render_one(qi)
        _render_job_state.update({
            "owner": self, "scene_pointer": self.scene.as_pointer(),
            "started": False, "complete": False, "cancelled": False,
        })
        try:
            show_render_window()
            result = bpy.ops.render.render(
                'INVOKE_DEFAULT', write_still=True, scene=self.scene.name)
        except Exception:
            _sr_clear_render_job(self)
            raise
        self._active_render = (qi, prepared)
        self._render_waiting = True
        if 'FINISHED' in result:
            _render_job_state["complete"] = True
            _sr_schedule_render_modal_wakeup()
        elif 'CANCELLED' in result:
            _render_job_state["cancelled"] = True
            _sr_schedule_render_modal_wakeup()

    def _record_failure(self, camera, error):
        record = self.records_by_name.get(camera.name)
        if record is not None:
            record["pending"] = True
            record["status"] = "PENDING"
            record["last_error"] = str(error)
        item = _sr_queue_item(self.cfg, camera)
        if item is not None:
            item.render_state = "PENDING"
        return self._persist_manifest()


    def _summary(self):
        n_new = len(self.kept) - len(self.reused)
        parts = [f"{n_new} rendered", f"{len(self.reused)} reused"]
        if self.failed:
            parts.append(f"{len(self.failed)} FAILED")
        if self.suspect_blank:
            parts.append(f"{len(self.suspect_blank)} look blank")
        return ", ".join(parts)

    def _finish(self):
        # Raw first: its nodes live inside the legacy capture's tree.
        raw_hooks.batch_restore(self)
        bd_export.restore_capture(getattr(self, "_dataset_capture", None))
        self._dataset_capture = None
        restore_render_output_state(
            self.scene, getattr(self, "_output_state", None))
        restore_render_display()
        if self._orig_camera is not None:
            self.scene.camera = self._orig_camera
        # Always persist the entire managed list. Successful, reused, failed,
        # cancelled, and not-yet-started cameras all keep an explicit status.
        if not self._persist_manifest():
            self.report({"ERROR"}, self._manifest_error)
            return False
        if not self.kept:
            if self.failed:
                for name, error in self.failed:
                    print(f"[sceneray_splat] camera '{name}' failed: {error}")
                first_name, first_error = self.failed[0]
                self.report({"ERROR"},
                            f"No images rendered - {len(self.failed)} camera(s) "
                            f"failed; first was '{first_name}': {first_error}")
            else:
                self.report({"ERROR"},
                            "No valid rendered images are available.")
            return False
        for name, error in self.failed:
            self.report({"ERROR"},
                        f"Camera '{name}' failed to render: {error}")
        if self.suspect_blank:
            self.report({"WARNING"},
                        "These renders look blank/uniform (kept): "
                        + ", ".join(self.suspect_blank))
        if self.metadata_failures:
            name, error = self.metadata_failures[0]
            self.report(
                {"WARNING"},
                f"RGB rendering completed, but SplatGen metadata had "
                f"{len(self.metadata_failures)} error(s); first at "
                f"'{name}': {error}. Canonical files will still be built.",
            )
        return True


    def execute(self, context):
        if not bpy.app.background:
            return _sr_begin_render_batch(context, self)
        try:
            self._setup(context)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        for qi in range(len(self.queue)):
            try:
                self._render_one(qi)
            except SceneRaySplatStorageError as e:
                self._fatal_error = str(e)
                break
            except Exception as e:
                name = self.queue[qi][0].name
                if not self._record_failure(self.queue[qi][0], e):
                    self._fatal_error = self._manifest_error
                    break
                self.failed.append((name, str(e)))
                print(f"[sceneray_splat] FAILED camera '{name}': {e} — continuing")
        if not self._finish():
            return {'CANCELLED'}
        self.report({'INFO'}, f"Step 1 done ({self._summary()}) → "
                    f"{self.images_dir} — next: step 2.")
        if self.cfg.write_metadata_after_render:
            return _sr_continue_dataset_generation(context)
        return {'FINISHED'}

    def invoke(self, context, event):
        if bpy.app.background:
            return self.execute(context)
        return _sr_begin_render_batch(context, self)

    def _advance_modal_progress(self, context):
        self._i += 1
        wm = context.window_manager
        wm.progress_update(
            sr_stage_progress(self._i / max(1, self._n), "RENDER")
        )
        self.cfg.render_progress = self._i
        self.cfg.render_progress_fac = self._i / max(1, self._n)
        eta = _fmt_eta((time.time() - self._t0) / max(1, self._i)
                       * (self._n - self._i))
        self.cfg.render_eta = eta
        _tag_redraw_sceneray_splat(context)
        context.workspace.status_text_set(
            f"SplatGen: rendered {self._i}/{self._n} "
            f"({100.0 * self._i / max(1, self._n):.0f}%) · ~{eta} left")

    def modal(self, context, event):
        escape = event.type == 'ESC' and event.value == 'PRESS'
        if escape:
            _cancel_flag["requested"] = True
            self._cancelled = True
            self.cfg.render_status = "Cancelling render…"
            _tag_redraw_sceneray_splat(context)
            if self._render_waiting:
                # Let Blender's own render operator receive Escape immediately.
                return {'PASS_THROUGH'}
        if _cancel_flag["requested"]:
            self._cancelled = True
            if self._render_waiting:
                self.cfg.render_status = (
                    "Stop requested — waiting for the active image. "
                    "Press Escape to cancel that render immediately.")
                _tag_redraw_sceneray_splat(context)
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._render_waiting:
            settle_render_window()
            state_matches = _render_job_state["owner"] is self
            complete = state_matches and _render_job_state["complete"]
            cancelled = state_matches and _render_job_state["cancelled"]
            if (state_matches and _render_job_state["started"]
                    and not complete and not cancelled
                    and not _sr_render_job_running()
                    and time.time() - self._active_render_started_at > 0.25):
                prepared_path = self._active_render[1][3]
                complete = prepared_path.is_file()
                cancelled = not complete
            if not complete and not cancelled:
                return {'RUNNING_MODAL'}

            # The first Render window has now had a chance to appear. Keep
            # every later image from reopening or raising it.
            settle_render_window(force=True)

            qi, prepared = self._active_render
            camera = self.queue[qi][0]
            self._render_waiting = False
            self._active_render = None
            _sr_clear_render_job(self)
            if cancelled:
                self._record_failure(camera, "Render cancelled")
                self._cancelled = True
                return self._wrap_up(context)
            try:
                self._finalize_render_one(qi, prepared, {'FINISHED'})
            except SceneRaySplatStorageError as exc:
                self._fatal_error = str(exc)
                self._cancelled = True
                return self._wrap_up(context)
            except Exception as exc:
                if not self._record_failure(camera, exc):
                    self._fatal_error = self._manifest_error
                    self._cancelled = True
                    return self._wrap_up(context)
                self.failed.append((camera.name, str(exc)))
                self.report({'WARNING'},
                            f"Camera '{camera.name}' failed: {exc} — continuing")
            self._advance_modal_progress(context)
            if self._cancelled:
                return self._wrap_up(context)
            # Continue in this same wake-up event. Waiting for one more event
            # here could strand the queue after a render-display context switch.

        if self._cancelled or self._i >= self._n:
            return self._wrap_up(context)
        try:
            self._start_render_one_async(self._i)
            self._active_render_started_at = time.time()
            self.cfg.render_status = (
                f"Rendering camera {self._i + 1}/{self._n} — UI remains active")
            _tag_redraw_sceneray_splat(context)
        except SceneRaySplatStorageError as exc:
            self._fatal_error = str(exc)
            self._cancelled = True
            return self._wrap_up(context)
        except Exception as exc:
            camera = self.queue[self._i][0]
            if not self._record_failure(camera, exc):
                self._fatal_error = self._manifest_error
                self._cancelled = True
                return self._wrap_up(context)
            self.failed.append((camera.name, str(exc)))
            self.report({'WARNING'},
                        f"Camera '{camera.name}' failed: {exc} — continuing")
            self._advance_modal_progress(context)
        return {'RUNNING_MODAL'}
    def _wrap_up(self, context):
        _sr_clear_render_job(self)
        self._render_waiting = False
        self._active_render = None
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        context.workspace.status_text_set(None)
        self.cfg.is_rendering = False
        self.cfg.render_eta = ""
        if self._fatal_error:
            self.cfg.render_progress_fac = self._i / max(1, self._n)
            self.cfg.render_status = f"Render stopped — {self._fatal_error}"
        elif self._cancelled:
            self.cfg.render_progress_fac = self._i / max(1, self._n)
            self.cfg.render_status = (f"Cancelled at {self._i}/{self._n} — "
                                      "progress saved, click step 1 to resume")
        else:
            self.cfg.render_progress_fac = 1.0
            self.cfg.render_status = f"Render finished — {self._summary()}"
        _cancel_flag["requested"] = False
        _tag_redraw_sceneray_splat(context)
        ok = self._finish()
        if self._fatal_error:
            self.cfg.write_metadata_after_render = False
            self.report({'ERROR'}, self._fatal_error)
            return {'CANCELLED'}
        if self._cancelled:
            self.cfg.write_metadata_after_render = False
            self.report({'WARNING'},
                        f"Render cancelled — {self._summary()} → "
                        f"{self.images_dir} — click step 1 again to resume."
                        if ok else "Render cancelled — no images kept.")
            return {'CANCELLED'}
        if not ok:
            self.cfg.write_metadata_after_render = False
            return {'CANCELLED'}
        self.report({'INFO'}, f"Step 1 done ({self._summary()}) → "
                    f"{self.images_dir} — next: step 2.")
        if self.cfg.write_metadata_after_render:
            return _sr_continue_dataset_generation(context)
        return {'FINISHED'}


class _SceneRaySplatRenderBatch:
    """Window-independent owner for the responsive multi-camera batch."""

    _setup = SCENERAY_SPLAT_OT_render_images._setup
    _persist_manifest = SCENERAY_SPLAT_OT_render_images._persist_manifest
    _prepare_render_one = (
        SCENERAY_SPLAT_OT_render_images._prepare_render_one
    )
    _finalize_render_one = (
        SCENERAY_SPLAT_OT_render_images._finalize_render_one
    )
    _record_failure = SCENERAY_SPLAT_OT_render_images._record_failure
    _summary = SCENERAY_SPLAT_OT_render_images._summary
    _finish = SCENERAY_SPLAT_OT_render_images._finish

    def __init__(self, reporter=None):
        self._reporter = reporter
        self._i = 0
        self._n = 0
        self.phase = "READY"
        self.active_qi = None
        self.active_prepared = None
        self.render_started = False
        self.render_complete = False
        self.render_cancelled = False
        self.render_launch_time = 0.0
        self.render_idle_since = None
        self.window_pointer = None
        self.scene_pointer = None
        self.complete_workflow = False
        self.completed_durations = []
        self.finished = False

    def report(self, levels, message):
        if self._reporter is not None:
            try:
                self._reporter.report(levels, message)
                return
            except (ReferenceError, RuntimeError):
                self._reporter = None
        severity = next(iter(levels), "INFO") if levels else "INFO"
        print(f"[sceneray_splat] {severity}: {message}")

    def matches_scene(self, scene):
        if self.finished or self.phase != "WAIT_RENDER":
            return False
        try:
            return self.scene_pointer == scene.as_pointer()
        except ReferenceError:
            return False

    def output_changed(self):
        if self.active_prepared is None:
            return False
        render_path = self.active_prepared[3]
        before = self.active_prepared[4]
        try:
            stat = render_path.stat()
            return before != (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return False


def _sr_live_window_context(window_pointer=None):
    """Resolve fresh window/area references for every image render."""
    try:
        windows = tuple(bpy.context.window_manager.windows)
    except (AttributeError, ReferenceError, RuntimeError):
        return None, None, None
    window = None
    for candidate in windows:
        try:
            if (
                window_pointer is not None
                and candidate.as_pointer() == window_pointer
            ):
                window = candidate
                break
        except ReferenceError:
            continue
    if window is None and windows:
        window = windows[0]
    if window is None or window.screen is None:
        return None, None, None
    areas = tuple(window.screen.areas)
    area = next((item for item in areas if item.type == "VIEW_3D"), None)
    if area is None:
        area = next(
            (item for item in areas if item.type != "TOPBAR"),
            None,
        )
    if area is None:
        return window, None, None
    region = next(
        (item for item in area.regions if item.type == "WINDOW"),
        None,
    )
    return window, area, region


def _sr_render_job_running():
    """True until both rendering and compositor evaluation are fully idle."""
    try:
        return bool(
            bpy.app.is_job_running("RENDER")
            or bpy.app.is_job_running("COMPOSITE")
        )
    except (AttributeError, RuntimeError):
        return False


def _sr_schedule_render_batch_tick(delay=0.05):
    if _render_batch_runtime["batch"] is None:
        return
    if _render_batch_runtime["timer_registered"]:
        return
    _render_batch_runtime["timer_registered"] = True
    try:
        bpy.app.timers.register(
            _sr_render_batch_tick,
            first_interval=max(0.01, float(delay)),
        )
    except (ReferenceError, RuntimeError, ValueError):
        _render_batch_runtime["timer_registered"] = False


def _sr_launch_batch_render(batch):
    if _sr_render_job_running():
        return False
    qi = batch._i
    prepared = batch._prepare_render_one(qi)
    batch.active_qi = qi
    batch.active_prepared = prepared
    batch.render_started = False
    batch.render_complete = False
    batch.render_cancelled = False
    batch.render_launch_time = time.time()
    batch.render_idle_since = None
    batch.phase = "WAIT_RENDER"
    batch.cfg.render_status = (
        f"Rendering camera {qi + 1}/{batch._n}"
    )
    _tag_redraw_sceneray_splat(bpy.context)

    window, area, region = _sr_live_window_context(batch.window_pointer)
    if window is None:
        raise RuntimeError(
            "No live Blender window is available to start rendering."
        )
    override = {
        "window": window,
        "screen": window.screen,
        "scene": batch.scene,
    }
    if area is not None:
        override["area"] = area
    if region is not None:
        override["region"] = region
    with bpy.context.temp_override(**override):
        show_render_window()
        result = bpy.ops.render.render(
            "INVOKE_DEFAULT",
            write_still=True,
            scene=batch.scene.name,
        )
    if "FINISHED" in result:
        batch.render_started = True
        batch.render_complete = True
    elif "CANCELLED" in result:
        batch.render_cancelled = True
    elif "RUNNING_MODAL" not in result:
        raise RuntimeError(
            f"Blender returned {sorted(result)} when starting render."
        )
    return True


# One 0-100 bar spans the whole build. Rendering owns the first 67% because
# it is by far the longest stage; camera sampling and dataset generation own
# the remaining 33%.
_SR_RENDER_SHARE = 67.0


def sr_stage_progress(local_fraction, stage=""):
    """A stage's own 0..1 progress as a percentage.

    Rendering and point-cloud generation are independent operations now, so
    each one owns its whole bar instead of sharing a combined one.
    """
    return max(0.0, min(1.0, float(local_fraction))) * 100.0


def _sr_update_render_batch_progress(batch):
    fraction = batch._i / max(1, batch._n)
    try:
        bpy.context.window_manager.progress_update(sr_stage_progress(fraction))
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    batch.cfg.render_progress = batch._i
    batch.cfg.render_progress_fac = fraction
    recent = batch.completed_durations[-8:]
    average = sum(recent) / max(1, len(recent))
    remaining = average * max(0, batch._n - batch._i)
    batch.cfg.render_eta = _fmt_eta(remaining)
    batch.cfg.render_status = (
        f"Rendering {batch._i}/{batch._n} images · "
        f"{max(0, batch._n - batch._i)} remaining"
    )
    progress.update(
        fraction=fraction,
        message=batch.cfg.render_status,
        detail=f"Camera {batch._i} of {batch._n}",
        eta=batch.cfg.render_eta,
    )
    _tag_redraw_sceneray_splat(bpy.context)


def _sr_schedule_dataset_continuation(batch):
    state = _render_dataset_continuation
    state["scene_pointer"] = batch.scene_pointer
    state["window_pointer"] = batch.window_pointer
    if state["timer_registered"]:
        return
    state["timer_registered"] = True
    try:
        bpy.app.timers.register(
            _sr_run_dataset_continuation,
            first_interval=0.1,
        )
    except (ReferenceError, RuntimeError, ValueError):
        state["timer_registered"] = False


def _sr_run_dataset_continuation():
    state = _render_dataset_continuation
    state["timer_registered"] = False
    scene = None
    for candidate in bpy.data.scenes:
        try:
            if candidate.as_pointer() == state["scene_pointer"]:
                scene = candidate
                break
        except ReferenceError:
            continue
    if scene is None:
        return None
    window, area, region = _sr_live_window_context(
        state["window_pointer"]
    )
    override = {"scene": scene}
    if window is not None:
        override.update({"window": window, "screen": window.screen})
    if area is not None:
        override["area"] = area
    if region is not None:
        override["region"] = region
    try:
        with bpy.context.temp_override(**override):
            _sr_continue_dataset_generation(bpy.context)
    except Exception as exc:
        # Anything that escapes here used to be printed and forgotten, which
        # left the progress bar sitting at the fraction the metadata step had
        # just set and never moving again - the build looked frozen with no
        # explanation. Close the progress out and say what happened.
        import traceback

        traceback.print_exc()
        cfg = getattr(scene, "SCENERAY_SPLAT", None)
        if cfg is not None:
            cfg.point_status = (
                f"Dataset generation could not continue: {exc}"
            )
            cfg.render_status = cfg.point_status
            cfg.build_workflow = 'NONE'
            cfg.write_metadata_after_render = False
        progress.fail(f"Dataset generation stopped: {exc}")
        print(f"[sceneray_splat] dataset continuation failed: {exc}")
    return None


def _sr_finish_render_batch(batch, outcome, error=""):
    if batch.finished:
        return
    batch.finished = True
    batch.phase = "FINISHED"
    continuation = bool(batch.cfg.write_metadata_after_render)
    batch.cfg.write_metadata_after_render = False
    try:
        ok = batch._finish()
    except Exception as exc:
        ok = False
        error = error or str(exc)
    try:
        bpy.context.window_manager.progress_end()
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    batch.cfg.is_rendering = False
    batch.cfg.render_eta = ""
    _cancel_flag["requested"] = False
    _render_batch_runtime["batch"] = None

    all_rendered = (
        ok
        and not batch.failed
        and all(
            not record.get("pending", True)
            for record in batch.records
        )
    )
    if outcome == "SUCCESS" and ok:
        batch.cfg.render_progress = batch._n
        batch.cfg.render_progress_fac = 1.0
        batch.cfg.render_status = (
            f"Render finished — {batch._summary()}"
        )
        if continuation and all_rendered:
            # Progress stays open: the metadata step, and for Build Dataset
            # the point cloud, are still to come.
            _sr_schedule_dataset_continuation(batch)
        else:
            if continuation:
                batch.cfg.point_status = (
                    "Build stopped because one or more cameras remain Pending."
                )
            batch.cfg.build_workflow = 'NONE'
            progress.end(batch.cfg.render_status)
    elif outcome == "CANCELLED":
        batch.cfg.render_status = (
            f"Cancelled at {batch._i}/{batch._n} — completed images were "
            "saved; run Render Images again to resume."
        )
        batch.cfg.build_workflow = 'NONE'
        progress.end(batch.cfg.render_status)
    else:
        batch.cfg.render_status = (
            f"Render stopped at {batch._i}/{batch._n}: "
            f"{error or 'unknown render error'}"
        )
        batch.cfg.build_workflow = 'NONE'
        progress.end(batch.cfg.render_status)
    _tag_redraw_sceneray_splat(bpy.context)


def _sr_abort_render_batch(reason="Render batch stopped."):
    batch = _render_batch_runtime.get("batch")
    if batch is None:
        return
    try:
        raw_hooks.batch_restore(batch)
    except Exception:
        pass
    try:
        restore_render_output_state(
            batch.scene,
            getattr(batch, "_output_state", None),
        )
        restore_render_display()
        if getattr(batch, "_orig_camera", None) is not None:
            batch.scene.camera = batch._orig_camera
        batch.cfg.is_rendering = False
        batch.cfg.render_status = reason
        batch.cfg.write_metadata_after_render = False
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    try:
        bpy.context.window_manager.progress_end()
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    batch.finished = True
    _render_batch_runtime["batch"] = None
    _render_batch_runtime["timer_registered"] = False
    _cancel_flag["requested"] = False


def _sr_render_batch_tick():
    batch = _render_batch_runtime["batch"]
    if batch is None or batch.finished:
        _render_batch_runtime["timer_registered"] = False
        return None
    try:
        if batch.phase == "READY":
            if _cancel_flag["requested"]:
                _sr_finish_render_batch(batch, "CANCELLED")
                _render_batch_runtime["timer_registered"] = False
                return None
            if batch._i >= batch._n:
                _sr_finish_render_batch(batch, "SUCCESS")
                _render_batch_runtime["timer_registered"] = False
                return None
            if not _sr_launch_batch_render(batch):
                return 0.2
            return 0.1

        if batch.phase != "WAIT_RENDER":
            return 0.1
        if _sr_render_job_running():
            batch.render_started = True
            batch.render_idle_since = None
            settle_render_window()
            now = time.time()
            if now - getattr(batch, "_last_ui_refresh", 0.0) >= 0.5:
                batch._last_ui_refresh = now
                recent = batch.completed_durations[-8:]
                if recent:
                    average = sum(recent) / len(recent)
                    active_elapsed = max(0.0, now - batch.render_launch_time)
                    batch.cfg.render_eta = _fmt_eta(
                        max(0.0, average - active_elapsed)
                        + average * max(0, batch._n - batch._i - 1)
                    )
                # The Render Result window can appear only after the render
                # starts. This refresh also opens its Blender Render sidebar.
                _tag_redraw_sceneray_splat(bpy.context)
            return 0.1

        # render_cancel/render_complete can be delivered at the edge of the
        # compositor job. Require a brief, continuous idle period before node
        # capture is restored or another camera mutates render state.
        now = time.time()
        if batch.render_idle_since is None:
            batch.render_idle_since = now
            return 0.1
        if now - batch.render_idle_since < 0.35:
            return 0.1

        # Blender's compositor and render job are now fully idle. Suppress
        # additional render displays before the next camera can launch.
        settle_render_window(force=True)
        elapsed = time.time() - batch.render_launch_time
        if not batch.render_complete and not batch.render_cancelled:
            if batch.output_changed():
                batch.render_complete = True
            elif batch.render_started or elapsed > 1.5:
                batch.render_cancelled = True
            else:
                return 0.1

        qi = batch.active_qi
        prepared = batch.active_prepared
        camera = batch.queue[qi][0]
        batch.active_qi = None
        batch.active_prepared = None
        if batch.render_cancelled:
            batch._record_failure(camera, "Render cancelled")
            _sr_finish_render_batch(batch, "CANCELLED")
            _render_batch_runtime["timer_registered"] = False
            return None
        try:
            batch._finalize_render_one(qi, prepared, {"FINISHED"})
        except SceneRaySplatStorageError as exc:
            _sr_finish_render_batch(batch, "ERROR", str(exc))
            _render_batch_runtime["timer_registered"] = False
            return None
        except Exception as exc:
            if not batch._record_failure(camera, exc):
                _sr_finish_render_batch(
                    batch,
                    "ERROR",
                    batch._manifest_error,
                )
                _render_batch_runtime["timer_registered"] = False
                return None
            batch.failed.append((camera.name, str(exc)))
            print(
                f"[sceneray_splat] FAILED camera '{camera.name}': "
                f"{exc} — continuing"
            )

        batch.completed_durations.append(
            max(0.0, time.time() - batch.render_launch_time)
        )
        batch._i += 1
        _sr_update_render_batch_progress(batch)
        batch.phase = "READY"
        if _cancel_flag["requested"]:
            _sr_finish_render_batch(batch, "CANCELLED")
            _render_batch_runtime["timer_registered"] = False
            return None
        return 0.05
    except SceneRaySplatStorageError as exc:
        _sr_finish_render_batch(batch, "ERROR", str(exc))
    except Exception as exc:
        _sr_finish_render_batch(batch, "ERROR", str(exc))
    _render_batch_runtime["timer_registered"] = False
    return None


def _sr_begin_render_batch(context, reporter):
    """Start the 4.2-style global render controller and return immediately."""
    if _render_batch_runtime["batch"] is not None:
        reporter.report(
            {"WARNING"},
            "A SplatGen render batch is already running.",
        )
        return {"CANCELLED"}
    batch = _SceneRaySplatRenderBatch(reporter)
    try:
        batch._setup(context)
    except (ValueError, SceneRaySplatStorageError) as exc:
        reporter.report({"ERROR"}, str(exc))
        return {"CANCELLED"}
    batch._n = len(batch.queue)
    batch.scene_pointer = batch.scene.as_pointer()
    batch.complete_workflow = bool(batch.cfg.write_metadata_after_render)
    workflow = str(getattr(batch.cfg, "build_workflow", "RENDER")) or "RENDER"
    if workflow == 'NONE':
        workflow = 'RENDER'
    # Only the stages this workflow will actually run are announced, so an
    # idle stage never appears in the status bar. Build Dataset has already
    # announced both and finished the first, so its report continues here
    # instead of restarting at zero.
    if not progress.is_active():
        progress.begin(
            BUILD_WORKFLOW_LABELS.get(workflow, "Render Images"),
            _sr_workflow_stages(
                batch.scene, workflow, (PHASE_RENDER, PHASE_FILES)
            ),
        )
    progress.set_stage(PHASE_RENDER)
    progress.update(message=f"Preparing {batch._n} image(s)")
    try:
        batch.window_pointer = context.window.as_pointer()
    except (AttributeError, ReferenceError):
        batch.window_pointer = None

    if batch._n == 0:
        if not batch._finish():
            batch.cfg.write_metadata_after_render = False
            return {"CANCELLED"}
        batch.cfg.render_progress = 0
        batch.cfg.render_total = 0
        batch.cfg.render_progress_fac = 1.0
        batch.cfg.render_status = (
            f"All {len(batch.reused)} managed camera(s) are rendered."
        )
        if batch.complete_workflow:
            batch.cfg.write_metadata_after_render = False
            _sr_schedule_dataset_continuation(batch)
        reporter.report({"INFO"}, batch.cfg.render_status)
        return {"FINISHED"}

    batch.cfg.is_rendering = True
    if batch.cfg.build_started_at <= 0.0:
        batch.cfg.build_started_at = time.time()
    batch.cfg.render_progress = 0
    batch.cfg.render_total = batch._n
    batch.cfg.render_progress_fac = 0.0
    batch.cfg.render_eta = ""
    batch.cfg.render_status = (
        f"Preparing image 1/{batch._n}; {batch._n} image(s) remaining"
    )
    _cancel_flag["requested"] = False
    try:
        context.window_manager.progress_begin(0, 100)
        context.window_manager.progress_update(0)
    except (AttributeError, ReferenceError, RuntimeError):
        pass
    _render_batch_runtime["batch"] = batch
    batch._reporter = None
    _sr_schedule_render_batch_tick(0.01)
    _tag_redraw_sceneray_splat(context)
    reporter.report(
        {"INFO"},
        f"Started responsive render batch for "
        f"{batch._n} pending camera(s).",
    )
    return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════
# 10. STEP 2 — WRITE THE CAMERA DATASET  (this is where M is decided)
# ═══════════════════════════════════════════════════════════════════════


def build_frames(context, cfg, manifest, report=None):
    """Turn the render manifest into export-frame frames.

    This is the ONLY place the world→export transform is computed, and the
    only place a camera pose is converted. It returns (frames, resolution,
    M, notes) where every frame carries both its world-frame pose and its
    export-frame pose, derived from each other by M and nothing else.
    """
    resolution = tuple(manifest["resolution"])
    w, h = resolution
    output_dir = _sr_effective_output_path(cfg)
    if output_dir is None:
        raise ValueError("Choose an output folder first.")

    class _RenderStub:
        resolution_x, resolution_y = w, h
        resolution_percentage = 100
        pixel_aspect_x = context.scene.render.pixel_aspect_x
        pixel_aspect_y = context.scene.render.pixel_aspect_y

    notes = {"missing_camera": [], "missing_image": [],
             "pending_image": [], "scaled": []}
    world_poses, staging = [], []
    for entry in manifest["cameras"]:
        if "pending" in entry:
            pending = bool(entry["pending"])
        elif "status" in entry:
            pending = str(entry["status"]).upper() != "RENDERED"
        else:
            pending = False
        if pending:
            notes["pending_image"].append(entry.get("name", "Camera"))
            continue
        cam_obj = bpy.data.objects.get(entry["name"])
        if cam_obj is None or cam_obj.type != 'CAMERA':
            notes["missing_camera"].append(entry["name"])
            continue
        img_path = output_dir / entry["file_path"].lstrip("./")
        if not img_path.is_file():
            notes["missing_image"].append(entry["file_path"])
            continue
        training_file_path = str(
            entry.get("training_file_path", entry["file_path"])
        )
        training_img_path = output_dir / training_file_path.lstrip("./")
        if not training_img_path.is_file():
            notes["missing_image"].append(training_file_path)
            continue
        c2w_world, had_scale = camera_pose_world(cam_obj)
        if had_scale:
            notes["scaled"].append(entry["name"])
        fx, fy, cx, cy, angle = compute_intrinsics(cam_obj.data, _RenderStub)
        world_poses.append(c2w_world)
        staging.append((entry, c2w_world, (fx, fy, cx, cy, angle)))

    if not staging:
        return [], resolution, Matrix.Identity(4), notes

    # ── THE decision point. One matrix, from here on used verbatim. ──
    # Datasets are always exported in raw Blender world coordinates, so the
    # result lines up with the scene it came from.
    M = export_transform_from_cameras(world_poses, False)

    frames = []
    for entry, c2w_world, (fx, fy, cx, cy, angle) in staging:
        c2w_export = apply_export_transform_to_pose(M, c2w_world)
        frame = {
            "file_path": entry["file_path"],
            "training_file_path": str(
                entry.get("training_file_path", entry["file_path"])
            ),
            "camera_name": entry["name"],
            "frame_index": int(entry.get("frame_index", len(frames))),
            "transform_matrix": matrix_to_list(c2w_export),   # EXPORT frame
            "world_matrix": matrix_to_list(c2w_world),        # Blender world
            "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
            "camera_angle_x": angle,
        }
        signature = bd_manifest.signature_of(entry)
        if signature is not None:
            frame["signature"] = signature
        frames.append(frame)
    return frames, resolution, M, notes


class SCENERAY_SPLAT_OT_calculate_cameras(bpy.types.Operator):
    """STEP 2 — Write Dataset(Default) cameras.txt and images.txt from the
    cameras' exact positions, rotations and lens settings.

    This step decides the dataset's coordinate frame — a single 4x4 matrix
    from Blender world space to the export frame (the identity unless
    Center Splat at Origin is on) — and stores it in the dataset manifest.
    Step 3 reads that same matrix back and applies it to the point cloud,
    which is what guarantees the two line up.

    Renders nothing and modifies no camera."""
    bl_idname = "sceneray_splat.calculate_cameras"
    bl_label = "2. Write Camera Data"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        if cfg is None or cfg.is_rendering or getattr(cfg, "is_generating_points", False):
            return False
        return (_sr_output_base_path(cfg) is not None)

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        output_dir = _sr_effective_output_path(cfg)
        if output_dir is None:
            self.report({'ERROR'}, "Choose an output folder first.")
            return {'CANCELLED'}
        manifest = (read_render_manifest(output_dir)
                    or read_completed_camera_data(output_dir))
        if manifest is None:
            self.report({'ERROR'}, "No render manifest — run step 1 first.")
            return {'CANCELLED'}

        frames, resolution, M, notes = build_frames(context, cfg, manifest)
        for name in notes["missing_camera"]:
            self.report({'WARNING'}, f"Camera '{name}' is no longer in the "
                        "scene — skipped.")
        for p in notes["missing_image"]:
            self.report({'WARNING'}, f"Missing image {p} — skipped.")
        if notes["pending_image"]:
            self.report({"WARNING"},
                        f"{len(notes['pending_image'])} camera(s) are still "
                        "Pending and were excluded from camera data.")
        if notes["scaled"]:
            self.report({'WARNING'},
                        f"{len(notes['scaled'])} camera(s) carry object scale; "
                        "the pose was exported with scale stripped, which is "
                        f"what the renderer does: {', '.join(notes['scaled'][:5])}"
                        f"{'…' if len(notes['scaled']) > 5 else ''}")
        if not frames:
            self.report({'ERROR'}, "No valid camera/image pairs — were the "
                        "cameras renamed or deleted since rendering?")
            return {'CANCELLED'}

        print("[sceneray_splat] export frame = Blender world (identity transform)")

        # Build and verify the complete camera contract away from the active
        # dataset. If anything fails, the previous camera files remain intact.
        # Short by design: this sits under an already-deep project path, and
        # Windows still refuses to create paths past its limit.
        staging_root = _work_dir_path(output_dir) / f"pub_{uuid.uuid4().hex[:8]}"
        color_management = _sr_color_management_snapshot(context.scene, cfg)
        try:
            cameras_path, images_path = write_colmap_cameras_images(
                staging_root,
                frames,
                resolution,
                color_management=color_management,
            )
            write_dataset_manifest(
                staging_root,
                frames,
                resolution,
                M,
                color_management=color_management,
            )

            # Self-check the staged images.txt before it can become active.
            centres = camera_centres_from_images_txt(staging_root)
            worst, worst_name = 0.0, ""
            for fr in frames:
                C = centres.get(
                    os.path.basename(
                        fr.get("training_file_path", fr["file_path"])
                    )
                )
                if C is None:
                    continue
                want = list_to_matrix(
                    fr["transform_matrix"]
                ).to_translation()
                d = (Vector(C) - want).length
                if d > worst:
                    worst, worst_name = d, fr["camera_name"]
            if worst > 1e-3:
                raise ValueError(
                    "Internal check FAILED: images.txt round-trips camera "
                    f"'{worst_name}' to a position {worst:.4f} units off. "
                    "Do not train on this dataset — please report it."
                )

            # The export frame is always Blender world space, so a point cloud
            # built before the render is still valid for it and is kept.
            dataset_root = bd_paths.standard_dataset_dir(output_dir)
            dataset_root.mkdir(parents=True, exist_ok=True)
            cameras_path.replace(_sr_dataset_file(output_dir, "cameras.txt"))
            images_path.replace(_sr_dataset_file(output_dir, "images.txt"))
            active_manifest = _dataset_manifest_path(output_dir)
            active_manifest.parent.mkdir(parents=True, exist_ok=True)
            _dataset_manifest_path(staging_root).replace(active_manifest)
        except (OSError, TypeError, ValueError) as exc:
            self.report(
                {"ERROR"},
                f"Camera data was not published safely: {exc}",
            )
            return {"CANCELLED"}
        finally:
            try:
                shutil.rmtree(staging_root)
            except OSError:
                pass

        print(f"[sceneray_splat] images.txt round-trip check OK "
              f"(max camera error {worst:.2e} units)")

        try:
            export_root = _sr_finalize_dataset_export(cfg, output_dir)
            if export_root is not None:
                self.report({'INFO'}, "Validated canonical dataset output.")
        except bd_export.GroundTruthValidationError as exc:
            self.report({'WARNING'}, str(exc))
        except Exception as exc:
            self.report({'ERROR'}, f"Dataset export failed validation: {exc}")
            return {'CANCELLED'}

        self.report({'INFO'},
                    f"Wrote cameras.txt + images.txt "
                    f"({len(frames)} camera(s)) — next: step 3.")
        return {'FINISHED'}




class SceneRaySplatQueueItem(PropertyGroup):
    camera: PointerProperty(type=bpy.types.Object)
    camera_name: StringProperty(
        name="Camera Name", options={'HIDDEN'},
        description="Last known name, retained when the object is deleted")
    render_state: EnumProperty(
        name="Render State",
        items=[
            ('PENDING', "Pending", "Needs to be rendered"),
            ('RENDERED', "Rendered", "Rendered image is current"),
        ],
        default='PENDING')


class SceneRaySplatRigCamera(PropertyGroup):
    """One camera inside a saved rig.

    The full object transform is stored — location, rotation AND scale,
    including any custom scale the user applied — plus the lens, sensor and
    viewport display size, so re-loading a rig reproduces it exactly.
    """
    source_name: StringProperty(name="Camera Name")
    location: FloatVectorProperty(size=3, subtype='TRANSLATION')
    rotation: FloatVectorProperty(size=4, subtype='QUATERNION',
                                  default=(1.0, 0.0, 0.0, 0.0))
    scale: FloatVectorProperty(size=3, default=(1.0, 1.0, 1.0))
    lens: FloatProperty(default=35.0)
    lens_unit: StringProperty(default="MILLIMETERS")
    sensor_width: FloatProperty(default=36.0)
    sensor_height: FloatProperty(default=24.0)
    sensor_fit: StringProperty(default="AUTO")
    camera_type: StringProperty(default="PERSP")
    ortho_scale: FloatProperty(default=6.0, min=0.000001)
    shift_x: FloatProperty(default=0.0)
    shift_y: FloatProperty(default=0.0)
    clip_start: FloatProperty(default=0.1)
    clip_end: FloatProperty(default=1000.0)
    display_size: FloatProperty(default=0.5)
    world_matrix: FloatVectorProperty(
        size=16, default=(1.0, 0.0, 0.0, 0.0,
                          0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0), options={'HIDDEN'})
    parent_source_id: StringProperty(options={'HIDDEN'})
    source_relative_matrix: FloatVectorProperty(
        size=16, default=(1.0, 0.0, 0.0, 0.0,
                          0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0), options={'HIDDEN'})
    has_source_relative: BoolProperty(default=False, options={'HIDDEN'})


class SceneRaySplatRigPreset(PropertyGroup):
    name: StringProperty(name="Rig Name", default="Camera Rig")
    file_name: StringProperty(options={'HIDDEN'})
    builtin: BoolProperty(default=False, options={'HIDDEN'})
    description: StringProperty(options={'HIDDEN'})
    cameras: CollectionProperty(type=SceneRaySplatRigCamera)


# ── Point-cloud density ────────────────────────────────────────────────
#

# The public point-cloud workflow has three orthogonal controls: initial ray
# density, the camera budget selected before back-projection, and merge strength.
# Selection still assigns adaptive per-camera importance internally.


# ── Live UI state and settings ─────────────────────────────────────────

_sr_bulk = {"busy": False}
_ui_scene_cache = {"revisions": {}}


def _sr_scene_revision(scene):
    try:
        return _ui_scene_cache["revisions"].get(scene.as_pointer(), 0)
    except ReferenceError:
        return 0


def _sr_mark_scene_cache_dirty(scene):
    """Invalidate expensive UI analysis only after relevant scene changes."""
    if scene is None:
        return
    try:
        pointer = scene.as_pointer()
    except ReferenceError:
        return
    revisions = _ui_scene_cache["revisions"]
    revisions[pointer] = revisions.get(pointer, 0) + 1


@persistent
def _sr_depsgraph_cache_invalidate(scene, depsgraph):
    """Track geometry/camera changes without scanning the scene during draw."""
    relevant = False
    for update in depsgraph.updates:
        data = update.id
        if isinstance(data, bpy.types.Object):
            try:
                if (update.is_updated_transform
                        or data.type in _RENDER_GEOMETRY_TYPES
                        or data.type == 'CAMERA'):
                    relevant = True
                    break
            except ReferenceError:
                relevant = True
                break
        elif isinstance(data, (bpy.types.Camera, bpy.types.Collection,
                               bpy.types.Mesh, bpy.types.Curve,
                               bpy.types.MetaBall)):
            relevant = True
            break
    if relevant:
        _sr_mark_scene_cache_dirty(scene)


#: (rays per camera, camera usage %, merge strength) per preset.
_POINT_QUALITY_VALUES = {
    'LOW': (0.5, 30.0, 1.5),
    'MEDIUM': (1.0, 50.0, 1.0),
    'HIGH': (2.5, 100.0, 0.75),
}


def _sr_workspace_step_update(self, context):
    """Redraw the viewports: the coverage overlay shows only on Prepare."""
    for window in getattr(getattr(context, "window_manager", None), "windows", ()):
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _sr_point_quality_update(self, context):
    values = _POINT_QUALITY_VALUES.get(self.point_quality)
    if values is None or _sr_bulk["busy"]:
        return
    _sr_bulk["busy"] = True
    try:
        (self.point_sampling_density, self.point_camera_usage,
         self.point_merging_strength) = values
    finally:
        _sr_bulk["busy"] = False
    _sr_mark_scene_cache_dirty(getattr(self, "id_data", None))
    _tag_redraw_sceneray_splat(context)


def _sr_point_advanced_update(self, context):
    if _sr_bulk["busy"]:
        return
    if self.point_quality != 'CUSTOM':
        self.point_quality = 'CUSTOM'
    _sr_mark_scene_cache_dirty(getattr(self, "id_data", None))
    _tag_redraw_sceneray_splat(context)


SceneRaySplatProperties.__annotations__.update({
    "workspace_step": EnumProperty(name="Workflow", default='PREPARE', items=[
        ('PREPARE','Prepare','Create cameras and build the dataset')], update=_sr_workspace_step_update),
    "show_job_details": BoolProperty(name="Details", default=False),
    "dismissed_job": StringProperty(default="", options={'SKIP_SAVE'}),
    "camera_queue": CollectionProperty(type=SceneRaySplatQueueItem),
    "active_camera_index": IntProperty(default=0, min=0),
    "rig_presets": CollectionProperty(type=SceneRaySplatRigPreset),
    "active_preset_index": IntProperty(default=0, min=0),
    "point_sampling_density": FloatProperty(
        name="Rays per Camera", default=5.0, min=0.1, max=25.0,
        soft_min=0.25, soft_max=5.0, precision=2,
        description=(
            "Multiplier for rays cast from every selected camera; higher values "
            "capture more surface detail but take longer. Nothing is capped: "
            "SplatGen warns when the point cloud goes over 2,000,000 points"
        ),
        update=_sr_point_advanced_update),
    "point_camera_usage": FloatProperty(
        name="Cameras Used", default=100.0, min=1.0, max=100.0,
        subtype='PERCENTAGE', precision=0,
        description=(
            "Percentage of queued cameras selected by smart scene coverage before "
            "any point-cloud rays are cast"
        ),
        update=_sr_point_advanced_update),
    "point_merging_strength": FloatProperty(
        name="Sample Merging", default=1.5, min=0.1, max=4.0,
        soft_min=0.25, soft_max=2.0, precision=2,
        description=(
            "Size of the merge cell relative to each sample's footprint. 1.0 keeps "
            "about one point per sample footprint; higher values produce fewer, "
            "cleaner points, lower values keep more. Never adjusted automatically"
        ),
        update=_sr_point_advanced_update),
    "point_quality": EnumProperty(
        name="Quality",
        items=[
            ('LOW', "Low", "Fast preview-quality point cloud"),
            ('MEDIUM', "Medium", "Balanced speed and scene coverage"),
            ('HIGH', "High", "Dense sampling across every camera"),
            ('CUSTOM', "Custom", "Configure coverage, rays and merging by hand"),
        ],
        default='CUSTOM', update=_sr_point_quality_update),
    # Advanced point-cloud controls appear when the Custom preset is chosen,
    # so they no longer need a toggle of their own.
    "point_final_density": FloatProperty(
        name="Final Point Cloud Density", default=1.0, min=0.1, max=10.0,
        description="Legacy density value retained for compatibility",
        options={'HIDDEN'}),
    "write_metadata_after_render": BoolProperty(default=False,
                                              options={'SKIP_SAVE'}),
    "active_dataset_dir": StringProperty(
        name="Active Dataset Version",
        subtype='DIR_PATH',
        options={'HIDDEN'},
    ),
    "latest_dataset_dir": StringProperty(
        name="Latest Completed Dataset",
        subtype='DIR_PATH',
        options={'HIDDEN'},
    ),
    "review_active": BoolProperty(default=False, options={'SKIP_SAVE'},
                                   description="Keyboard camera review is active"),
    "is_generating_points": BoolProperty(default=False, options={'SKIP_SAVE'}),
    "point_progress": IntProperty(default=0, options={'SKIP_SAVE'}),
    "point_total": IntProperty(default=0, options={'SKIP_SAVE'}),
    "point_progress_fac": FloatProperty(default=0.0, min=0.0, max=1.0,
                                        subtype='FACTOR',
                                        options={'SKIP_SAVE'}),
    "point_eta": StringProperty(default="", options={'SKIP_SAVE'}),
    "point_status": StringProperty(default="", options={'SKIP_SAVE'}),
    # What the last generation produced. Saved with the file so the next
    # estimate for this scene is learned from a real result, not a guess.
    "point_last_count": IntProperty(default=0, min=0, options={'HIDDEN'}),
    "point_last_retention": FloatProperty(default=0.0, min=0.0,
                                          options={'HIDDEN'}),
    "point_last_merging": FloatProperty(default=0.0, min=0.0,
                                        options={'HIDDEN'}),
})


# ── Queue helpers ──────────────────────────────────────────────────────

def _sr_reset_transient_build_state(scene):
    """Clear process-only progress whenever Blender opens another file.

    A dataset process cannot survive File > New or loading a .blend, so any
    bars carried into the new scene are stale. Explicitly clearing every
    field also repairs startup files saved by older add-on versions.
    """
    cfg = getattr(scene, "SCENERAY_SPLAT", None)
    if cfg is None:
        return
    cfg.is_rendering = False
    cfg.render_progress = 0
    cfg.render_total = 0
    cfg.render_progress_fac = 0.0
    cfg.render_eta = ""
    cfg.render_status = ""
    cfg.build_started_at = 0.0
    cfg.is_generating_points = False
    cfg.point_progress = 0
    cfg.point_total = 0
    cfg.point_progress_fac = 0.0
    cfg.point_eta = ""
    cfg.point_status = ""
    cfg.write_metadata_after_render = False
    cfg.build_workflow = 'NONE'
    progress.reset()


@persistent
def _sr_reset_build_state_after_load(_unused):
    """Drop all previous-file progress and cache keys after New/Open."""
    _render_status_sync["pending"].clear()
    _render_status_sync["signatures"].clear()
    _render_status_sync["last_checks"].clear()
    _render_status_sync["timer"] = False
    # A build interrupted by opening another file must not leave the user's
    # Render Display preference switched off.
    restore_render_display()
    for scene in bpy.data.scenes:
        _sr_reset_transient_build_state(scene)
    # A newly opened file decides for itself whether the overlay is on; the
    # previous file's setting must not leak into it.
    from . import camera_overlay

    camera_overlay.sync(
        getattr(bpy.context.scene, "SCENERAY_SPLAT", None)
        if getattr(bpy.context, "scene", None) is not None else None
    )


def _sr_has_queued_camera(cfg):
    """Fast operator/UI availability check that stops at the first camera."""
    for item in cfg.camera_queue:
        camera = item.camera
        if camera is not None and camera.type == 'CAMERA':
            return True
    return False


def gather_scene_cameras(context, cfg):
    """The explicit render queue is authoritative."""
    cams, seen = [], set()
    for item in cfg.camera_queue:
        cam = item.camera
        if cam is None or cam.type != 'CAMERA' or cam.name in seen:
            continue
        seen.add(cam.name)
        cams.append(cam)
    return cams


def _sr_queue_contains(cfg, camera):
    return any(item.camera == camera for item in cfg.camera_queue)


def _sr_queue_add(cfg, camera, apply_settings=True):
    if (camera is None or camera.type != 'CAMERA'
            or _sr_queue_contains(cfg, camera)):
        return False
    item = cfg.camera_queue.add()
    item.camera = camera
    item.camera_name = camera.name
    item.render_state = 'PENDING'
    if apply_settings:
        _sr_apply_global_camera_settings(cfg, camera)
    # A parented camera inherits its parent's scale. Queueing it is the point
    # at which that starts to matter, so the lock goes on here too - which
    # also repairs cameras made before this existed.
    if camera.parent is not None:
        _sr_lock_camera_scale(camera)
    _sr_mark_scene_cache_dirty(getattr(cfg, "id_data", None))
    return True


#: Name of the constraint that keeps a parented camera the right size.
_SR_SCALE_LOCK = "SplatGen Keep Scale"


def _sr_lock_camera_scale(camera):
    """Let a camera move with its parent, but never be scaled by it.

    A camera created from a face is parented to the mesh, so scaling that mesh
    scales the camera too - it stretches in the viewport and its status box
    with it. A Limit Scale constraint pinned to 1 keeps the camera following
    the face's position and rotation while staying its own size.

    Blender ignores camera scale when rendering, so this changes nothing about
    the images or the exported poses; it is purely what the user sees.
    """
    if camera is None or camera.type != 'CAMERA':
        return None
    existing = camera.constraints.get(_SR_SCALE_LOCK)
    if existing is not None:
        return existing
    try:
        constraint = camera.constraints.new('LIMIT_SCALE')
    except (AttributeError, RuntimeError, TypeError):
        return None
    constraint.name = _SR_SCALE_LOCK
    for axis in ("x", "y", "z"):
        setattr(constraint, f"use_min_{axis}", True)
        setattr(constraint, f"use_max_{axis}", True)
        setattr(constraint, f"min_{axis}", 1.0)
        setattr(constraint, f"max_{axis}", 1.0)
    constraint.owner_space = 'WORLD'
    return constraint


def _sr_queue_item(cfg, camera):
    return next((item for item in cfg.camera_queue if item.camera == camera),
                None)


def _sr_live_camera_map(scene):
    """Resolve live cameras once instead of doing one scene lookup per item."""
    live = {}
    for obj in scene.objects:
        try:
            if obj.type == 'CAMERA':
                live[obj.as_pointer()] = obj.name
        except ReferenceError:
            continue
    return live


def _sr_camera_matches_live_map(camera, live):
    try:
        return (camera is not None and camera.type == 'CAMERA'
                and live.get(camera.as_pointer()) == camera.name)
    except (ReferenceError, RuntimeError):
        return False


def _sr_mark_cameras_pending(cfg, cameras):
    """Persist an explicit re-render request without deleting its old image.

    The manifest keeps the stable frame assignment while its pending marker
    prevents a matching old file from being treated as a completed render.
    """
    names = {camera.name for camera in cameras if camera is not None}
    if not names:
        return
    for item in cfg.camera_queue:
        if item.camera is not None and item.camera.name in names:
            item.render_state = 'PENDING'
    root = _sr_effective_output_path(cfg)
    if root is None:
        return
    manifest = read_render_manifest(root)
    if manifest is None:
        return
    changed = False
    for entry in manifest.get("cameras", []):
        if entry.get("name") in names and not entry.get("pending", False):
            entry["pending"] = True
            entry["status"] = "PENDING"
            changed = True
    if changed:
        write_render_manifest(root, manifest.get("cameras", []),
                              tuple(manifest.get("resolution", (1, 1))))

def _sr_prune_camera_from_manifests(cfg, names):
    """Remove deleted cameras from the persisted render and dataset queues."""
    if not names:
        return
    root = _sr_effective_output_path(cfg)
    if root is None:
        return
    manifest = read_render_manifest(root)
    if manifest is not None:
        manifest["cameras"] = [entry for entry in manifest.get("cameras", [])
                               if entry.get("name") not in names]
        write_render_manifest(root, manifest["cameras"],
                              tuple(manifest.get("resolution", (1, 1))))
    dataset = read_dataset_manifest(root)
    if dataset is not None:
        frames = [frame for frame in dataset.get("frames", [])
                  if frame.get("camera_name") not in names]
        dataset["frames"] = frames
        try:
            with open(_dataset_manifest_path(root), "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=1)
            resolution = tuple(dataset.get("resolution", (1, 1)))
            write_colmap_cameras_images(
                str(root),
                frames,
                resolution,
                color_management=dataset.get("color_management"),
            )
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# 13. ADD CAMERA FROM VIEW  (hotkey-driven capture)
# ═══════════════════════════════════════════════════════════════════════


def _sceneray_splat_region_3d(context):
    """The RegionView3D the user is actually looking through, plus its
    space. Works from the sidebar, from the viewport, from a hotkey and
    from quad view."""
    rv3d = getattr(context, "region_data", None)
    space = getattr(context, "space_data", None)
    if isinstance(rv3d, bpy.types.RegionView3D):
        return rv3d, space
    if space is not None and getattr(space, "type", "") == 'VIEW_3D':
        rv3d = getattr(space, "region_3d", None)
        if rv3d is not None:
            return rv3d, space
    for window in context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            rv3d = getattr(space, "region_3d", None)
            if rv3d is not None:
                return rv3d, space
    return None, None


def _sceneray_splat_next_camera_name():
    i = 1
    while f"{_MANAGED_CAMERA_PREFIX}_{i:04d}" in bpy.data.objects:
        i += 1
    return f"{_MANAGED_CAMERA_PREFIX}_{i:04d}"



class SCENERAY_SPLAT_OT_add_camera_from_view(bpy.types.Operator):
    """Create a camera exactly where you are looking from and add it to the
    render queue.

    Position, rotation and viewing direction are taken from the 3D Viewport,
    so you can navigate and tap the shortcut to drop viewpoint after
    viewpoint. Assign your own key in Preferences ▸ Keymap ▸ 3D View
    (default: Shift F)"""
    bl_idname = "sceneray_splat.add_camera_from_view"
    bl_label = "Add Camera from View"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and not cfg.is_rendering
                and not cfg.is_generating_points)

    def execute(self, context):
        scene = context.scene
        cfg = scene.SCENERAY_SPLAT
        rv3d, space = _sceneray_splat_region_3d(context)
        if rv3d is None:
            self.report({'ERROR'}, "No 3D Viewport to take a view from.")
            return {'CANCELLED'}

        name = _sceneray_splat_next_camera_name()
        data = bpy.data.cameras.new(name)
        cam = bpy.data.objects.new(name, data)
        (context.collection or scene.collection).objects.link(cam)

        _sr_apply_global_camera_settings(cfg, cam)

        # The inverse view matrix IS the viewer's camera-to-world transform,
        # so this reproduces the view exactly — position, orientation and
        # viewing direction in one assignment.
        cam.matrix_world = rv3d.view_matrix.inverted()

        _sr_queue_add(cfg, cam)
        cfg.active_camera_index = len(cfg.camera_queue) - 1
        for obj in context.selected_objects:
            obj.select_set(False)
        cam.select_set(True)
        context.view_layer.objects.active = cam
        scene.camera = cam
        _tag_redraw_sceneray_splat(context)

        if not rv3d.is_perspective and rv3d.view_perspective != 'CAMERA':
            self.report({'WARNING'},
                        f"{name} placed at the view, but as a PERSPECTIVE "
                        "camera: COLMAP's pinhole model cannot describe an "
                        "orthographic view, so an ortho camera could not be "
                        "trained from.")
        else:
            self.report({'INFO'},
                        f"{name} created from the current view "
                        f"({len(cfg.camera_queue)} queued).")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════
class SCENERAY_SPLAT_OT_create_cameras_from_faces(bpy.types.Operator):
    """Create and queue one outward-facing camera for every selected mesh face"""
    bl_idname = "sceneray_splat.create_cameras_from_faces"
    bl_label = "Create Cameras From Faces"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        obj = context.active_object
        return (cfg is not None and obj is not None and obj.type == 'MESH'
                and not cfg.is_rendering and not cfg.is_generating_points)

    def execute(self, context):
        scene, cfg, source = (context.scene, context.scene.SCENERAY_SPLAT,
                              context.active_object)
        if not source.data.polygons:
            self.report({'WARNING'}, "The selected mesh has no faces.")
            return {'CANCELLED'}
        target = context.collection or scene.collection
        normal_matrix = source.matrix_world.to_3x3().inverted().transposed()
        _sr_hide_rig_source_from_render(source)
        source.display_type = 'WIRE'
        _sr_bulk["busy"] = True
        created = 0
        try:
            for face in source.data.polygons:
                name = _sceneray_splat_next_camera_name()
                data = bpy.data.cameras.new(name)
                camera = bpy.data.objects.new(name, data)
                target.objects.link(camera)
                _sr_apply_global_camera_settings(cfg, camera)
                location = source.matrix_world @ face.center
                normal = (normal_matrix @ face.normal).normalized()
                rotation = normal.to_track_quat('-Z', 'Y')
                world_matrix = rotation.to_matrix().to_4x4()
                world_matrix.translation = location
                camera.parent = source
                camera.matrix_parent_inverse = source.matrix_world.inverted()
                camera.matrix_world = world_matrix
                # Follow the face, but never be squashed by it. Parenting
                # hands down the source object's scale, so resizing the mesh
                # would otherwise stretch every camera with it.
                _sr_lock_camera_scale(camera)
                _sr_queue_add(cfg, camera)
                created += 1
        finally:
            _sr_bulk["busy"] = False
        cfg.active_camera_index = max(0, len(cfg.camera_queue) - 1)
        self.report({'INFO'}, f"Created {created} outward-facing camera(s). "
                    "The source mesh is hidden from renders and shown as wireframe.")
        return {'FINISHED'}

# 14. QUEUE OPERATORS
# ═══════════════════════════════════════════════════════════════════════


class SCENERAY_SPLAT_OT_queue_add_selected(bpy.types.Operator):
    """Add the selected Camera objects to the render queue"""
    bl_idname = "sceneray_splat.queue_add_selected"
    bl_label = "Add Selected Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        cameras = [o for o in context.selected_objects if o.type == 'CAMERA']
        active = context.active_object
        if not cameras and active is not None and active.type == 'CAMERA':
            cameras = [active]
        _sr_bulk["busy"] = True
        added = reset = 0
        reset_cameras = []
        try:
            for camera in cameras:
                if _sr_queue_add(cfg, camera):
                    added += 1
                else:
                    item = _sr_queue_item(cfg, camera)
                    if item is not None:
                        reset_cameras.append(camera)
                        reset += 1
        finally:
            _sr_bulk["busy"] = False
        _sr_mark_cameras_pending(cfg, reset_cameras)
        if not added and not reset:
            self.report({'WARNING'}, "Select one or more Camera objects first.")
            return {'CANCELLED'}
        cfg.active_camera_index = len(cfg.camera_queue) - 1
        message = f"Queued {added} camera(s)"
        if reset:
            message += f"; marked {reset} existing camera(s) pending"
        self.report({'INFO'}, message + ".")
        return {'FINISHED'}



class SCENERAY_SPLAT_OT_queue_add_all(bpy.types.Operator):
    """Add every Camera object in the scene to the render queue"""
    bl_idname = "sceneray_splat.queue_add_all"
    bl_label = "Add All Scene Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        _sr_bulk["busy"] = True
        try:
            added = sum(1 for cam in context.scene.objects
                        if cam.type == 'CAMERA' and _sr_queue_add(cfg, cam))
        finally:
            _sr_bulk["busy"] = False
        if added:
            cfg.active_camera_index = len(cfg.camera_queue) - 1
        self.report({'INFO'}, f"Queued {added} camera(s).")
        return {'FINISHED'}


def _sr_activate_queue_index(context, index):
    cfg = context.scene.SCENERAY_SPLAT
    if not cfg.camera_queue:
        return False
    index %= len(cfg.camera_queue)
    camera = cfg.camera_queue[index].camera
    if camera is None or camera.type != 'CAMERA':
        return False
    # A queued camera may live in a collection excluded from the current
    # View Layer. It is still a valid render/review camera, but Blender
    # forbids selecting it here; review must never abort for that reason.
    view_layer_camera = context.view_layer.objects.get(camera.name)
    if view_layer_camera == camera:
        try:
            for obj in context.selected_objects:
                obj.select_set(False)
            camera.select_set(True)
            context.view_layer.objects.active = camera
        except RuntimeError:
            cfg.render_status = (f"Reviewing '{camera.name}' (it cannot be "
                                 "selected in the active View Layer).")
    else:
        cfg.render_status = (f"Reviewing '{camera.name}' (its collection is "
                             "excluded from the active View Layer).")
    cfg.active_camera_index = index
    _show_camera_in_viewports(context, camera)
    return True


def _sr_end_camera_review(context):
    cfg = context.scene.SCENERAY_SPLAT
    cfg.review_active = False
    for window in context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            if space.type == 'VIEW_3D':
                space.region_3d.view_perspective = 'PERSP'
            area.tag_redraw()
    _tag_redraw_sceneray_splat(context)


class SCENERAY_SPLAT_OT_review_cameras(bpy.types.Operator):
    """Enter or exit persistent keyboard camera review"""
    bl_idname = "sceneray_splat.review_cameras"
    bl_label = "Review Cameras"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and not cfg.is_rendering
                and not cfg.is_generating_points)

    def _start(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        if not _sr_activate_queue_index(context, cfg.active_camera_index):
            self.report({'WARNING'}, "Queue a camera before starting review.")
            return False
        cfg.review_active = True
        _tag_redraw_sceneray_splat(context)
        self.report({'INFO'}, "Camera review active: Left/Right to navigate, Backspace to delete, Esc to exit.")
        return True

    def invoke(self, context, event):
        cfg = context.scene.SCENERAY_SPLAT
        if cfg.review_active:
            _sr_end_camera_review(context)
            self.report({'INFO'}, "Camera review ended.")
            return {'FINISHED'}
        if not self._start(context):
            return {'CANCELLED'}
        # A modal session receives the arrows anywhere in Blender's window,
        # so review does not depend on keymap priority or mouse focus.
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        return self.invoke(context, None)

    def _still_in_camera_view(self, context):
        """False once every viewport has navigated away from the camera.

        Scans the same viewports _show_camera_in_viewports writes to, so a
        second window still showing the camera keeps review alive.
        """
        for window in context.window_manager.windows:
            screen = window.screen
            if screen is None:
                continue
            for area in screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for space in area.spaces:
                    if space.type != 'VIEW_3D':
                        continue
                    region = getattr(space, "region_3d", None)
                    if (region is not None
                            and region.view_perspective == 'CAMERA'):
                        return True
        return False

    def modal(self, context, event):
        cfg = context.scene.SCENERAY_SPLAT
        if not cfg.review_active:
            return {'FINISHED'}
        # Leaving camera view is itself the decision to stop reviewing, so
        # review ends with it rather than lingering invisibly.
        try:
            if not self._still_in_camera_view(context):
                _sr_end_camera_review(context)
                self.report({'INFO'}, "Camera review ended: left camera view.")
                return {'FINISHED'}
        except (AttributeError, ReferenceError, RuntimeError):
            pass
        if event.value == 'PRESS':
            if event.type == 'LEFT_ARROW':
                _sr_activate_queue_index(context, cfg.active_camera_index - 1)
                return {'RUNNING_MODAL'}
            if event.type == 'RIGHT_ARROW':
                _sr_activate_queue_index(context, cfg.active_camera_index + 1)
                return {'RUNNING_MODAL'}
            if event.type in {'BACK_SPACE', 'DEL'}:
                bpy.ops.sceneray_splat.delete_active_camera('EXEC_DEFAULT')
                return {'RUNNING_MODAL'}
            if event.type == 'ESC':
                _sr_end_camera_review(context)
                self.report({'INFO'}, "Camera review ended.")
                return {'FINISHED'}
        return {'PASS_THROUGH'}


class SCENERAY_SPLAT_OT_previous_camera(bpy.types.Operator):
    """Review the previous managed camera in the viewport"""
    bl_idname = "sceneray_splat.previous_camera"
    bl_label = "Previous Camera"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and cfg.review_active
                and not cfg.is_rendering and not cfg.is_generating_points)

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        if not _sr_activate_queue_index(context, cfg.active_camera_index - 1):
            self.report({'WARNING'}, "No managed camera to review.")
            return {'CANCELLED'}
        return {'FINISHED'}


class SCENERAY_SPLAT_OT_next_camera(bpy.types.Operator):
    """Review the next managed camera in the viewport"""
    bl_idname = "sceneray_splat.next_camera"
    bl_label = "Next Camera"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and cfg.review_active
                and not cfg.is_rendering and not cfg.is_generating_points)

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        if not _sr_activate_queue_index(context, cfg.active_camera_index + 1):
            self.report({'WARNING'}, "No managed camera to review.")
            return {'CANCELLED'}
        return {'FINISHED'}


class SCENERAY_SPLAT_OT_delete_active_camera(bpy.types.Operator):
    """Delete the camera currently being reviewed"""
    bl_idname = "sceneray_splat.delete_active_camera"
    bl_label = "Delete Current Camera"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and cfg.review_active
                and not cfg.is_rendering and not cfg.is_generating_points)

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        if cfg.active_camera_index >= len(cfg.camera_queue):
            return {'CANCELLED'}
        index = cfg.active_camera_index
        camera = cfg.camera_queue[index].camera
        if camera is None or camera.type != 'CAMERA':
            cfg.camera_queue.remove(index)
            _sr_mark_scene_cache_dirty(context.scene)
            return {'FINISHED'}
        name = camera.name
        _sr_bulk["busy"] = True
        try:
            cfg.camera_queue.remove(index)
            _sr_remove_objects([camera])
            cfg.active_camera_index = min(index,
                                          max(0, len(cfg.camera_queue) - 1))
        finally:
            _sr_bulk["busy"] = False
        _sr_prune_camera_from_manifests(cfg, {name})
        _sr_mark_scene_cache_dirty(context.scene)
        if cfg.camera_queue:
            _sr_activate_queue_index(context, cfg.active_camera_index)
        self.report({'INFO'}, f"Deleted '{name}'.")
        return {'FINISHED'}


class SCENERAY_SPLAT_OT_mark_pending(bpy.types.Operator):
    """Mark a rendered camera as pending so only it is re-rendered"""
    bl_idname = "sceneray_splat.mark_pending"
    bl_label = "Mark Pending"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(min=0)

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        if self.index >= len(cfg.camera_queue):
            return {'CANCELLED'}
        item = cfg.camera_queue[self.index]
        _sr_mark_cameras_pending(cfg, [item.camera])
        cfg.render_status = f"'{item.camera.name if item.camera else 'Camera'}' marked pending."
        _tag_redraw_sceneray_splat(context)
        return {'FINISHED'}

class SCENERAY_SPLAT_OT_queue_remove(bpy.types.Operator):
    """Remove this camera from the render queue (the object is kept)"""
    bl_idname = "sceneray_splat.queue_remove"
    bl_label = "Remove from Queue"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(min=0)

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        if self.index >= len(cfg.camera_queue):
            return {'CANCELLED'}
        item = cfg.camera_queue[self.index]
        name = item.camera.name if item.camera else "missing camera"
        cfg.camera_queue.remove(self.index)
        _sr_bulk["busy"] = True
        try:
            cfg.active_camera_index = min(cfg.active_camera_index,
                                          max(0, len(cfg.camera_queue) - 1))
        finally:
            _sr_bulk["busy"] = False
        _sr_mark_scene_cache_dirty(context.scene)
        self.report({'INFO'}, f"Removed '{name}' from the queue.")
        return {'FINISHED'}


def _sr_queue_needs_sync(scene, cfg):
    live = _sr_live_camera_map(scene)
    for item in cfg.camera_queue:
        camera = item.camera
        if not _sr_camera_matches_live_map(camera, live):
            return True
        try:
            if item.camera_name != camera.name:
                return True
        except ReferenceError:
            return True
    return False


def _sr_clean_camera_queue(scene, cfg, remove_duplicates=True):
    """Synchronize the queue with live scene cameras and optionally dedupe."""
    tolerance = 1e-5
    dead_indices, duplicates, kept = [], [], []
    removed_names = set()
    live = _sr_live_camera_map(scene)
    for index, item in enumerate(cfg.camera_queue):
        camera = item.camera
        if not _sr_camera_matches_live_map(camera, live):
            dead_indices.append(index)
            if item.camera_name:
                removed_names.add(item.camera_name)
            continue
        item.camera_name = camera.name
        if not remove_duplicates:
            continue
        pose, _scaled = camera_pose_world(camera)
        location, rotation = pose.to_translation(), pose.to_quaternion()
        duplicate = any(
            (location - old_location).length <= tolerance
            and abs(rotation.dot(old_rotation)) >= 1.0 - tolerance
            for old_location, old_rotation in kept)
        if duplicate:
            duplicates.append((index, camera))
        else:
            kept.append((location, rotation))

    _sr_bulk["busy"] = True
    try:
        for _index, camera in duplicates:
            try:
                removed_names.add(camera.name)
            except ReferenceError:
                pass
        _sr_remove_objects([camera for _index, camera in duplicates])
        duplicate_indices = [index for index, _camera in duplicates]
        for index in sorted(dead_indices + duplicate_indices, reverse=True):
            if index < len(cfg.camera_queue):
                cfg.camera_queue.remove(index)
        cfg.active_camera_index = min(
            cfg.active_camera_index, max(0, len(cfg.camera_queue) - 1))
    finally:
        _sr_bulk["busy"] = False
    _sr_prune_camera_from_manifests(cfg, removed_names)
    if dead_indices or duplicates:
        _sr_mark_scene_cache_dirty(scene)
    return len(dead_indices), len(duplicates)


_queue_sync = {
    "pending": set(), "timer_registered": False, "checks": {},
}


def _sr_run_pending_queue_sync():
    pending = set(_queue_sync["pending"])
    _queue_sync["pending"].clear()
    _queue_sync["timer_registered"] = False
    for scene in bpy.data.scenes:
        cfg = getattr(scene, "SCENERAY_SPLAT", None)
        if cfg is None:
            continue
        try:
            pointer = cfg.as_pointer()
        except ReferenceError:
            continue
        if pointer in pending and _sr_queue_needs_sync(scene, cfg):
            _sr_clean_camera_queue(scene, cfg, remove_duplicates=False)
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    return None


def _sr_request_queue_sync(scene, cfg):
    pointer = cfg.as_pointer()
    check = (_sr_scene_revision(scene), len(cfg.camera_queue))
    if _queue_sync["checks"].get(pointer) == check:
        return
    _queue_sync["checks"][pointer] = check
    if not _sr_queue_needs_sync(scene, cfg):
        return
    _queue_sync["pending"].add(pointer)
    if not _queue_sync["timer_registered"]:
        _queue_sync["timer_registered"] = True
        bpy.app.timers.register(_sr_run_pending_queue_sync,
                                first_interval=0.05)


class SCENERAY_SPLAT_OT_queue_clean(bpy.types.Operator):
    """Remove missing and duplicate managed cameras"""
    bl_idname = "sceneray_splat.queue_clean"
    bl_label = "Clean Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        missing, duplicate_count = _sr_clean_camera_queue(
            context.scene, cfg, remove_duplicates=True)
        self.report({'INFO'}, f"Cleaned {missing} missing and "
                    f"{duplicate_count} duplicate camera(s).")
        _tag_redraw_sceneray_splat(context)
        return {'FINISHED'}

class SCENERAY_SPLAT_OT_queue_clear(bpy.types.Operator):
    """Empty the render queue. The camera objects themselves are kept"""
    bl_idname = "sceneray_splat.queue_clear"
    bl_label = "Clear Render Queue"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        n = len(cfg.camera_queue)
        _sr_bulk["busy"] = True
        try:
            cfg.camera_queue.clear()
            cfg.active_camera_index = 0
        finally:
            _sr_bulk["busy"] = False
        _sr_mark_scene_cache_dirty(context.scene)
        self.report({'INFO'}, f"Cleared {n} camera(s) from the queue.")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════
# 15. CAMERA RIG PRESETS
# ═══════════════════════════════════════════════════════════════════════


def _sr_rig_library_dir():
    addon = bpy.context.preferences.addons.get(__package__)
    prefs = addon.preferences if addon is not None else None
    raw = getattr(prefs, "rig_library_dir", "") if prefs else ""
    # Blank/old preferences work immediately; user files never live inside the
    # add-on, so updates do not replace the user's library.
    if not raw or raw.startswith("//"):
        return camera_rigs.default_user_dir()
    path = Path(raw).expanduser()
    if not path.is_absolute() or path.resolve() == camera_rigs.bundled_dir().resolve():
        return camera_rigs.default_user_dir()
    return path


def _sr_preset_path(preset):
    folder = camera_rigs.bundled_dir() if preset.builtin else _sr_rig_library_dir()
    return folder / Path(preset.file_name).name


def _sr_rig_file_name(name, used=()):
    safe = "".join(ch if (ch.isalnum() or ch in " -_") else "_"
                   for ch in name).strip(" ._") or "Camera Rig"
    candidate, suffix = f"{safe}.json", 2
    used = {os.path.normcase(value) for value in used}
    while os.path.normcase(candidate) in used:
        candidate = f"{safe} {suffix}.json"
        suffix += 1
    return candidate


def _sr_camera_spec_data(camera, parent_source_id="", world_matrix=None,
                         source_world_matrix=None):
    """Serialize evaluated transforms so parenting and unapplied scale survive."""
    world_matrix = (world_matrix.copy() if world_matrix is not None
                    else camera.matrix_world.copy())
    location, rotation, scale = world_matrix.decompose()
    data = camera.data
    payload = {
        "source_name": camera.name, "location": list(location),
        "rotation": [rotation.w, rotation.x, rotation.y, rotation.z],
        "scale": list(scale), "matrix_world": matrix_to_flat_list(world_matrix),
        "lens": data.lens,
        "lens_unit": data.lens_unit, "sensor_width": data.sensor_width,
        "sensor_height": data.sensor_height, "sensor_fit": data.sensor_fit,
        "camera_type": data.type, "ortho_scale": data.ortho_scale, "shift_x": data.shift_x,
        "shift_y": data.shift_y, "clip_start": data.clip_start,
        "clip_end": data.clip_end, "display_size": data.display_size,
        "parent_source_id": parent_source_id,
    }
    if parent_source_id and source_world_matrix is not None:
        relative = source_world_matrix.inverted_safe() @ world_matrix
        payload["source_relative_matrix"] = matrix_to_flat_list(relative)
    return payload

def _sr_camera_spec_data_from_spec(spec):
    return {
        "source_name": spec.source_name, "location": list(spec.location),
        "rotation": list(spec.rotation), "scale": list(spec.scale),
        "matrix_world": list(spec.world_matrix),
        "lens": spec.lens, "lens_unit": spec.lens_unit,
        "sensor_width": spec.sensor_width, "sensor_height": spec.sensor_height,
        "sensor_fit": spec.sensor_fit, "camera_type": spec.camera_type, "ortho_scale": spec.ortho_scale,
        "shift_x": spec.shift_x, "shift_y": spec.shift_y,
        "clip_start": spec.clip_start, "clip_end": spec.clip_end,
        "display_size": spec.display_size,
        "parent_source_id": spec.parent_source_id,
        "source_relative_matrix": (list(spec.source_relative_matrix)
                                   if spec.has_source_relative else []),
    }

def _sr_data_to_spec(data, spec):
    spec.source_name = str(data.get("source_name", "SplatGen Camera"))
    spec.location = data.get("location", (0.0, 0.0, 0.0))
    spec.rotation = data.get("rotation", (1.0, 0.0, 0.0, 0.0))
    spec.scale = data.get("scale", (1.0, 1.0, 1.0))
    raw_matrix = data.get("matrix_world", ())
    try:
        spec.world_matrix = matrix_to_flat_list(flat_list_to_matrix(raw_matrix))
    except (TypeError, ValueError):
        spec.world_matrix = matrix_to_flat_list(
            Matrix.LocRotScale(Vector(spec.location), Quaternion(spec.rotation), Vector(spec.scale)))
    spec.lens = float(data.get("lens", 35.0))
    spec.lens_unit = str(data.get("lens_unit", "MILLIMETERS"))
    spec.sensor_width = float(data.get("sensor_width", 36.0))
    spec.sensor_height = float(data.get("sensor_height", 24.0))
    spec.sensor_fit = str(data.get("sensor_fit", "AUTO"))
    spec.camera_type = str(data.get("camera_type", "PERSP"))
    spec.ortho_scale = float(data.get('ortho_scale', 6.0))
    spec.shift_x = float(data.get("shift_x", 0.0))
    spec.shift_y = float(data.get("shift_y", 0.0))
    spec.clip_start = float(data.get("clip_start", 0.1))
    spec.clip_end = float(data.get("clip_end", 1000.0))
    spec.display_size = float(data.get("display_size", 0.5))
    spec.parent_source_id = str(data.get("parent_source_id", ""))
    raw_relative = data.get("source_relative_matrix", ())
    try:
        spec.source_relative_matrix = matrix_to_flat_list(
            flat_list_to_matrix(raw_relative))
        spec.has_source_relative = True
    except (TypeError, ValueError):
        spec.has_source_relative = False


def _sr_apply_spec_to_camera(spec, cam):
    """Rebuild a camera from a snapshot — the same scale it was saved with."""
    data = cam.data
    data.type = spec.camera_type or 'PERSP'
    data.ortho_scale = spec.ortho_scale
    data.sensor_fit = spec.sensor_fit or 'AUTO'
    data.sensor_width, data.sensor_height = spec.sensor_width, spec.sensor_height
    data.lens = spec.lens
    try:
        data.lens_unit = spec.lens_unit or 'MILLIMETERS'
    except TypeError:
        pass
    data.shift_x, data.shift_y = spec.shift_x, spec.shift_y
    data.clip_start, data.clip_end = spec.clip_start, spec.clip_end
    data.display_size = spec.display_size or 0.5
    cam.location = spec.location
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = spec.rotation
    cam.scale = spec.scale
    # Keep the full matrix as well as location/rotation/scale. A camera
    # parented under a non-uniformly scaled source can contain shear, which
    # decomposition cannot represent without changing its pose.
    try:
        cam.matrix_world = flat_list_to_matrix(spec.world_matrix)
    except (TypeError, ValueError):
        pass


_RIG_SOURCE_ID_KEY = "_sceneray_splat_rig_source_id"


def _sr_hide_rig_source_from_render(obj):
    """Keep a camera-placement helper visible for editing, never rendering."""
    obj.hide_render = True
    for attribute in (
        "visible_camera",
        "visible_diffuse",
        "visible_glossy",
        "visible_transmission",
        "visible_volume_scatter",
        "visible_shadow",
    ):
        if hasattr(obj, attribute):
            try:
                setattr(obj, attribute, False)
            except (AttributeError, RuntimeError, TypeError):
                pass
    display = getattr(obj, "display", None)
    if display is not None and hasattr(display, "show_shadows"):
        try:
            display.show_shadows = False
        except (AttributeError, RuntimeError, TypeError):
            pass


def _sr_set_exact_parented_world(obj, parent, world_matrix):
    """Store an arbitrary evaluated 4x4 without losing shear to decomposition."""
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted_safe() @ world_matrix
    obj.matrix_basis = Matrix.Identity(4)


def _sr_source_id(source):
    """A stable ID lets a loaded rig reconnect to its original helper mesh."""
    source_id = str(source.get(_RIG_SOURCE_ID_KEY, ""))
    if not source_id:
        source_id = uuid.uuid4().hex
        source[_RIG_SOURCE_ID_KEY] = source_id
    return source_id


def _sr_source_spec_data(source, source_id, world_matrix=None):
    world_matrix = (world_matrix.copy() if world_matrix is not None
                    else source.matrix_world.copy())
    payload = {
        "id": source_id,
        "name": source.name,
        "object_type": source.type,
        "matrix_world": matrix_to_list(world_matrix),
        "hide_render": bool(source.hide_render),
        "display_type": source.display_type,
    }
    if source.type == 'MESH':
        payload["mesh"] = {
            "vertices": [list(vertex.co) for vertex in source.data.vertices],
            "faces": [list(face.vertices) for face in source.data.polygons],
        }
    elif source.type == 'EMPTY':
        payload['empty_display_type'] = source.empty_display_type
        payload['empty_display_size'] = source.empty_display_size
    return payload

def _sr_rig_payload_from_cameras(cameras, depsgraph=None):
    """Serialize evaluated camera/source transforms as one coherent snapshot."""
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    sources, source_matrices, camera_specs = {}, {}, []
    for camera in cameras:
        try:
            camera_world = camera.evaluated_get(depsgraph).matrix_world.copy()
        except (ReferenceError, RuntimeError):
            camera_world = camera.matrix_world.copy()
        source = camera.parent
        source_id = ""
        source_world = None
        if source is not None:
            source_id = _sr_source_id(source)
            if source_id not in sources:
                try:
                    source_world = source.evaluated_get(
                        depsgraph).matrix_world.copy()
                except (ReferenceError, RuntimeError):
                    source_world = source.matrix_world.copy()
                source_matrices[source_id] = source_world
                sources[source_id] = _sr_source_spec_data(
                    source, source_id, source_world)
            else:
                source_world = source_matrices[source_id]
        camera_specs.append(_sr_camera_spec_data(
            camera, source_id, camera_world, source_world))
    return camera_specs, list(sources.values())

def _sr_preset_sources(preset):
    """Read source payload only when it is needed, not during UI drawing."""
    if not preset.file_name:
        return []
    try:
        with open(_sr_preset_path(preset), encoding='utf-8') as f:
            sources = json.load(f).get("sources", [])
    except (OSError, ValueError, TypeError):
        return []
    return [source for source in sources if isinstance(source, dict)]


def _sr_rig_cursor_placement(scene, preset, source_payloads):
    """Translate the saved rig as one unit so its origin lands at the cursor."""
    if preset.builtin:
        # Bundled rigs are authored in metres around a deliberate base origin.
        units = max(float(scene.unit_settings.scale_length), 1e-9)
        return Matrix.Translation(scene.cursor.location) @ Matrix.Scale(1.0/units, 4)
    locations = []
    for source_data in source_payloads:
        try:
            locations.append(
                list_to_matrix(
                    source_data.get("matrix_world", ())
                ).to_translation()
            )
        except (TypeError, ValueError):
            continue
    if not locations:
        for spec in preset.cameras:
            try:
                locations.append(
                    flat_list_to_matrix(spec.world_matrix).to_translation()
                )
            except (TypeError, ValueError):
                continue
    if not locations:
        saved_origin = Vector((0.0, 0.0, 0.0))
    elif source_payloads:
        saved_origin = sum(
            locations,
            Vector((0.0, 0.0, 0.0)),
        ) / len(locations)
    else:
        minimum = Vector(
            tuple(min(point[axis] for point in locations) for axis in range(3))
        )
        maximum = Vector(
            tuple(max(point[axis] for point in locations) for axis in range(3))
        )
        saved_origin = (minimum + maximum) * 0.5
    return Matrix.Translation(scene.cursor.location - saved_origin)


def _sr_restore_rig_source(
    scene,
    target,
    source_data,
    placement_matrix=None,
):
    """Rebuild only the saved rig source, without an extra carrier Empty."""
    source_id = str(source_data.get("id", ""))
    if not source_id:
        return None, "saved source has no identifier"
    try:
        saved_world = list_to_matrix(source_data.get("matrix_world", []))
    except (TypeError, ValueError):
        saved_world = Matrix.Identity(4)
    if placement_matrix is not None:
        saved_world = placement_matrix @ saved_world
    source_name = str(source_data.get("name") or "Rig Source")

    obj = None
    if source_data.get("object_type") == "MESH":
        mesh_data = source_data.get("mesh")
        if not isinstance(mesh_data, dict):
            return (
                None,
                f"source '{source_name}' has no mesh data",
            )
        try:
            vertices = [
                tuple(float(value) for value in vertex)
                for vertex in mesh_data.get("vertices", [])
            ]
            faces = [
                tuple(int(value) for value in face)
                for face in mesh_data.get("faces", [])
            ]
            if not vertices:
                raise ValueError("no vertices")
            mesh = bpy.data.meshes.new(f"{source_name} Mesh")
            mesh.from_pydata(vertices, [], faces)
            mesh.update()
            obj = bpy.data.objects.new(source_name, mesh)
        except (TypeError, ValueError, RuntimeError) as exc:
            return (
                None,
                f"source '{source_name}' could not be rebuilt ({exc})",
            )
    elif source_data.get('object_type') == 'EMPTY':
        obj = bpy.data.objects.new(source_name, None)
        obj.empty_display_type = source_data.get('empty_display_type', 'PLAIN_AXES')
        obj.empty_display_size = float(source_data.get('empty_display_size', 1.0))
    else:
        template = next(
            (
                candidate
                for candidate in scene.objects
                if str(candidate.get(_RIG_SOURCE_ID_KEY, "")) == source_id
            ),
            None,
        )
        if template is None:
            return (
                None,
                f"source '{source_name}' is unavailable in this scene",
            )
        obj = template.copy()
        if getattr(template, "data", None) is not None:
            try:
                obj.data = template.data.copy()
            except (AttributeError, RuntimeError):
                obj.data = template.data

    target.objects.link(obj)
    obj.parent = None
    obj.matrix_world = saved_world
    _sr_hide_rig_source_from_render(obj)
    try:
        obj.display_type = source_data.get("display_type", 'WIRE')
    except (TypeError, ValueError):
        obj.display_type = 'WIRE'
    obj[_RIG_SOURCE_ID_KEY] = uuid.uuid4().hex
    for view_layer in scene.view_layers:
        try:
            view_layer.update()
        except RuntimeError:
            pass
    return obj, None

def _sr_parent_camera_to_source(
    camera,
    source,
    spec=None,
    placement_matrix=None,
):
    """Rebuild the saved local relationship under the restored source."""
    try:
        saved_world = (flat_list_to_matrix(spec.world_matrix)
                       if spec else camera.matrix_world.copy())
    except (TypeError, ValueError):
        saved_world = camera.matrix_world.copy()
    if placement_matrix is not None:
        saved_world = placement_matrix @ saved_world
    camera.parent = source
    if spec is not None and spec.has_source_relative:
        try:
            relative = flat_list_to_matrix(spec.source_relative_matrix)
            # Parent inverse is a full matrix property; unlike transform
            # channels it preserves affine shear introduced by scaled parents.
            camera.matrix_parent_inverse = relative
            camera.matrix_basis = Matrix.Identity(4)
            return
        except (TypeError, ValueError):
            pass
    # Backward compatibility for presets saved before source-relative matrices.
    _sr_set_exact_parented_world(camera, source, saved_world)

_rig_library_cache = {
    "key": None, "pending": {}, "timer_registered": False,
    "last_checks": {},
}
_RIG_LIBRARY_CHECK_INTERVAL = 2.0


def _sr_run_pending_rig_library_refresh():
    pending = dict(_rig_library_cache["pending"])
    _rig_library_cache["pending"].clear()
    _rig_library_cache["timer_registered"] = False
    for scene in bpy.data.scenes:
        cfg = getattr(scene, "SCENERAY_SPLAT", None)
        if cfg is None:
            continue
        pointer = cfg.as_pointer()
        if pointer not in pending:
            continue
        try:
            _sr_refresh_rig_library(cfg, force=pending[pointer])
            _tag_redraw_sceneray_splat(bpy.context)
        except ReferenceError:
            pass
    return None


def _sr_request_rig_library_refresh(cfg, force=False):
    pointer = cfg.as_pointer()
    folder = _sr_rig_library_dir()
    if folder is None:
        return
    folder_key = str(folder)
    now = time.monotonic()
    last = _rig_library_cache["last_checks"].get(pointer)
    if (not force and last is not None and last[0] == folder_key
            and now - last[1] < _RIG_LIBRARY_CHECK_INTERVAL):
        return
    _rig_library_cache["last_checks"][pointer] = (folder_key, now)
    _rig_library_cache["pending"][pointer] = (
        _rig_library_cache["pending"].get(pointer, False) or force)
    if _rig_library_cache["timer_registered"]:
        return
    try:
        bpy.app.timers.register(_sr_run_pending_rig_library_refresh,
                                first_interval=0.05)
        _rig_library_cache["timer_registered"] = True
    except Exception:
        from . import diagnostics as _diag
        _diag.swallowed("sceneray_splat.py")


def _sr_refresh_rig_library(cfg, force=False):
    """Merge immutable bundled rigs and the independent user library."""
    folder = _sr_rig_library_dir()
    files, stamp = [], []
    for directory, builtin in ((camera_rigs.bundled_dir(), True), (folder, False)):
        try:
            for path in sorted(directory.glob('*.json')):
                files.append((path, builtin))
                stamp.append((str(path), path.stat().st_mtime_ns, path.stat().st_size))
        except OSError:
            continue
    key = (cfg.as_pointer(), str(folder), tuple(stamp))
    if not force and _rig_library_cache["key"] == key:
        return
    selected = ((cfg.rig_presets[cfg.active_preset_index].builtin,
                 cfg.rig_presets[cfg.active_preset_index].file_name)
                if 0 <= cfg.active_preset_index < len(cfg.rig_presets) else None)
    cfg.rig_presets.clear()
    for path, builtin in files:
        try:
            with open(path, encoding='utf-8') as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or not isinstance(payload.get("cameras"), list) or not payload['cameras']:
                continue
            if any(not isinstance(item, dict) for item in payload['cameras']):
                continue
            preset = cfg.rig_presets.add()
            preset.name = str(payload.get("name") or path.stem)
            preset.file_name = path.name
            preset.builtin = builtin
            preset.description = str(payload.get('description', ''))
            for spec_data in payload.get("cameras", []):
                _sr_data_to_spec(spec_data, preset.cameras.add())
        except (OSError, ValueError, TypeError):
            # Invalid files must not leave a half-populated entry in the list.
            if len(cfg.rig_presets) and cfg.rig_presets[-1].file_name == path.name and cfg.rig_presets[-1].builtin == builtin:
                cfg.rig_presets.remove(len(cfg.rig_presets)-1)
            continue
    cfg.active_preset_index = next(
        (i for i, preset in enumerate(cfg.rig_presets)
         if (preset.builtin, preset.file_name) == selected), 0)
    _rig_library_cache["key"] = key


def _sr_write_rig_file(folder, file_name, name, cameras, sources=()):
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"format": "sceneray_splat-rig", "version": 3, "name": name,
               "cameras": cameras, "sources": list(sources)}
    with open(folder / file_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class SCENERAY_SPLAT_OT_preset_save(bpy.types.Operator):
    """Save selected or queued cameras into the shared rig library"""
    bl_idname = "sceneray_splat.preset_save"
    bl_label = "Save Rig"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        folder = _sr_rig_library_dir()
        if folder is None:
            self.report({'ERROR'}, "Choose a Camera Rig Library folder in add-on preferences first.")
            return {'CANCELLED'}
        selected = list(context.selected_objects)
        cameras = list(dict.fromkeys(o for root in selected
            for o in (root, *root.children_recursive) if o.type == 'CAMERA'))
        cameras = cameras or gather_scene_cameras(context, cfg)
        if not cameras:
            self.report({'ERROR'}, "Select the rig's cameras, or queue cameras first.")
            return {'CANCELLED'}
        _sr_refresh_rig_library(cfg)
        name, suffix = 'Camera Rig', 2
        while any(p.name == name for p in cfg.rig_presets):
            name = f'Camera Rig {suffix}'
            suffix += 1
        file_name = _sr_rig_file_name(name, [p.name for p in folder.glob('*.json')])
        try:
            context.view_layer.update()
        except RuntimeError:
            pass
        camera_specs, source_specs = _sr_rig_payload_from_cameras(
            cameras, context.evaluated_depsgraph_get())
        try:
            _sr_write_rig_file(folder, file_name, name, camera_specs, source_specs)
        except OSError as exc:
            self.report({'ERROR'}, f"Could not save rig: {exc}")
            return {'CANCELLED'}
        _sr_refresh_rig_library(cfg, force=True)
        cfg.active_preset_index = next(i for i,p in enumerate(cfg.rig_presets) if not p.builtin and p.file_name==file_name)
        source_note = f" with {len(source_specs)} source object(s)" if source_specs else ""
        self.report({'INFO'}, f"Saved {len(cameras)} camera(s){source_note} to shared rig '{name}'.")
        return {'FINISHED'}


class SCENERAY_SPLAT_OT_preset_rename(bpy.types.Operator):
    bl_idname = "sceneray_splat.preset_rename"
    bl_label = "Rename Rig"
    bl_options = {'REGISTER'}
    new_name: StringProperty(name="Name")

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and 0 <= cfg.active_preset_index < len(cfg.rig_presets)
                and not cfg.rig_presets[cfg.active_preset_index].builtin)

    def invoke(self, context, event):
        cfg = context.scene.SCENERAY_SPLAT
        self.new_name = cfg.rig_presets[cfg.active_preset_index].name
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        theme.heading(self.layout, "Rename camera rig", fallback="BOOKMARKS")
        theme.form(self.layout).prop(self, "new_name")

    def execute(self, context):
        cfg, folder = context.scene.SCENERAY_SPLAT, _sr_rig_library_dir()
        if not self.poll(context):
            return {'CANCELLED'}
        preset, name = cfg.rig_presets[cfg.active_preset_index], self.new_name.strip()
        if not name:
            self.report({'ERROR'}, "Enter a rig name.")
            return {'CANCELLED'}
        old_path = _sr_preset_path(preset)
        used = [p.name for p in folder.glob('*.json')
                if os.path.normcase(p.name) != os.path.normcase(preset.file_name)]
        new_file_name = _sr_rig_file_name(name, used)
        try:
            _sr_write_rig_file(folder, new_file_name, name,
                               [_sr_camera_spec_data_from_spec(spec)
                                for spec in preset.cameras],
                               _sr_preset_sources(preset))
            if os.path.normcase(new_file_name) != os.path.normcase(preset.file_name) and old_path.is_file():
                old_path.unlink()
        except OSError as exc:
            self.report({'ERROR'}, f"Could not rename rig: {exc}")
            return {'CANCELLED'}
        _sr_refresh_rig_library(cfg, force=True)
        cfg.active_preset_index = next(i for i,p in enumerate(cfg.rig_presets)
            if not p.builtin and os.path.normcase(p.file_name)==os.path.normcase(new_file_name))
        return {'FINISHED'}


class SCENERAY_SPLAT_OT_preset_delete(bpy.types.Operator):
    bl_idname = "sceneray_splat.preset_delete"
    bl_label = "Delete Rig"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and 0 <= cfg.active_preset_index < len(cfg.rig_presets)
                and not cfg.rig_presets[cfg.active_preset_index].builtin)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        cfg, folder = context.scene.SCENERAY_SPLAT, _sr_rig_library_dir()
        if not self.poll(context):
            return {'CANCELLED'}
        preset = cfg.rig_presets[cfg.active_preset_index]
        name = preset.name
        try:
            _sr_preset_path(preset).unlink(missing_ok=True)
        except OSError as exc:
            self.report({'ERROR'}, f"Could not delete rig: {exc}")
            return {'CANCELLED'}
        _sr_refresh_rig_library(cfg, force=True)
        self.report({'INFO'}, f"Deleted shared rig '{name}'.")
        return {'FINISHED'}


class SCENERAY_SPLAT_OT_preset_load(bpy.types.Operator):
    """Recreate the selected shared rig's cameras in this scene and queue them"""
    bl_idname = "sceneray_splat.preset_load"
    bl_label = "Add Rig to Scene"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and len(cfg.rig_presets) > 0
                and not cfg.is_rendering and not cfg.is_generating_points)

    def execute(self, context):
        scene, cfg = context.scene, context.scene.SCENERAY_SPLAT
        if cfg.active_preset_index >= len(cfg.rig_presets):
            self.report({'ERROR'}, "Choose a saved rig first.")
            return {'CANCELLED'}
        preset, target = cfg.rig_presets[cfg.active_preset_index], (context.collection or scene.collection)
        source_payloads = _sr_preset_sources(preset)
        placement = _sr_rig_cursor_placement(
            scene,
            preset,
            source_payloads,
        )
        controller = None
        if preset.builtin:
            controller = bpy.data.objects.new(f'{preset.name} Rig', None)
            target.objects.link(controller)
            controller.empty_display_type = 'CUBE'
            controller.empty_display_size = .25
            controller.matrix_world = placement
            controller['splatgen_camera_rig'] = preset.name
            controller['splatgen_rig_units'] = 'METERS'
            _sr_hide_rig_source_from_render(controller)
        sources, missing_sources, restored_sources = {}, [], 0
        for source_data in source_payloads:
            source, problem = _sr_restore_rig_source(
                scene,
                target,
                source_data,
                placement,
            )
            source_id = str(source_data.get("id", ""))
            if source is not None:
                sources[source_id] = source
                restored_sources += 1
            elif problem:
                missing_sources.append(problem)
        _sr_bulk["busy"] = True
        added = 0
        try:
            for spec in preset.cameras:
                base = spec.source_name or "SplatGen Camera"
                data = bpy.data.cameras.new(base)
                cam = bpy.data.objects.new(base, data)
                target.objects.link(cam)
                _sr_apply_spec_to_camera(spec, cam)
                source = sources.get(spec.parent_source_id)
                if spec.parent_source_id and source is not None:
                    _sr_parent_camera_to_source(
                        cam,
                        source,
                        spec,
                        placement,
                    )
                elif spec.parent_source_id:
                    missing_sources.append(f"camera '{base}' has no restored source object")
                    cam.matrix_world = placement @ cam.matrix_world
                else:
                    cam.matrix_world = placement @ cam.matrix_world
                if preset.builtin:
                    # Keep the authored field of view on the smaller image axis.
                    render = scene.render
                    wide = render.resolution_x*render.pixel_aspect_x >= render.resolution_y*render.pixel_aspect_y
                    data.sensor_fit = 'VERTICAL' if wide else 'HORIZONTAL'
                    units = max(scene.unit_settings.scale_length, 1e-9)
                    data.clip_start = spec.clip_start / units
                    data.clip_end = spec.clip_end / units
                    # Queued parented cameras keep world scale 1 for clean icons.
                    data.display_size = .15 / units
                    _sr_set_exact_parented_world(cam, controller, cam.matrix_world.copy())
                _sr_queue_add(cfg, cam)
                added += 1
        finally:
            _sr_bulk["busy"] = False
        if added:
            cfg.active_camera_index = len(cfg.camera_queue) - 1
            # Newly added rigs can immediately be moved/scaled together with G/S.
            for obj in context.selected_objects: obj.select_set(False)
            if controller is not None:
                controller.select_set(True)
                context.view_layer.objects.active = controller
            else:
                for item in list(cfg.camera_queue)[-added:]: item.camera.select_set(True)
                context.view_layer.objects.active = cfg.camera_queue[-1].camera
        if missing_sources:
            self.report({'WARNING'}, "Rig loaded, but some source objects could not be restored: "
                        + "; ".join(missing_sources[:2]))
        else:
            source_note = f" and {restored_sources} source object(s)" if restored_sources else ""
            self.report(
                {'INFO'},
                f"Added shared rig '{preset.name}' at the 3D Cursor - "
                f"{added} camera(s){source_note} queued.",
            )
        return {'FINISHED'}

# 16. COMPLETE DATASET WORKFLOW  (guarded)
# ═══════════════════════════════════════════════════════════════════════


_SR_DATASET_FILES = bd_paths.DATASET_FILES


def _sr_repoint_dataset_references(scene, old_path, new_path):
    """Follow a renamed version so stored paths never dangle.

    Retiring a folder changes its name but not its contents, so anything
    that referred to it must be pointed at the new name. Without this the
    next validation looks for a folder that no longer exists under that name
    and reports the dataset as missing.
    """
    if scene is None or old_path is None or new_path is None:
        return

    def same(value):
        return str(value or "").rstrip("\\/") == str(old_path).rstrip("\\/")

    cfg = getattr(scene, "SCENERAY_SPLAT", None)
    if cfg is not None:
        for attribute in ("active_dataset_dir", "latest_dataset_dir"):
            if same(getattr(cfg, attribute, "")):
                setattr(cfg, attribute, str(new_path))


def _sr_retire_previous_version(base, previous):
    """Keep exactly one recovery copy of a superseded unfinished build.

    Its reusable work has already been copied into the new build, so the old
    folder becomes *the* recovery folder and any earlier recovery folder is
    removed. One fallback is kept; they never accumulate.

    Returns ``(recovery_path, removed_names)``.
    """
    return bd_paths.rotate_recovery(base, previous)


def _sr_inspect_dataset_state(context, cfg):
    """Report what already exists before anything is written.

    Camera status is the source of truth, but a status is only trusted when
    the image behind it is actually on disk: ``_sr_sync_render_states``
    demotes any camera whose file has gone missing, so a dataset can never be
    reported complete on the strength of a stale flag.
    """
    base = _sr_output_base_path(cfg)
    previous = (
        _sr_latest_completed_dataset(base, getattr(cfg, "latest_dataset_dir", ""))
        if base is not None
        else None
    )
    # Reconcile every camera against the files actually present.
    try:
        _sr_sync_render_states(context.scene, cfg, previous)
    except Exception:
        from . import diagnostics as _diag
        _diag.swallowed("sceneray_splat.py")

    queued = [
        item
        for item in cfg.camera_queue
        if item.camera is not None and item.camera.type == 'CAMERA'
    ]
    rendered = sum(1 for item in queued if item.render_state == 'RENDERED')
    pending = len(queued) - rendered

    missing_files = []
    if previous is not None:
        missing_files = [
            name for name in _SR_DATASET_FILES
            if not _sr_dataset_file(previous, name).is_file()
        ]

    if previous is None:
        # With explicit Continue/New actions, camera flags no longer imply
        # permission to replace or bypass an existing on-disk build.
        state = 'NEW'
    elif pending > 0:
        state = 'INCOMPLETE'
    elif missing_files:
        state = 'INCOMPLETE'
    else:
        state = 'COMPLETE'

    return {
        "state": state,
        "previous": previous,
        "previous_name": previous.name if previous is not None else "",
        "total": len(queued),
        "rendered": rendered,
        "pending": pending,
        "missing_files": missing_files,
    }


def _sr_prepare_versioned_dataset(context, cfg, reuse_images=True):
    """Create one version folder and carry forward only reusable images."""
    base = _sr_output_base_path(cfg)
    if base is None:
        raise ValueError("Choose an output folder before building the dataset.")
    if not base.is_absolute():
        raise ValueError(
            "The output folder must resolve to an absolute path. "
            "Save the .blend file or choose an absolute folder."
        )
    previous = _sr_latest_completed_dataset(
        base,
        getattr(cfg, "latest_dataset_dir", ""),
    )
    # new_build_folder already creates the images folder that holds both the
    # renders and their metadata.
    target = _sr_new_version_folder(base)

    previous_data = None
    if previous is not None:
        previous_data = (
            read_render_manifest(previous)
            or read_completed_camera_data(previous)
        )
    previous_entries = {
        entry.get("name"): entry
        for entry in (previous_data or {}).get("cameras", ())
        if isinstance(entry, dict) and entry.get("name")
    }
    queue_items = {
        item.camera.name: item
        for item in cfg.camera_queue
        if item.camera is not None and item.camera.type == 'CAMERA'
    }
    reusable = []
    for name, item in queue_items.items():
        if not reuse_images:
            # Starting over: every camera renders again and nothing is
            # carried forward. The earlier version stays where it is.
            item.render_state = 'PENDING'
            continue
        entry = previous_entries.get(name)
        if entry is None or item.render_state != 'RENDERED':
            item.render_state = 'PENDING'
            continue
        relative = str(entry.get("file_path", "")).replace("\\", "/")
        training_relative = str(
            entry.get("training_file_path", relative)
        ).replace("\\", "/")
        source = (
            previous / relative.lstrip("./")
            if previous is not None and relative
            else None
        )
        training_source = (
            previous / training_relative.lstrip("./")
            if previous is not None and training_relative
            else None
        )
        if (
            source is None
            or not source.is_file()
            or training_source is None
            or not training_source.is_file()
        ):
            item.render_state = 'PENDING'
            continue
        target_relative = (
            bd_paths.data_dir(target) / Path(relative).name
        ).relative_to(target).as_posix()
        destination = target / target_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        training_target_relative = (
            bd_paths.data_dir(target) / Path(training_relative).name
        ).relative_to(target).as_posix()
        training_destination = target / training_target_relative
        if training_destination != destination:
            training_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(training_source, training_destination)
        frame_index = int(entry.get("frame_index", len(reusable)))
        if not bd_export.copy_view_outputs(
            previous, target, frame_index, cfg
        ):
            # RGB alone is not a complete standard view: every view also owns
            # its geometry-derived mask and any explicitly requested GT data.
            item.render_state = 'PENDING'
            continue
        # Raw files are optional here: whatever is missing is rendered by
        # the raw stage, so a reused view never needs a legacy re-render.
        raw_hooks.copy_view(previous, target, frame_index)
        carried = {
            "name": name,
            "file_path": f"./{target_relative}",
            "training_file_path": f"./{training_target_relative}",
            "frame_index": frame_index,
            "pending": False,
            "status": "RENDERED",
        }
        # The copied image was rendered with the old settings, so it carries
        # the old signature: if the camera has since changed, the next plan
        # sees the mismatch and re-renders it.
        signature = bd_manifest.signature_of(entry)
        if signature is not None:
            carried["signature"] = signature
        reusable.append(carried)
        item.render_state = 'RENDERED'

    if reusable:
        resolution = tuple(
            (previous_data or {}).get(
                "resolution",
                effective_resolution(context.scene.render),
            )
        )
        write_render_manifest(
            target,
            reusable,
            resolution,
            color_management=(previous_data or {}).get("color_management"),
        )

    cfg.active_dataset_dir = str(target)
    _sr_sync_render_states(context.scene, cfg, target)
    _render_status_sync["signatures"][cfg.as_pointer()] = (
        _sr_render_status_signature(cfg)
    )
    pending = sum(
        1
        for item in cfg.camera_queue
        if item.camera is not None and item.render_state == 'PENDING'
    )
    cfg.render_status = (
        f"Dataset version {target.name}: {len(reusable)} image(s) reused, "
        f"{pending} camera(s) to render."
    )
    return target, previous, len(reusable), pending


def _sr_continue_dataset_generation(context):
    """Phase 2, and the handover into phase 3.

    Rendering has finished, so the metadata is rewritten from the images that
    were just made; the full workflow then verifies those images and starts
    the point cloud.
    """
    cfg = context.scene.SCENERAY_SPLAT
    cfg.write_metadata_after_render = False
    progress.set_stage(PHASE_FILES, message="Writing cameras.txt")
    progress.update(fraction=0.15, detail="0 of 2 files written")
    # Calling an operator whose poll fails raises, which used to strand the
    # build at this exact fraction. Ask first, and say which flag is wrong.
    if not SCENERAY_SPLAT_OT_calculate_cameras.poll(context):
        reason = (
            "a render is still marked as running" if cfg.is_rendering
            else "the point cloud is still marked as running"
            if cfg.is_generating_points
            else "no project folder is set"
        )
        cfg.render_status = (
            f"Rendering finished, but writing cameras.txt could not start: "
            f"{reason}."
        )
        cfg.build_workflow = 'NONE'
        progress.fail(cfg.render_status)
        return {'CANCELLED'}
    try:
        result = bpy.ops.sceneray_splat.calculate_cameras('EXEC_DEFAULT')
    except RuntimeError as exc:
        cfg.render_status = f"cameras.txt could not be written: {exc}"
        cfg.build_workflow = 'NONE'
        progress.fail(cfg.render_status)
        return {'CANCELLED'}
    if 'FINISHED' not in result:
        cfg.render_status = (
            "Rendering finished, but cameras.txt/images.txt could not be "
            "written."
        )
        cfg.build_workflow = 'NONE'
        progress.fail("cameras.txt could not be written")
        return result
    progress.update(
        fraction=0.85,
        message="Writing images.txt",
        detail="cameras.txt written",
    )

    progress.update(
        fraction=1.0,
        message="cameras.txt and images.txt written",
        detail="2 of 2 files written",
    )
    if cfg.build_workflow == 'FULL':
        # The point cloud takes its colour from these images, so it does not
        # start until every one of them is on disk and can actually be read.
        missing = _sr_unreadable_render_images(cfg)
        if missing:
            names = ", ".join(missing[:3])
            if len(missing) > 3:
                names += f" and {len(missing) - 3} more"
            cfg.render_status = f"Renders missing or unreadable: {names}"
            cfg.build_workflow = 'NONE'
            progress.fail(cfg.render_status)
            _tag_redraw_sceneray_splat(context)
            return {'CANCELLED'}
        progress.set_stage(
            PHASE_POINTS, message="Preparing to sample the point cloud"
        )
        outcome = bpy.ops.sceneray_splat.generate_points3d('INVOKE_DEFAULT')
        if 'CANCELLED' in outcome:
            cfg.build_workflow = 'NONE'
            progress.fail("Point cloud generation did not start")
        return result

    cfg.render_status = (
        "Rendering Images complete: images, cameras.txt and images.txt are "
        "up to date. Next: Point Cloud."
    )
    finished = "Images and camera data are up to date"
    if (raw_stage.should_run(context.scene, cfg.build_workflow)
            and raw_stage.start(context, _sr_effective_output_path(cfg),
                                finished_message=finished)):
        _tag_redraw_sceneray_splat(context)
        return result
    cfg.build_workflow = 'NONE'
    progress.end(finished)
    _tag_redraw_sceneray_splat(context)
    return result


def _sr_unreadable_render_images(cfg):
    """Images a finished render should have produced but did not.

    Existing is not enough: a truncated or unreadable file would silently
    become grey points, so each one is opened before the point cloud starts.
    """
    output_dir = _sr_effective_output_path(cfg)
    if output_dir is None:
        return ["no build folder"]
    manifest = read_render_manifest(output_dir) or read_completed_camera_data(
        output_dir
    )
    if not manifest:
        return ["no render manifest"]
    missing = []
    for entry in manifest.get("cameras", ()):
        if str(entry.get("status", "")).upper() != "RENDERED":
            continue
        relative = str(
            entry.get("training_file_path", entry.get("file_path", ""))
        ).lstrip("./\\")
        if not relative:
            missing.append(str(entry.get("name", "camera")))
            continue
        path = output_dir / relative
        if not path.is_file():
            missing.append(path.name)
            continue
        try:
            _read_image_rgb(path)
        except Exception:
            missing.append(path.name)
    return missing


def _sr_record_completed_dataset(scene, output_dir):
    """Record the completed dataset without launching a trainer or viewer."""
    cfg = getattr(scene, "SCENERAY_SPLAT", None)
    if cfg is not None:
        cfg.active_dataset_dir = str(Path(output_dir))
        cfg.latest_dataset_dir = str(Path(output_dir))



_SR_SCENE_COLLECTION = "All Scene"
_SR_ADDON_COLLECTION = "Gaussian Splats"
_SR_RECON_COLLECTION = "Training Data"
_SR_DATA_COLLECTION = "data"
_SR_SPLAT_COLLECTION = "splats"
_SR_TRANSFORM_EMPTY = "SplatGen Control"
_SR_SPLAT_MARKER = "Splats Display"
# Generated objects use a predictable publishing-friendly hierarchy:
# Gaussian Splats > Training Data > data / splats.
_SR_MANAGED_COLLECTIONS = (
    _SR_SCENE_COLLECTION,
    _SR_ADDON_COLLECTION,
    _SR_RECON_COLLECTION,
    _SR_DATA_COLLECTION,
    _SR_SPLAT_COLLECTION,
)
_SR_COLLECTION_PARENTS = {
    _SR_RECON_COLLECTION: _SR_ADDON_COLLECTION,
    _SR_DATA_COLLECTION: _SR_RECON_COLLECTION,
    _SR_SPLAT_COLLECTION: _SR_RECON_COLLECTION,
}


def sr_shortcut_label(idname):
    """The key combination currently bound to an operator, or "".

    Read from the user's keymap rather than hard-coded, so a rebound key is
    reflected on the button instead of the label quietly going stale.
    """
    # Read the current binding; a cached label can outlive a user's keymap edit.
    label = ""
    try:
        configs = bpy.context.window_manager.keyconfigs
        for config in (configs.user, configs.addon):
            if config is None:
                continue
            matched = False
            for keymap in config.keymaps:
                for item in keymap.keymap_items:
                    if item.idname != idname:
                        continue
                    matched = True
                    if not item.active:
                        continue
                    parts = []
                    if item.ctrl:
                        parts.append("Ctrl")
                    if item.alt:
                        parts.append("Alt")
                    if item.shift:
                        parts.append("Shift")
                    if item.oskey:
                        parts.append("Cmd")
                    key = str(item.type).replace("_", " ").title()
                    parts.append(key)
                    label = "+".join(parts)
                    break
                if label:
                    break
            if matched:
                break
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        label = ""
    return label


_SR_ROLE = "splatgen_role"


def _sr_scene_collections(scene):
    try:
        return list(scene.collection.children_recursive)
    except (AttributeError, ReferenceError, RuntimeError):
        return []


def _sr_used_elsewhere(collection, scene):
    """Whether another scene links this collection too."""
    for other in bpy.data.scenes:
        if other != scene and collection in _sr_scene_collections(other):
            return True
    return False


def _sr_collection(name, scene=None):
    """The managed collection ``name`` that belongs to ``scene``.

    Every scene gets its own. Looked up by name alone, one collection ended
    up shared by every scene of the file - an appended scene asset, say -
    and replacing cameras in one scene deleted the other scene's cameras
    behind its back; Blender's draw code then crashed walking the other
    scene's stale dependency graph (DEG_iterator_objects_next).
    """
    if scene is None:
        collection = bpy.data.collections.get(name)
        return collection if collection is not None else bpy.data.collections.new(name)
    mine = _sr_scene_collections(scene)
    for collection in mine:
        if collection.get(_SR_ROLE) == name and not _sr_used_elsewhere(collection, scene):
            return collection
    # A collection from an earlier version: adopt it when no other scene uses it.
    for collection in mine:
        if collection.name == name and not _sr_used_elsewhere(collection, scene):
            collection[_SR_ROLE] = name
            return collection
    existing = bpy.data.collections.get(name)
    if (existing is not None and existing.get(_SR_ROLE) in (None, name)
            and not any(existing in _sr_scene_collections(s) for s in bpy.data.scenes)):
        existing[_SR_ROLE] = name
        return existing
    collection = bpy.data.collections.new(name)
    collection[_SR_ROLE] = name
    return collection


def _sr_remove_objects(objects):
    """Delete objects safely, with their now unused data.

    An object may also be linked in another scene. Blender does not
    re-evaluate a scene no window shows, and its periodic GPU cache cleanup
    then walks that scene's dependency graph into the freed object. Every
    other scene that showed a removed object is updated right away.
    """
    current = getattr(bpy.context, "scene", None)
    others = set()
    removed = 0
    for obj in list(objects):
        try:
            for scene in obj.users_scene:
                if scene != current:
                    others.add(scene)
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        except (ReferenceError, RuntimeError):
            continue
        try:
            if data is not None and data.users == 0 and isinstance(data, bpy.types.Camera):
                bpy.data.cameras.remove(data)
        except (ReferenceError, RuntimeError):
            pass
    for scene in others:
        try:
            for view_layer in scene.view_layers:
                view_layer.update()
        except (ReferenceError, RuntimeError):
            pass
    return removed


def _sr_link_once(parent, child):
    if child.name not in {existing.name for existing in parent.children}:
        try:
            parent.children.link(child)
        except RuntimeError:
            return False
    return True


def _sr_get_or_create_collection(scene, name):
    """Return a managed collection, placed at its correct depth.

    The two content collections live inside Training Data so artists can hide
    the source data and trained Splats independently.
    """
    collection = _sr_collection(name, scene)
    parent_name = _SR_COLLECTION_PARENTS.get(name)
    parent = (
        _sr_get_or_create_collection(scene, parent_name)
        if parent_name
        else scene.collection
    )

    def unlink_wrong_parent(candidate):
        for child in tuple(candidate.children):
            if child is collection:
                if candidate is not parent:
                    try:
                        candidate.children.unlink(collection)
                    except RuntimeError:
                        pass
                continue
            try:
                unlink_wrong_parent(child)
            except ReferenceError:
                pass

    unlink_wrong_parent(scene.collection)
    _sr_link_once(parent, collection)
    return collection




# 17. STEP 3 — POINT CLOUD
# ═══════════════════════════════════════════════════════════════════════


def _sr_camera_forward_from_frame(frame):
    matrix = list_to_matrix(frame["world_matrix"])
    forward = -(matrix.to_3x3() @ Vector((0.0, 0.0, 1.0)))
    return matrix.to_translation(), forward.normalized()


def _sr_scene_sampling_bounds(frames, visibility_state):
    """World-space bounds of geometry that can contribute to the render."""
    corners = []
    for obj, _hide_viewport, _hidden_view_layer in (
            (visibility_state or {}).get("objects", ())):
        try:
            matrix = obj.matrix_world
            corners.extend(matrix @ Vector(corner) for corner in obj.bound_box)
        except (AttributeError, ReferenceError, RuntimeError):
            continue
    if not corners:
        corners = [_sr_camera_forward_from_frame(frame)[0] for frame in frames]
    minimum = Vector((min(point.x for point in corners),
                      min(point.y for point in corners),
                      min(point.z for point in corners)))
    maximum = Vector((max(point.x for point in corners),
                      max(point.y for point in corners),
                      max(point.z for point in corners)))
    span = maximum - minimum
    longest = max(span.x, span.y, span.z, 1e-4)
    padding = longest * 0.02
    return (minimum - Vector((padding,) * 3),
            maximum + Vector((padding,) * 3))


def _sr_scene_grid(bounds, axis_cells):
    """Create an aspect-aware 3D grid over the renderable scene bounds."""
    minimum, maximum = bounds
    span = maximum - minimum
    longest = max(span.x, span.y, span.z, 1e-4)
    dimensions = tuple(
        max(1, min(axis_cells,
                   int(math.ceil(axis_cells * max(component, 1e-4) / longest))))
        for component in span)
    step = Vector((
        max(span.x / dimensions[0], 1e-4),
        max(span.y / dimensions[1], 1e-4),
        max(span.z / dimensions[2], 1e-4),
    ))
    radius = step.length * 0.5
    cells = []
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                center = minimum + Vector(((x + 0.5) * step.x,
                                           (y + 0.5) * step.y,
                                           (z + 0.5) * step.z))
                cells.append(((x, y, z), center, radius))
    return dimensions, step, cells


def _sr_camera_contributes_to_cell(frame, center, radius, resolution,
                                   view=None, half_angle=None):
    """Conservative cached frustum test used before back-projection starts."""
    position, forward = view or _sr_camera_forward_from_frame(frame)
    delta = center - position
    distance = delta.length
    if distance <= max(radius, 1e-6):
        return True
    ray = delta / distance
    if half_angle is None:
        width, height = resolution
        fx = max(1e-6, float(frame["fl_x"]))
        fy = max(1e-6, float(frame["fl_y"]))
        half_angle = max(math.atan(width * 0.5 / fx),
                         math.atan(height * 0.5 / fy))
    allowance = math.asin(min(0.95, radius / distance))
    return forward.dot(ray) >= math.cos(
        min(math.pi * 0.49, half_angle + allowance))


def _sr_point_selection_target(cfg, camera_count):
    usage = max(1.0, min(100.0, float(cfg.point_camera_usage)))
    target = int(math.floor(camera_count * usage / 100.0 + 0.5))
    return usage, max(1, min(camera_count, target)) if camera_count else 0


_SR_BASE_RAYS = 12000
#: Above this many points SplatGen warns that the seed is very large for the
#: trainer. It is never a cap: every point the settings produce is written.
_SR_POINT_WARNING = bd_pointcloud.WARNING_POINTS
#: Share of cast rays expected to become seed points at merge strength 1.0
#: when the scene has no earlier generation to learn from.
_SR_DEFAULT_RETENTION = 0.55


def _sr_budget_sampling(cfg, weighted_cameras):
    """Sampling density and merge strength, exactly as the user set them.

    Nothing is reduced to fit a point budget; a cloud over
    ``_SR_POINT_WARNING`` is reported instead of being trimmed.
    """
    density = max(0.1, float(cfg.point_sampling_density))
    merging = max(0.1, float(cfg.point_merging_strength))
    return density, merging


def _sr_expected_retention(cfg, merging):
    """``(fraction of cast rays expected to become points, learned?)``.

    Learned from this scene's last generation when there is one - that
    figure already includes background pixels and multi-view overlap - and
    scaled for a changed merge strength: merge cells grow with the strength,
    so the share kept falls with its square.
    """
    merging = max(0.1, float(merging))
    learned = float(getattr(cfg, "point_last_retention", 0.0) or 0.0)
    previous = float(getattr(cfg, "point_last_merging", 0.0) or 0.0)
    if learned > 0.0 and previous > 0.0:
        return min(1.0, learned * (previous / merging) ** 2), True
    return min(1.0, _SR_DEFAULT_RETENTION / (merging * merging)), False


def _sr_estimate_point_cloud(cfg, camera_count):
    """Predict the point cloud the current settings will produce.

    Mirrors the arithmetic in _sr_point_sampling_plan and
    _sr_point_selection_target so the figure shown before a build matches
    what the build does. Per-camera ray counts are weighted by scene
    coverage at generation time and the fused count depends on how much the
    views overlap, so this is an estimate, not a promise; it becomes more
    accurate after the first generation of a scene.
    """
    usage = max(1.0, min(100.0, float(cfg.point_camera_usage)))
    selected = int(math.floor(camera_count * usage / 100.0 + 0.5))
    selected = max(1, min(camera_count, selected)) if camera_count else 0
    density, merging = _sr_budget_sampling(cfg, selected)
    rays_each = max(64, int(round(_SR_BASE_RAYS * density)))
    raw = selected * rays_each
    retained, learned = _sr_expected_retention(cfg, merging)
    final = int(round(raw * retained))
    return {
        "cameras_total": int(camera_count),
        "cameras_used": int(selected),
        "usage_percent": usage,
        "rays_per_camera": int(rays_each),
        "raw_samples": int(raw),
        "retained_fraction": retained,
        "retention_learned": learned,
        "final_points": final,
        "warning_points": _SR_POINT_WARNING,
        "exceeds_warning": final > _SR_POINT_WARNING,
        "effective_density": density,
    }


def _sr_large_point_cloud_message(count, estimated=False):
    """The one wording used everywhere a cloud goes over the warning size."""
    what = "is estimated at about" if estimated else "has"
    return (
        f"The point cloud {what} {int(count):,} points, more than "
        f"{_SR_POINT_WARNING:,}. This is too much for comfortable training: "
        "expect slow loading, heavy VRAM use and slower steps. Lower Rays "
        "per Camera or Cameras Used, or raise Sample Merging. Nothing is "
        "capped - every point is kept."
    )


def _sr_show_large_point_cloud_warning(context, count):
    """Pop up the size warning so it is seen, not only logged."""
    if bpy.app.background:
        return
    import textwrap

    lines = textwrap.wrap(_sr_large_point_cloud_message(count), 70)

    def draw(menu, _context):
        for line in lines:
            menu.layout.label(text=line)

    try:
        context.window_manager.popup_menu(
            draw, title="Point cloud over 2,000,000 points", icon='ERROR')
    except (AttributeError, ReferenceError, RuntimeError, TypeError):
        pass


def _sr_direction_bin(vector, bin_size):
    """Quantized spherical direction key for scalable angle clustering."""
    azimuth = math.atan2(vector.y, vector.x)
    elevation = math.asin(max(-1.0, min(1.0, vector.z)))
    return (int(math.floor((azimuth + math.pi) / bin_size)),
            int(math.floor((elevation + math.pi * 0.5) / bin_size)))


def _sr_view_diversity(first, second, frame_views, scene_diagonal):
    position_a, forward_a = frame_views[first]
    position_b, forward_b = frame_views[second]
    spatial = min(1.5, (position_a - position_b).length
                  / max(scene_diagonal, 1e-6))
    directional = 0.5 * (1.0 - max(-1.0, min(1.0,
                                           forward_a.dot(forward_b))))
    return spatial + directional


def _sr_farthest_view_order(frame_views, scene_diagonal):
    """Deterministic view-diverse order used to break equal coverage scores."""
    if not frame_views:
        return []
    try:
        import numpy as np
        positions = np.asarray([tuple(view[0]) for view in frame_views],
                               dtype=np.float64)
        forwards = np.asarray([tuple(view[1]) for view in frame_views],
                              dtype=np.float64)
        centroid = positions.mean(axis=0)
        first = int(np.argmax(np.linalg.norm(
            positions - centroid, axis=1)))
        remaining = np.ones(len(frame_views), dtype=bool)
        remaining[first] = False
        order = [first]

        def diversity_to(chosen):
            spatial = np.minimum(
                1.5,
                np.linalg.norm(positions - positions[chosen], axis=1)
                / max(scene_diagonal, 1e-6))
            directional = 0.5 * (1.0 - np.clip(
                forwards @ forwards[chosen], -1.0, 1.0))
            return spatial + directional

        minimum = diversity_to(first)
        while len(order) < len(frame_views):
            scores = np.where(remaining, minimum, -np.inf)
            chosen = int(np.argmax(scores))
            order.append(chosen)
            remaining[chosen] = False
            minimum = np.minimum(minimum, diversity_to(chosen))
        return order
    except (ImportError, RuntimeError, ValueError):
        centroid = sum((view[0] for view in frame_views), Vector())
        centroid /= len(frame_views)
        first = max(
            range(len(frame_views)),
            key=lambda index: (
                (frame_views[index][0] - centroid).length, -index))
        order = [first]
        remaining = set(range(len(frame_views)))
        remaining.remove(first)
        minimum = {
            index: _sr_view_diversity(
                index, first, frame_views, scene_diagonal)
            for index in remaining
        }
        while remaining:
            chosen = max(
                remaining, key=lambda index: (minimum[index], -index))
            order.append(chosen)
            remaining.remove(chosen)
            for index in remaining:
                minimum[index] = min(
                    minimum[index],
                    _sr_view_diversity(
                        index, chosen, frame_views, scene_diagonal))
        return order


def _sr_select_point_sampling_frames(frames, cfg, resolution,
                                     visibility_state=None):
    """Select the requested camera percentage before any ray is cast.

    Renderable scene cells and camera-position/direction groups become compact
    coverage tokens. A lazy greedy set-cover pass preserves as many useful
    cells and view groups as the requested camera budget permits. Remaining
    slots are filled by farthest-view sampling, never by random choice.
    """
    usage, target_count = _sr_point_selection_target(cfg, len(frames))
    if not frames:
        return [], {
            "similar": 0, "cells": 0, "grid": (0, 0, 0),
            "weight_min": 0.0, "weight_max": 0.0,
            "target": 0, "usage": usage, "scene_volume": 0.0,
            "scene_diagonal": 0.0,
        }

    bounds = _sr_scene_sampling_bounds(frames, visibility_state)
    grid_axis = max(4, min(10, int(round(
        4.0 + math.sqrt(max(1, target_count)) * 0.45))))
    dimensions, step, grid_cells = _sr_scene_grid(bounds, grid_axis)
    frame_views = [_sr_camera_forward_from_frame(frame) for frame in frames]
    width, height = resolution
    frame_half_angles = [
        max(math.atan(width * 0.5 / max(1e-6, float(frame["fl_x"]))),
            math.atan(height * 0.5 / max(1e-6, float(frame["fl_y"]))))
        for frame in frames
    ]
    camera_tokens = [set() for _frame in frames]
    tokens = []
    active_cells = 0
    try:
        import numpy as np
        positions_array = np.asarray(
            [tuple(view[0]) for view in frame_views], dtype=np.float64)
        forwards_array = np.asarray(
            [tuple(view[1]) for view in frame_views], dtype=np.float64)
        half_angles_array = np.asarray(
            frame_half_angles, dtype=np.float64)
    except (ImportError, RuntimeError, ValueError):
        np = None
        positions_array = forwards_array = half_angles_array = None

    # Scene-cell tokens reward cameras that cover renderable space.
    for _key, center, radius in grid_cells:
        if np is not None:
            delta = np.asarray(tuple(center)) - positions_array
            distance = np.linalg.norm(delta, axis=1)
            safe_distance = np.maximum(distance, 1e-12)
            rays = delta / safe_distance[:, None]
            facing = np.einsum("ij,ij->i", forwards_array, rays)
            allowance = np.arcsin(
                np.minimum(0.95, radius / safe_distance))
            threshold = np.cos(np.minimum(
                math.pi * 0.49, half_angles_array + allowance))
            mask = ((distance <= max(radius, 1e-6))
                    | (facing >= threshold))
            indices = np.flatnonzero(mask).tolist()
        else:
            indices = [
                index for index, frame in enumerate(frames)
                if _sr_camera_contributes_to_cell(
                    frame, center, radius, resolution,
                    frame_views[index], frame_half_angles[index])
            ]
        if not indices:
            continue
        active_cells += 1
        token_index = len(tokens)
        member_set = set(indices)
        tokens.append(member_set)
        for index in member_set:
            camera_tokens[index].add(token_index)

    # Camera-distribution tokens group nearby cameras with equivalent viewing
    # directions. Different angles inside one position cell remain separate.
    positions = [view[0] for view in frame_views]
    camera_minimum = Vector((
        min(position.x for position in positions),
        min(position.y for position in positions),
        min(position.z for position in positions),
    ))
    camera_maximum = Vector((
        max(position.x for position in positions),
        max(position.y for position in positions),
        max(position.z for position in positions),
    ))
    camera_span = camera_maximum - camera_minimum
    camera_axis = max(2, min(12, int(math.ceil(
        max(1, target_count) ** (1.0 / 3.0) * 1.5))))
    direction_size = math.radians(12.0)

    def camera_cell(position):
        values = []
        for value, minimum, span in zip(
                position, camera_minimum, camera_span):
            if abs(span) <= 1e-8:
                values.append(0)
            else:
                values.append(min(
                    camera_axis - 1,
                    int((value - minimum) / span * camera_axis)))
        return tuple(values)

    view_groups = {}
    for index, (position, forward) in enumerate(frame_views):
        key = (camera_cell(position),
               _sr_direction_bin(forward, direction_size))
        view_groups.setdefault(key, set()).add(index)
    for member_set in view_groups.values():
        token_index = len(tokens)
        tokens.append(member_set)
        for index in member_set:
            camera_tokens[index].add(token_index)


    if target_count >= len(frames):
        selected = set(range(len(frames)))
    else:
        import heapq
        scene_diagonal = max((bounds[1] - bounds[0]).length, 1e-6)
        diversity_order = _sr_farthest_view_order(
            frame_views, scene_diagonal)
        diversity_rank = {index: rank
                          for rank, index in enumerate(diversity_order)}
        selected = set()
        uncovered = set(range(len(tokens)))
        scores = [len(memberships) for memberships in camera_tokens]
        heap = [(-scores[index], diversity_rank[index], index)
                for index in range(len(frames))]
        heapq.heapify(heap)
        while heap and uncovered and len(selected) < target_count:
            negative_estimate, _rank, index = heapq.heappop(heap)
            if index in selected:
                continue
            if -negative_estimate != scores[index]:
                heapq.heappush(
                    heap, (-scores[index], diversity_rank[index], index))
                continue
            if scores[index] <= 0:
                break
            selected.add(index)
            newly_covered = camera_tokens[index] & uncovered
            uncovered.difference_update(newly_covered)
            # Each token is removed once, so score maintenance scales with the
            # membership table instead of repeatedly intersecting every camera.
            for token_index in newly_covered:
                for member in tokens[token_index]:
                    if member not in selected:
                        scores[member] -= 1


        # Once all coverage tokens are represented, fill any remaining budget
        # with the cameras farthest in position and viewing direction.
        candidates = set(range(len(frames))) - selected
        minimum_diversity = {}
        for index in candidates:
            minimum_diversity[index] = min(
                (_sr_view_diversity(index, chosen, frame_views, scene_diagonal)
                 for chosen in selected),
                default=float("inf"))
        while candidates and len(selected) < target_count:
            chosen = max(
                candidates,
                key=lambda index: (minimum_diversity[index],
                                   len(camera_tokens[index]), -index))
            selected.add(chosen)
            candidates.remove(chosen)
            for index in candidates:
                minimum_diversity[index] = min(
                    minimum_diversity[index],
                    _sr_view_diversity(
                        index, chosen, frame_views, scene_diagonal))

    selected_token_overlap = [
        max(1, len(members & selected)) for members in tokens
    ]
    weights = {}
    for index in selected:
        evidence = []
        for token_index in camera_tokens[index]:
            members = tokens[token_index]
            selected_overlap = selected_token_overlap[token_index]
            uniqueness = 1.0 / math.sqrt(max(1, len(members)))
            preservation = 1.0 / selected_overlap
            evidence.append(0.70 * uniqueness + 0.30 * preservation)
        if evidence:
            importance = (0.65 * (sum(evidence) / len(evidence))
                          + 0.35 * max(evidence))
            weights[index] = max(0.45, min(1.6, 0.50 + 1.10 * importance))
        else:
            weights[index] = 0.75

    selected_frames = []
    for index in sorted(selected):
        frame = dict(frames[index])
        frame["_sr_sample_weight"] = weights[index]
        selected_frames.append(frame)
    values = list(weights.values()) or [1.0]
    scene_span = bounds[1] - bounds[0]
    scene_volume = max(
        abs(scene_span.x * scene_span.y * scene_span.z), 1e-12)
    stats = {
        "similar": len(frames) - len(selected_frames),
        "cells": active_cells,
        "grid": dimensions,
        "cell_size": max(step),
        "weight_min": min(values),
        "weight_max": max(values),
        "scene_volume": scene_volume,
        "scene_diagonal": scene_span.length,
        "target": target_count,
        "usage": usage,
        "tokens": len(tokens),
    }
    return selected_frames, stats


def _sr_point_sampling_plan(frames, resolution, cfg):
    """Per-camera sample grids and the merge strength used by generation."""
    sampling_density, merging_strength = _sr_budget_sampling(cfg, len(frames))
    grids, offsets = [], [0]
    for frame in frames:
        weight = max(0.35, min(
            1.6, float(frame.get("_sr_sample_weight", 1.0))))
        grid = bd_pointcloud.grid_for_rays(
            _SR_BASE_RAYS * sampling_density * weight,
            resolution[0], resolution[1])
        grids.append(grid)
        offsets.append(offsets[-1] + grid[2])
    return {
        "grids": grids,
        "offsets": offsets,
        "total": offsets[-1],
        "merge_scale": merging_strength,
        "base_rays": _SR_BASE_RAYS,
        "sampling_density": sampling_density,
        "merging_strength": merging_strength,
    }


def _sr_preflight_frames(context, cfg):
    scene = context.scene
    resolution = effective_resolution(scene.render)
    frames = []
    live = _sr_live_camera_map(scene)
    for item in cfg.camera_queue:
        camera = item.camera
        if not _sr_camera_matches_live_map(camera, live):
            continue
        pose, _scaled = camera_pose_world(camera)
        fx, fy, cx, cy, angle = compute_intrinsics(
            camera.data, scene.render)
        frames.append({
            "camera_name": camera.name,
            "world_matrix": matrix_to_list(pose),
            "transform_matrix": matrix_to_list(pose),
            "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
            "camera_angle_x": angle,
            "file_path": "",
        })
    return frames, resolution


class _SceneRaySplatPointSampler:
    """Progressive RGB-D back-projector that yields to Blender between views.

    Each camera is one vectorised pass through ``building_data.pointcloud``:
    samples at exact pixel centres, edge and floater detection on the depth
    map, and a per-sample footprint for the adaptive fusion that follows.
    The modal loop processes whole cameras until its time slice is spent.
    """

    _KEYS = ("xyz", "rgb", "normal", "spacing", "edge", "camera")

    def __init__(self, scene, frames, resolution, output_dir, export_matrix,
                 cfg, visibility_state=None):
        import numpy as np
        self.scene, self.frames, self.resolution = scene, frames, resolution
        self.output_dir, self.export_matrix, self.cfg = Path(output_dir), export_matrix, cfg
        self.guided = bd_export.ground_truth_enabled(cfg)
        plan = _sr_point_sampling_plan(frames, resolution, cfg)
        self.frame_grids = plan["grids"]
        self.frame_offsets = plan["offsets"]
        self.total = plan["total"]
        self.merge_scale = plan["merge_scale"]
        self.camera_index = 0
        export = np.array([[export_matrix[r][c] for c in range(4)]
                           for r in range(4)], dtype=np.float64)
        self._export_linear = export[:3, :3]
        # Footprints are lengths: carry them into the export frame as well.
        self._export_scale = abs(float(np.linalg.det(export[:3, :3]))) ** (
            1.0 / 3.0) or 1.0
        self._chunks = {key: [] for key in self._KEYS}
        self._unknown_material = _sr_material_guidance(None)
        self.material_records = [dict(
            self._unknown_material, id=0, linked_inputs=[]
        )]
        self.object_records = [{
            "id": 0, "name": "<Unknown Object>", "type": "UNKNOWN",
            "library": "", "matrix_world": matrix_to_list(Matrix.Identity(4)),
        }]
        self.stats = {
            "cams_used": 0,
            "skipped": [],
            "cam_pos": [],
            "ray_grids": list(self.frame_grids),
            "invalid_depth_samples": 0,
            "edge_samples": 0,
            "floaters_removed": 0,
            "raw_samples": 0,
        }

    def _load_camera(self, fr):
        """RGB, metric depth, far clip and (when exported) world normals."""
        path = self.output_dir / fr.get(
            "training_file_path",
            fr.get("file_path", ""),
        ).lstrip("./")
        rgb = None
        if path.is_file():
            try:
                rgb = _read_image_rgb(path)
            except Exception as exc:
                print(f"[sceneray_splat] could not read {path.name}: {exc}")
        if rgb is None:
            raise ValueError(
                f"No rendered RGB image for camera '{fr['camera_name']}'. "
                "Render the images before generating the point cloud."
            )
        if tuple(rgb.shape[:2]) != (
                int(self.resolution[1]), int(self.resolution[0])):
            raise ValueError(
                f"RGB dimensions for camera '{fr['camera_name']}' are "
                f"{rgb.shape[1]}x{rgb.shape[0]}; expected "
                f"{self.resolution[0]}x{self.resolution[1]}."
            )

        frame_index = int(fr.get("frame_index", self.camera_index))
        signature = fr.get("signature") if isinstance(fr, dict) else None
        try:
            maximum_depth = float(
                (signature or {}).get(
                    "clip_end",
                    getattr(
                        getattr(bpy.data.objects.get(fr["camera_name"]), "data", None),
                        "clip_end",
                        bd_pointcloud.BACKGROUND_DEPTH,
                    ),
                )
            )
        except (TypeError, ValueError):
            maximum_depth = bd_pointcloud.BACKGROUND_DEPTH
        try:
            depth = bd_export.read_typed_pass(
                self.output_dir, "depth", frame_index, self.resolution
            )
        except Exception as exc:
            raise ValueError(
                f"No valid metric depth map for camera "
                f"'{fr['camera_name']}' (frame_{frame_index:04d}.exr). "
                "Render this camera again before generating the RGB-D point "
                f"cloud: {exc}"
            ) from exc

        normal = None
        if self.guided:
            try:
                normal = bd_export.read_typed_pass(
                    self.output_dir, "normal", frame_index, self.resolution
                )
            except Exception as exc:
                # Normals are derived from the depth map when the pass is
                # missing; depth stays authoritative for every position.
                print(
                    f"[sceneray_splat] no normal map for "
                    f"{fr['camera_name']}; using depth-derived normals: {exc}"
                )
            if normal is not None and (
                    normal.ndim != 3 or normal.shape[-1] < 3):
                normal = None
        return rgb, depth, normal, maximum_depth

    def _sample_camera(self):
        import numpy as np
        fr = self.frames[self.camera_index]
        columns, rows, _count = self.frame_grids[self.camera_index]
        rgb, depth, normal_map, maximum_depth = self._load_camera(fr)
        c2w = list_to_matrix(fr["world_matrix"])
        rotation = [[c2w[r][c] for c in range(3)] for r in range(3)]
        location = c2w.to_translation()
        samples, counts = bd_pointcloud.backproject_camera(
            depth, rgb, rotation, (location.x, location.y, location.z),
            (fr["fl_x"], fr["fl_y"], fr["cx"], fr["cy"]),
            (columns, rows), self.camera_index,
            max_depth=maximum_depth, normal_map=normal_map,
        )
        del rgb, depth, normal_map
        self.stats["invalid_depth_samples"] += counts["background"]
        self.stats["edge_samples"] += counts["edges"]
        self.stats["floaters_removed"] += counts["floaters"]
        if not counts["kept"]:
            self.stats["skipped"].append(
                (fr["camera_name"], "no finite rendered depth (sky-only view)"))
            return
        samples["xyz"] = apply_export_transform_to_points(
            self.export_matrix, samples["xyz"])
        normals = samples["normal"].astype(np.float64) @ self._export_linear.T
        normals /= np.maximum(
            np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
        samples["normal"] = normals.astype(np.float32)
        samples["spacing"] = (
            samples["spacing"] * self._export_scale).astype(np.float32)
        for key in self._KEYS:
            self._chunks[key].append(samples[key])
        self.stats["cams_used"] += 1
        pos = list_to_matrix(fr["transform_matrix"]).to_translation()
        self.stats["cam_pos"].append((pos.x, pos.y, pos.z))

    def step(self, seconds=0.03):
        deadline = time.perf_counter() + seconds
        while self.camera_index < len(self.frames):
            self._sample_camera()
            self.camera_index += 1
            if time.perf_counter() >= deadline:
                break
        return self.camera_index >= len(self.frames)

    @property
    def progress(self):
        return self.frame_offsets[min(self.camera_index, len(self.frames))]

    def result(self):
        """``(samples, stats, metadata)``; hands the samples over and forgets
        them so only the finalize worker holds that memory."""
        import numpy as np
        if not self._chunks["xyz"]:
            return None, self.stats, None
        samples = {
            key: np.concatenate(chunks) for key, chunks in self._chunks.items()
        }
        self._chunks = {key: [] for key in self._KEYS}
        self.stats["raw_samples"] = int(samples["xyz"].shape[0])
        metadata = None
        if self.guided:
            metadata = _sr_scene_guidance_metadata(self.scene)
            metadata.update({
                "objects": self.object_records,
                "materials": self.material_records,
                "export_from_world": matrix_to_list(self.export_matrix),
                "resolution": list(self.resolution),
                "units": "Blender scene units",
                "unit_scale_length": float(self.scene.unit_settings.scale_length),
                "sampling": {
                    "method": "rgbd_metric_depth_backprojection",
                    "fusion": "footprint_adaptive_finest_first",
                    "quality": str(getattr(self.cfg, "point_quality", "CUSTOM")),
                    "density": float(getattr(self.cfg, "point_sampling_density", 1.0)),
                    "camera_usage_percent": float(
                        getattr(self.cfg, "point_camera_usage", 100.0)
                    ),
                    "merge_strength": float(self.merge_scale),
                    "raw_samples": int(samples["xyz"].shape[0]),
                    "edge_samples": int(self.stats["edge_samples"]),
                    "floaters_removed": int(self.stats["floaters_removed"]),
                },
                "limitations": [
                    "Point existence and placement come from the rendered metric depth map.",
                    "Normals come from the rendered normal pass, with depth-derived fallbacks.",
                    "Object and material properties are unknown without per-pixel index mapping.",
                ],
            })
        return samples, self.stats, metadata


def _sr_finalize_point_cloud_worker(
    state,
    cancel_event,
    output_dir,
    samples,
    merge_scale,
    guided=False,
    guidance_metadata=None,
    unknown_material=None,
):
    """Fuse and write outside Blender's UI thread; publish atomically.

    No point limit is applied: every fused point is written. ``state``
    reports ``n_points`` as soon as fusion knows it, before the slow write,
    so the interface can warn about a very large cloud straight away.
    """
    import numpy as np
    output_dir = Path(output_dir)
    generation_id = uuid.uuid4().hex
    temporary = (
        _work_dir_path(output_dir)
        / f"points3D.{generation_id}.pending.txt"
    )
    guidance_temporary = (
        _work_dir_path(output_dir) / f"surface_seed.{generation_id}.pending.npz"
    )
    manifest_temporary = (
        _work_dir_path(output_dir) / f"scene_guidance.{generation_id}.pending.json"
    )
    try:
        state["phase"] = "fusing"

        def fusing(fraction):
            state["fraction"] = fraction

        fused = bd_pointcloud.fuse_samples(
            samples["xyz"], samples["rgb"], samples["normal"],
            samples["spacing"], samples["edge"], samples["camera"],
            merge_strength=merge_scale, cancelled=cancel_event.is_set,
            report=fusing,
        )
        state["raw_samples"] = int(samples["xyz"].shape[0])
        # The raw samples are the largest thing in memory; the fused cloud
        # is all that is needed from here on.
        samples.clear()
        if fused is None or cancel_event.is_set():
            state["cancelled"] = True
            return
        seed_xyz, seed_rgb = fused["xyz"], fused["rgb"]
        n_points = int(seed_xyz.shape[0])
        if n_points == 0:
            raise ValueError("Fusion produced no points.")
        state["n_points"] = n_points
        state["levels"] = int(fused["levels"])
        state["phase"] = "writing"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with open(temporary, "w", encoding="utf-8") as handle:
            finished = bd_pointcloud.write_points3d(
                handle, seed_xyz, seed_rgb,
                header_lines=(f"SplatGen generation: {generation_id}",),
                cancelled=cancel_event.is_set,
            )
        if not finished or cancel_event.is_set():
            state["cancelled"] = True
            return

        if guided:
            material = unknown_material or _sr_material_guidance(None)

            def constant(value, dtype):
                value = np.asarray(value, dtype=dtype)
                return np.repeat(value[None, ...], n_points, axis=0)

            retained = {
                "normal": fused["normal"].astype(np.float32),
                "radius": fused["radius"].astype(np.float32),
                "object_id": np.zeros(n_points, dtype=np.int32),
                "material_id": np.zeros(n_points, dtype=np.int32),
                "base_color": constant(material["base_color"], np.float16),
                "roughness": constant(material["roughness"], np.float16),
                "metallic": constant(material["metallic"], np.float16),
                "transmission": constant(material["transmission"], np.float16),
                "alpha": constant(material["alpha"], np.float16),
                "material_valid_mask": constant(material["valid_mask"], np.uint8),
                "support": fused["support"].astype(np.int32),
                "edge_only": fused["edge_only"].astype(np.uint8),
                "position": seed_xyz.astype(np.float32),
                "appearance_rgb": seed_rgb.astype(np.uint8),
            }
            guidance_temporary.parent.mkdir(parents=True, exist_ok=True)
            np.savez(guidance_temporary, **retained)
            digest = hashlib.sha256()
            with open(guidance_temporary, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            manifest = dict(guidance_metadata or {})
            manifest.update({
                "format": "SplatGen Scene Guidance",
                "schema_version": 1,
                "generation_id": generation_id,
                "coordinate_frame": "SplatGen export frame",
                "point_count": int(n_points),
                "seed_file": bd_paths.SCENE_GUIDANCE_SEED,
                "seed_sha256": digest.hexdigest(),
                "arrays": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in retained.items()
                },
            })
            manifest_temporary.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            if cancel_event.is_set():
                state["cancelled"] = True
                return

        # Publish only after every requested artifact is complete. The compact
        # guidance manifest is the final commit marker for the enriched seed.
        _sr_data_dir(output_dir).mkdir(parents=True, exist_ok=True)
        if guided:
            bd_paths.scene_guidance_dir(output_dir).mkdir(
                parents=True, exist_ok=True
            )
            guidance_temporary.replace(bd_paths.scene_guidance_seed(output_dir))
        temporary.replace(_sr_dataset_file(output_dir, "points3D.txt"))
        if guided:
            manifest_temporary.replace(
                bd_paths.scene_guidance_manifest(output_dir)
            )
            state["guidance_points"] = int(n_points)
        else:
            guidance_dir = bd_paths.scene_guidance_dir(output_dir)
            if guidance_dir.is_dir():
                shutil.rmtree(guidance_dir)
        _clean_obsolete_dataset_layout(output_dir)
        _cleanup_work_directory(output_dir)
    except Exception as exc:
        state["error"] = str(exc)
    finally:
        for pending in (temporary, guidance_temporary, manifest_temporary):
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                pass
        state["done"] = True


class SCENERAY_SPLAT_OT_generate_points3d(bpy.types.Operator):
    """STEP 3 — Sample a coloured point cloud and write root points3D.txt.

    Runs progressively so Blender stays responsive; press Esc or Stop to
    cancel without replacing anything"""
    bl_idname = "sceneray_splat.generate_points3d"
    bl_label = "Generate Point Cloud"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return (cfg is not None and not cfg.is_rendering
                and not cfg.is_generating_points
                and (_sr_output_base_path(cfg) is not None))

    def _prepare(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        output_dir = _sr_effective_output_path(cfg)
        if output_dir is None:
            raise ValueError("Choose an output folder before building.")
        ds = read_dataset_manifest(output_dir)
        if ds is not None and ds.get("frames"):
            frames = ds["frames"]
            resolution = tuple(ds["resolution"])
            export_matrix = list_to_matrix(ds["export_from_world"])
            for fr in frames:
                cam = bpy.data.objects.get(fr["camera_name"])
                if cam and cam.type == 'CAMERA':
                    live, _ = camera_pose_world(cam)
                    moved = (live.to_translation()
                             - list_to_matrix(fr["world_matrix"]).to_translation())
                    if moved.length > 1e-4:
                        raise ValueError(
                            f"Camera '{cam.name}' moved after its image was "
                            "rendered. Render it again, then rewrite the "
                            "camera data.")
        else:
            raise ValueError(
                "The RGB-D point cloud needs rendered images, metric depth "
                "maps, and camera data. Run Build Dataset (or Render Images) "
                "before generating the point cloud."
            )
        self.cfg, self.output_dir = cfg, output_dir
        self.frames = frames
        self.resolution = resolution
        self.export_matrix = export_matrix
        self.vis_state = snapshot_and_reveal_render_geometry(context, cfg)
        try:
            sampling_frames, reduction = _sr_select_point_sampling_frames(
                frames, cfg, tuple(resolution), self.vis_state)
            if not sampling_frames:
                raise ValueError(
                    "No camera views remain after point-cloud camera reduction.")
            grid = "x".join(str(value) for value in reduction["grid"])
            self.selection_summary = (
                f"Smart selection chose {len(sampling_frames)}/"
                f"{len(frames)} cameras "
                f"({reduction['usage']:.0f}% requested) before RGB-D "
                f"back-projection; "
                f"{reduction['similar']} excluded across "
                f"{reduction['cells']} scene cells ({grid} grid).")
            self.sampler = _SceneRaySplatPointSampler(
                context.scene, sampling_frames, tuple(resolution),
                output_dir, export_matrix, cfg, self.vis_state)
            retained, _learned = _sr_expected_retention(
                cfg, self.sampler.merge_scale)
            self.estimated_points = int(round(self.sampler.total * retained))
        except Exception:
            restore_geometry_visibility(context, self.vis_state)
            self.vis_state = None
            raise

    def _begin(self, context):
        """Shared setup for the interactive and background paths."""
        self._prepare(context)
        cfg = self.cfg
        cfg.is_generating_points = True
        cfg.point_progress = 0
        cfg.point_total = self.sampler.total
        cfg.point_progress_fac = 0.0
        cfg.point_eta = ""
        cfg.point_status = getattr(
            self, "selection_summary", "Preparing camera samples"
        )
        _cancel_flag["requested"] = False
        self._finalizing = False
        self._point_worker = None
        self._point_worker_state = None
        self._point_cancel_event = None
        self._point_stats = None
        self._timer = None
        self._t0 = time.time()
        workflow = str(getattr(cfg, "build_workflow", "")) or "POINTS"
        if workflow != 'FULL':
            cfg.build_workflow = 'POINTS'
            workflow = 'POINTS'
        # Build Dataset announces both stages up front and starts on this one,
        # so the bar never restarts when rendering takes over.
        if not progress.is_active():
            progress.begin(
                BUILD_WORKFLOW_LABELS.get(workflow, "Generate Point Cloud"),
                _sr_workflow_stages(cfg.id_data, workflow, (PHASE_POINTS,)),
            )
        progress.set_stage(PHASE_POINTS)
        progress.update(fraction=0.0, message=cfg.point_status)
        self._size_warned = False
        if self.estimated_points > _SR_POINT_WARNING:
            # Said before the work starts; the exact count follows fusion.
            self.report({'WARNING'}, _sr_large_point_cloud_message(
                self.estimated_points, estimated=True))

    def invoke(self, context, event):
        if bpy.app.background:
            return self.execute(context)
        try:
            self._begin(context)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self._timer = context.window_manager.event_timer_add(
            0.02, window=context.window)
        context.window_manager.progress_begin(0, 100)
        context.window_manager.progress_update(0.0)
        context.window_manager.modal_handler_add(self)
        _tag_redraw_sceneray_splat(context)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        """Run to completion without a modal loop.

        Background Blender has no window to attach a timer to, so a scripted
        or command-line build drives the same sampler and finalizer straight
        through instead.
        """
        if not bpy.app.background:
            return self.invoke(context, None)
        try:
            self._begin(context)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        try:
            while not self.sampler.step():
                self.cfg.point_progress = self.sampler.progress
                self.cfg.point_progress_fac = (
                    self.sampler.progress / max(1, self.sampler.total)
                )
            self._start_write_result(context)
            while True:
                outcome = self._poll_write_result(context)
                if outcome != {'RUNNING_MODAL'}:
                    return outcome
                self._point_worker.join(0.05)
        except Exception as exc:
            self.cfg.point_status = f"Point-cloud generation failed: {exc}"
            self._cleanup(context)
            self.report({'ERROR'}, self.cfg.point_status)
            return {'CANCELLED'}

    def _cleanup(self, context, end_progress=True):
        self.cfg.build_workflow = 'NONE'
        if end_progress:
            progress.end(self.cfg.point_status)
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            # Only the modal path opens the progress bar.
            context.window_manager.progress_end()
        restore_geometry_visibility(context, self.vis_state)
        self.cfg.is_generating_points = False
        self.cfg.point_eta = ""
        _cancel_flag["requested"] = False
        _tag_redraw_sceneray_splat(context)

    def _start_write_result(self, context):
        samples, stats, guidance_metadata = self.sampler.result()
        if samples is None or stats["cams_used"] == 0:
            raise ValueError(
                "No points were reconstructed. Check that the rendered depth "
                "maps contain finite foreground depth."
            )
        self.cfg.point_status = (
            f"Fusing {stats['raw_samples']:,} samples from "
            f"{stats['cams_used']} camera(s)…"
        )
        self.cfg.point_eta = "Finalizing"
        progress.update(fraction=1.0, message="Fusing surface samples")
        _tag_redraw_sceneray_splat(context)
        self._point_stats = stats
        self._point_worker_state = {
            "done": False,
            "cancelled": False,
            "error": "",
            "phase": "fusing",
            "fraction": 0.0,
            "n_points": 0,
            "raw_samples": int(stats["raw_samples"]),
            "levels": 0,
            "guidance_points": 0,
        }
        self._point_cancel_event = threading.Event()
        self._point_worker = threading.Thread(
            target=_sr_finalize_point_cloud_worker,
            args=(self._point_worker_state, self._point_cancel_event,
                  str(self.output_dir), samples, self.sampler.merge_scale,
                  self.sampler.guided, guidance_metadata,
                  self.sampler._unknown_material),
            daemon=True,
            name="SceneRaySplatPointFinalize")
        del samples
        self._finalizing = True
        self._point_worker.start()

    def _poll_write_result(self, context):
        state = self._point_worker_state
        if _cancel_flag["requested"]:
            self._point_cancel_event.set()
            self.cfg.point_status = "Cancelling point-cloud finalization…"
        if not state["done"]:
            count = int(state.get("n_points", 0))
            if state.get("phase") == "writing" and count:
                self.cfg.point_status = f"Writing {count:,} points…"
                if count > _SR_POINT_WARNING:
                    self.cfg.point_status += f" (over {_SR_POINT_WARNING:,})"
                    if not self._size_warned:
                        # Known the moment fusion ends, well before the write.
                        self._size_warned = True
                        self.report({'WARNING'},
                                    _sr_large_point_cloud_message(count))
                progress.update(message=self.cfg.point_status)
            elif state.get("phase") == "fusing":
                progress.update(
                    message="Fusing surface samples",
                    detail=f"{state.get('fraction', 0.0) * 100.0:.0f}%",
                )
            _tag_redraw_sceneray_splat(context)
            return {'RUNNING_MODAL'}
        if state["cancelled"]:
            self.cfg.point_status = "Cancelled — no files were replaced."
            self._cleanup(context)
            self.report({'WARNING'}, self.cfg.point_status)
            return {'CANCELLED'}
        progress.update(fraction=1.0, message="Writing points3D.txt")
        if state["error"]:
            raise RuntimeError(state["error"])
        stats = self._point_stats
        self.cfg.point_progress = self.cfg.point_total
        self.cfg.point_progress_fac = 1.0
        self.cfg.point_eta = ""
        n_points = int(state["n_points"])
        # Remembered with the scene so the next estimate is learned, not
        # guessed.
        self.cfg.point_last_count = min(n_points, 2 ** 31 - 1)
        self.cfg.point_last_retention = n_points / max(1, self.sampler.total)
        self.cfg.point_last_merging = float(self.sampler.merge_scale)
        self.cfg.point_status = (
            f"Dataset complete — {n_points:,} points"
            f" from {stats['cams_used']} camera(s)"
        )
        if n_points > _SR_POINT_WARNING:
            self.cfg.point_status += f" · over {_SR_POINT_WARNING:,} points"
        try:
            export_root = _sr_finalize_dataset_export(self.cfg, self.output_dir)
            if export_root is not None:
                self.cfg.point_status += " · dataset verified"
        except bd_export.GroundTruthValidationError as exc:
            self.cfg.point_status += (
                " · canonical dataset complete; SplatGen metadata incomplete"
            )
            self.report({'WARNING'}, str(exc))
        except Exception as exc:
            raise RuntimeError(
                f"Point cloud was written, but dataset validation failed: {exc}"
            ) from exc
        _sr_record_completed_dataset(
            self.cfg.id_data,
            self.output_dir,
        )
        for name, why in stats["skipped"]:
            self.report({'WARNING'}, f"Camera '{name}' skipped: {why}")
        if n_points > _SR_POINT_WARNING:
            self.report({'WARNING'}, _sr_large_point_cloud_message(n_points))
            _sr_show_large_point_cloud_warning(context, n_points)
        else:
            self.report({'INFO'}, self.cfg.point_status)
        # _cleanup clears build_workflow, so what to do next is read first.
        workflow = str(getattr(self.cfg, "build_workflow", ""))
        follow_raw = raw_stage.should_run(context.scene, workflow)
        self._cleanup(context, end_progress=not follow_raw)

        # The point cloud is the last legacy phase. Build Dataset then hands
        # on to the raw export, which never modifies the legacy files.
        self.cfg.build_workflow = 'NONE'
        if follow_raw and raw_stage.start(
                context, self.output_dir,
                finished_message=self.cfg.point_status):
            _tag_redraw_sceneray_splat(context)
            return {'FINISHED'}
        progress.end(self.cfg.point_status)
        _tag_redraw_sceneray_splat(context)
        return {'FINISHED'}

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            _cancel_flag["requested"] = True
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if self._finalizing:
            try:
                return self._poll_write_result(context)
            except Exception as exc:
                self.cfg.point_status = f"Point-cloud generation failed: {exc}"
                self._cleanup(context)
                self.report({'ERROR'}, self.cfg.point_status)
                return {'CANCELLED'}
        if _cancel_flag["requested"]:
            self.cfg.point_status = ("Cancelled — no files were replaced.")
            self._cleanup(context)
            self.report({'WARNING'}, self.cfg.point_status)
            return {'CANCELLED'}
        try:
            complete = self.sampler.step()
            # Not named `progress`: that is the module every stage reports
            # through, and shadowing it here broke point-cloud generation.
            sampled = self.sampler.progress
            self.cfg.point_progress = sampled
            fraction = sampled / max(1, self.sampler.total)
            self.cfg.point_progress_fac = fraction
            camera_no = min(len(self.sampler.frames),
                            self.sampler.camera_index + 1)
            eta = ""
            if fraction > 0.01:
                elapsed = time.time() - self._t0
                self.cfg.point_eta = _fmt_eta(
                    elapsed / fraction - elapsed
                )
                eta = f" · ~{self.cfg.point_eta} left"
            else:
                self.cfg.point_eta = ""
            overall = sr_stage_progress(fraction)
            self.cfg.point_status = (
                f"Sampling camera {camera_no}/{len(self.sampler.frames)} "
                f"({overall:.0f}%){eta}")
            progress.update(
                fraction=fraction,
                message="Sampling surface points",
                detail=f"camera {camera_no}/{len(self.sampler.frames)}",
                eta=self.cfg.point_eta,
            )
            context.window_manager.progress_update(overall)
            _tag_redraw_sceneray_splat(context)
            if not complete:
                return {'RUNNING_MODAL'}
            self._start_write_result(context)
            return {'RUNNING_MODAL'}
        except Exception as exc:
            self.cfg.point_status = f"Point-cloud generation failed: {exc}"
            self._cleanup(context)
            self.report({'ERROR'}, self.cfg.point_status)
            return {'CANCELLED'}


class SCENERAY_SPLAT_OT_stop_render(bpy.types.Operator):
    """Cancel the active job promptly. Completed work remains resumable."""
    bl_idname = "sceneray_splat.stop_render"
    bl_label = "Stop"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        if cfg is not None and (cfg.is_rendering or cfg.is_generating_points):
            return True
        # Camera coverage and the Smart camera rig set neither flag, but
        # they are still operations the user needs to stop from this button.
        return (progress.is_owner(progress.OWNER_COVERAGE)
                or progress.is_owner(progress.OWNER_AUTO_RIG))

    def execute(self, context):
        _cancel_flag["requested"] = True
        from .auto_rig import coverage_ops, operators as auto_rig_operators

        coverage_ops.request_cancel()
        auto_rig_operators.request_cancel()
        cfg = context.scene.SCENERAY_SPLAT
        if cfg.is_rendering:
            # Do not inject Escape into Blender's temporary Render window.
            # In Blender 5.2 that can report render_cancel while the compositor
            # is still evaluating, and restoring our capture nodes then races
            # the compositor. Finish the active image and stop before another
            # camera starts; completed work remains resumable and safe.
            if _sr_render_job_running():
                cfg.render_status = "Stop requested - finishing the active image safely."
            else:
                cfg.render_status = "Stop requested - stopping before the next image."
            _sr_schedule_render_batch_tick(0.01)
            _sr_schedule_render_modal_wakeup()
        elif cfg.is_generating_points:
            cfg.point_status = "Stop requested — cancelling current sample batch…"
        _tag_redraw_sceneray_splat(context)
        self.report({'INFO'}, "Stop requested.")
        return {'FINISHED'}


class SCENERAY_SPLAT_OT_open_output(bpy.types.Operator):
    """Open the dataset folder in your file browser"""
    bl_idname = "sceneray_splat.open_output"
    bl_label = "Open Dataset Folder"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
        return cfg is not None and (_sr_output_base_path(cfg) is not None)

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        # The remembered build folder can be gone - superseded into a recovery
        # copy, moved, or never created. Fall back to the newest build that
        # does exist, then to the project root, rather than refusing to open
        # anything at all.
        path = _sr_effective_output_path(cfg)
        root = _sr_output_base_path(cfg)
        if path is None or not path.is_dir():
            path = bd_paths.latest_build(root) or root
        if path is None:
            self.report({'WARNING'}, "Choose a project folder first.")
            return {'CANCELLED'}
        if not path.is_dir():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.report(
                    {'WARNING'}, f"Could not open or create {path}: {exc}")
                return {'CANCELLED'}
        _sr_sync_render_states(context.scene, cfg, path)
        _render_status_sync["signatures"][cfg.as_pointer()] = _sr_render_status_signature(cfg)
        bpy.ops.wm.path_open(filepath=str(path))
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════
# 18. INTERFACE
# ═══════════════════════════════════════════════════════════════════════


class SCENERAY_SPLAT_UL_camera_queue(bpy.types.UIList):
    """One compact line per camera: name, render state, remove."""
    bl_idname = "SCENERAY_SPLAT_UL_camera_queue"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        camera = item.camera
        if camera is None or camera.type != 'CAMERA':
            broken = row.row(align=True)
            broken.alert = True
            broken.label(text="Missing camera", icon=theme.STATUS_ERROR)
        else:
            rendered = item.render_state == 'RENDERED'
            state_icon = theme.custom_icon(
                "camera_rendered" if rendered else "camera_pending"
            )
            if state_icon:
                row.label(text=camera.name, icon_value=state_icon)
            else:
                row.label(text=camera.name, icon='OUTLINER_OB_CAMERA')
            if rendered:
                sub = row.row(align=True)
                sub.emboss = 'NONE'
                theme.operator(sub, "sceneray_splat.mark_pending", text="", icon='FILE_REFRESH').index = index
        sub = row.row(align=True)
        sub.alignment = 'RIGHT'
        sub.emboss = 'NONE'
        theme.operator(sub, "sceneray_splat.queue_remove", text="", icon='X').index = index

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        flt_flags, flt_neworder = [], []
        try:
            helper = bpy.types.UI_UL_list
            if self.filter_name:
                pattern = self.filter_name.lower().strip("*")
                flt_flags = [
                    self.bitflag_filter_item
                    if (it.camera and pattern in it.camera.name.lower()) else 0
                    for it in items]
                if self.use_filter_invert:
                    flt_flags = [0 if f else self.bitflag_filter_item
                                 for f in flt_flags]
            if self.use_filter_sort_alpha:
                keyed = [(i, (it.camera.name.lower() if it.camera else ""))
                         for i, it in enumerate(items)]
                flt_neworder = helper.sort_items_helper(keyed, lambda e: e[1])
        except Exception:
            return [], []
        return flt_flags, flt_neworder


class SCENERAY_SPLAT_UL_rig_presets(bpy.types.UIList):
    """Bundled templates and custom rigs share one labelled list."""
    bl_idname = "SCENERAY_SPLAT_UL_rig_presets"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.name, icon='LOCKED' if item.builtin else 'BOOKMARKS')
        sub = row.row()
        sub.alignment = 'RIGHT'
        sub.label(text=f"{'Built-in' if item.builtin else 'Custom'} · {len(item.cameras)}")


def _sceneray_splat_progress(layout, cfg):
    """The unified report, showing only what is actually running."""
    progress.draw_panel(layout)


def _sr_progress_panel_poll(context):
    return progress.is_active()


class SCENERAY_SPLAT_PT_blender_render_progress(bpy.types.Panel):
    """Dataset progress inside Blender's Render Result image window."""

    bl_idname = "SCENERAY_SPLAT_PT_blender_render_progress"
    bl_label = "SplatGen Dataset Build"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Blender Render"
    bl_order = 0

    @classmethod
    def poll(cls, context):
        if not _sr_progress_panel_poll(context):
            return False
        image = getattr(getattr(context, "space_data", None), "image", None)
        return (
            image is None
            or getattr(image, "type", "") == 'RENDER_RESULT'
            or getattr(image, "name", "") == "Render Result"
        )

    def draw(self, context):
        layout = self.layout
        _sceneray_splat_progress(
            layout,
            context.scene.SCENERAY_SPLAT,
        )
        layout.separator()
        stop = layout.column()
        stop.scale_y = 1.35
        theme.operator(stop, 
            "sceneray_splat.stop_render",
            text="STOP DATASET BUILD",
            icon='CANCEL',
        )


def _sr_addon_preferences():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


def _sr_pipeline_section(layout, cfg, property_name, label, icon, *,
                         custom=None, detail=None, state=None):
    """Draw a branded disclosure card and return its expandable body."""
    section = layout.box()
    expanded = bool(getattr(cfg, property_name, False))
    header = section.row(align=True)
    header.scale_y = 1.25
    header.use_property_split = False
    custom_id = theme.custom_icon(custom) if custom else 0
    if custom_id:
        header.label(text="", icon_value=custom_id)
    else:
        header.label(text="", icon=icon)
    header.prop(
        cfg,
        property_name,
        text=label,
        icon='DISCLOSURE_TRI_DOWN' if expanded else 'DISCLOSURE_TRI_RIGHT',
        emboss=False,
    )
    if state:
        marker = header.row(align=True)
        marker.alignment = 'RIGHT'
        marker.label(text="", icon=state)
    if expanded and detail:
        theme.muted(section, detail)
        section.separator(factor=0.35)
    if not expanded:
        return None
    section.separator(factor=.3)
    body = section.column()
    body.use_property_decorate = False
    return body


class _SplatGenDrawProxy:
    """Give an existing section drawer a different layout container."""

    def __init__(self, layout):
        self.layout = layout


def _sr_draw_quick_access(layout, context):
    """The essential branded masthead and utility shortcuts."""
    from . import icons, version

    theme.hero(layout, title="SplatGen", subtitle="Place cameras · Build datasets",
               version_text=f"{edition.label()}  {version.addon_stringversion}")
    utilities = layout.row(align=True)
    theme.operator(utilities, "splatray.open_preferences", text="Preferences", icon='PREFERENCES')
    theme.operator(utilities, "sceneray_splat.open_manual", text="User guide", icon='HELP')

#: Point sampling defaults: (quality, cameras used %, rays per camera, sample merging).
POINT_DEFAULTS = ('CUSTOM', 100.0, 5.0, 1.5)


def point_settings_are_default(cfg):
    return (cfg.point_quality == POINT_DEFAULTS[0]
            and abs(cfg.point_camera_usage - POINT_DEFAULTS[1]) < 1e-4
            and abs(cfg.point_sampling_density - POINT_DEFAULTS[2]) < 1e-4
            and abs(cfg.point_merging_strength - POINT_DEFAULTS[3]) < 1e-4)


class SCENERAY_SPLAT_OT_point_defaults(bpy.types.Operator):
    """Set point sampling back to the defaults: Custom, all cameras, 5 rays per camera, merging 1.5"""
    bl_idname = "sceneray_splat.point_defaults"
    bl_label = "Reset Point Sampling"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        cfg = context.scene.SCENERAY_SPLAT
        _sr_bulk["busy"] = True
        try:
            cfg.point_quality = POINT_DEFAULTS[0]
            (cfg.point_camera_usage, cfg.point_sampling_density,
             cfg.point_merging_strength) = POINT_DEFAULTS[1:]
        finally:
            _sr_bulk["busy"] = False
        _sr_mark_scene_cache_dirty(context.scene)
        _tag_redraw_sceneray_splat(context)
        return {'FINISHED'}





def _sr_preferences_area():
    """An already-open Preferences editor, if the user has one."""
    window_manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        screen = getattr(window, "screen", None)
        for area in getattr(screen, "areas", ()):
            if area.type == 'PREFERENCES':
                return window, area
    return None, None


def _sr_main_window_context():
    """A window and a real editor area to run window-opening operators from.

    ``screen.userpref_show`` polls the context it is called in, and the
    Settings button lives in a popover whose context does not pass that poll -
    which is why pressing it did nothing at all. Overriding onto an ordinary
    editor area gives the operator the context it expects wherever the button
    was actually pressed from.
    """
    window_manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        screen = getattr(window, "screen", None)
        areas = [
            area for area in getattr(screen, "areas", ())
            if area.type not in {'PREFERENCES', 'TOPBAR', 'STATUSBAR'}
        ]
        if not areas:
            continue
        area = max(areas, key=lambda item: item.width * item.height)
        region = next(
            (r for r in area.regions if r.type == 'WINDOW'),
            area.regions[-1] if len(area.regions) else None,
        )
        if region is not None:
            return window, area, region
    return None, None, None


class SCENERAY_SPLAT_OT_open_preferences(bpy.types.Operator):
    """Open SplatGen's own Add-on Preferences page"""

    bl_idname = "splatray.open_preferences"
    bl_label = "Open SplatGen Preferences"

    def _focus(self, context):
        """Point the Preferences editor at this add-on's entry."""
        preferences = getattr(context, "preferences", None)
        if preferences is not None:
            preferences.active_section = 'ADDONS'
        window_manager = getattr(context, "window_manager", None)
        if window_manager is not None:
            # Filtering by the add-on's own name rather than its module: the
            # module of an installed extension is bl_ext.<repo>.splatgen,
            # which is not what the search field matches against.
            for attribute, value in (
                ("addon_filter", 'All'),
                ("addon_search", edition.name()),
            ):
                try:
                    setattr(window_manager, attribute, value)
                except (AttributeError, TypeError):
                    pass
        try:
            bpy.ops.preferences.addon_expand(module=__package__)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def execute(self, context):
        window, area = _sr_preferences_area()
        if area is not None:
            # Already open: switching its section beats stacking a second
            # Preferences window on top of the one the user has.
            self._focus(context)
            area.tag_redraw()
            return {'FINISHED'}

        window, area, region = _sr_main_window_context()
        opened = False
        if window is not None:
            try:
                with context.temp_override(window=window, area=area,
                                           region=region,
                                           screen=window.screen):
                    bpy.ops.screen.userpref_show('INVOKE_DEFAULT')
                opened = True
            except (AttributeError, RuntimeError, TypeError):
                opened = False
        if not opened:
            try:
                bpy.ops.screen.userpref_show('INVOKE_DEFAULT')
                opened = True
            except (AttributeError, RuntimeError, TypeError) as exc:
                self.report(
                    {'ERROR'},
                    "Could not open Preferences: "
                    f"{exc}. Open Edit ▸ Preferences ▸ Add-ons and search "
                    f"for {edition.name()}.",
                )
                return {'CANCELLED'}

        self._focus(context)
        return {'FINISHED'}





class SCENERAY_SPLAT_OT_open_manual(bpy.types.Operator):
    """Open the SplatGen manual: what it does, the workflow, and every feature"""

    bl_idname = "sceneray_splat.open_manual"
    bl_label = "SplatGen Manual"
    bl_options = {'REGISTER'}

    def execute(self, context):
        manual = Path(__file__).parent / "docs" / "manual.html"
        if not manual.is_file():
            self.report({'ERROR'}, f"The manual is missing from {manual.parent}.")
            return {'CANCELLED'}
        # A local file, opened in whatever the system uses for HTML. No network
        # access, so the manual works offline and cannot go stale against the
        # installed version.
        bpy.ops.wm.path_open(filepath=str(manual))
        return {'FINISHED'}

class SCENERAY_SPLAT_PT_panel(bpy.types.Panel):
    """Shared drawing surface for SplatGen's selectable interface location."""
    bl_idname = "SCENERAY_SPLAT_PT_panel"
    bl_label = "SplatGen"
    bl_space_type = 'OUTLINER'
    bl_region_type = 'HEADER'
    bl_options = {'HIDE_HEADER'}
    bl_ui_units_x = 34

    def draw(self, context):
        from . import workspace_ui
        _sr_request_render_status_sync(context.scene.SCENERAY_SPLAT)
        workspace_ui.draw(self.layout, context)


class SPLATKIT_PT_properties(bpy.types.Panel):
    """SplatGen inside Blender's standard Scene Properties context."""

    bl_idname = "SPLATKIT_PT_properties"
    bl_label = "SplatGen"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'scene'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        preferences = _sr_addon_preferences()
        return (
            preferences is None
            or preferences.interface_location == 'PROPERTIES'
        )

    def draw(self, context):
        SCENERAY_SPLAT_PT_panel.draw(self, context)


class SPLATKIT_PT_sidebar(bpy.types.Panel):
    """Optional conventional add-on panel in the 3D View sidebar."""

    bl_idname = "SPLATKIT_PT_sidebar"
    bl_label = "SplatGen"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SplatGen"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        preferences = _sr_addon_preferences()
        return bool(
            preferences is not None
            and preferences.interface_location == 'SIDEBAR'
        )

    def draw(self, context):
        SCENERAY_SPLAT_PT_panel.draw(self, context)


class SCENERAY_SPLAT_PT_rigs(bpy.types.Panel):
    bl_idname = "SCENERAY_SPLAT_PT_rigs"
    bl_parent_id = "SCENERAY_SPLAT_PT_panel"
    bl_label = "Camera Rigs"
    bl_space_type = 'OUTLINER'
    bl_region_type = 'WINDOW'
    bl_category = "SplatGen"
    bl_order = 0

    def draw(self, context):
        from . import workspace_ui
        workspace_ui.draw_camera_methods(self.layout, context)
        workspace_ui.draw_camera_tools(self.layout, context)


class SCENERAY_SPLAT_PT_cameras(bpy.types.Panel):
    bl_idname = "SCENERAY_SPLAT_PT_cameras"
    bl_parent_id = "SCENERAY_SPLAT_PT_panel"
    bl_label = "Camera List"
    bl_space_type = 'OUTLINER'
    bl_region_type = 'WINDOW'
    bl_category = "SplatGen"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        cfg = context.scene.SCENERAY_SPLAT
        busy = cfg.is_rendering or cfg.is_generating_points
        _sr_request_render_status_sync(cfg)
        _sr_request_queue_sync(context.scene, cfg)

        _sr_apply_camera_status_colors(cfg)

        header = layout.row(align=True)
        custom_id = theme.custom_icon("stage_cameras")
        if custom_id:
            header.label(text="Camera queue", icon_value=custom_id)
        else:
            header.label(text="Camera queue", icon='OUTLINER_OB_CAMERA')
        header.prop(cfg, "camera_status_colors", text="",
                    icon='COLOR', toggle=True)

        # The list first, then what can be done to it, then how to look
        # through it. Reading top to bottom follows what the user is doing.
        if len(cfg.camera_queue):
            prefs = _sr_addon_preferences()
            rows = getattr(prefs, "camera_list_rows", 8) if prefs else 8
            rows = min(rows, max(2, len(cfg.camera_queue)))
            layout.template_list("SCENERAY_SPLAT_UL_camera_queue", "", cfg,
                                 "camera_queue", cfg, "active_camera_index",
                                 rows=rows, maxrows=rows)

            valid_cameras = [
                item for item in cfg.camera_queue
                if item.camera is not None and item.camera.type == 'CAMERA'
            ]
            rendered_count = sum(
                1 for item in valid_cameras if item.render_state == 'RENDERED'
            )
            pending_count = len(valid_cameras) - rendered_count
            counts = layout.box()
            theme.stat(counts, (
                ("Rendered", f"{rendered_count:,} / {len(valid_cameras):,}"),
                ("Pending", f"{pending_count:,}"),
            ))

        _sr_draw_queue_actions(layout, cfg, busy)
        _sr_draw_camera_navigation(layout, cfg, busy)

        # The settings shared by every camera live with the list they apply
        # to, so both panels that show the list show them identically.
        settings = layout.box()
        settings.prop(
            cfg,
            "show_global_camera_settings",
            text="Global Camera Settings",
            icon='TRIA_DOWN' if cfg.show_global_camera_settings else 'TRIA_RIGHT',
            emboss=False,
        )
        if cfg.show_global_camera_settings:
            SCENERAY_SPLAT_PT_global_camera_settings.draw(
                _SplatGenDrawProxy(settings), context)


def _sr_draw_queue_actions(layout, cfg, busy):
    """Add and remove queue entries: the four everyday list actions."""
    actions = layout.column(align=True)
    actions.enabled = not busy

    add = actions.row(align=True)
    theme.operator(add, "sceneray_splat.queue_add_selected",
                 text="Add selected", icon='ADD')
    theme.operator(add, "sceneray_splat.queue_add_all", text="Add all",
                 icon='OUTLINER_OB_CAMERA')

    remove = actions.row(align=True)
    remove.enabled = not busy and bool(len(cfg.camera_queue))
    clear_selected = theme.operator(remove, 
        "sceneray_splat.queue_remove",
        text="Remove selected",
        icon='REMOVE',
    )
    clear_selected.index = max(0, cfg.active_camera_index)
    theme.operator(remove, "sceneray_splat.queue_clear", text="Remove all",
                    icon='TRASH')
    # Maintenance rather than an everyday action, so it keeps its icon-only
    # button at the end of the row instead of a label of its own.
    theme.operator(remove, "sceneray_splat.queue_clean", text="", custom='action_delete', icon='BRUSH_DATA')


def _sr_draw_camera_navigation(layout, cfg, busy):
    """Stepping through the queued cameras, one at a time.

    This used to be a separate Camera View section. It lives with the list it
    navigates so reviewing a camera never means switching panels.
    """
    box = layout.box()
    theme.heading(
        box,
        "CAMERA PREVIEW",
        custom="stage_viewer",
        fallback='CAMERA_DATA',
    )
    if not len(cfg.camera_queue):
        return

    row = box.row(align=True)
    row.enabled = not busy
    step_back = row.row(align=True)
    step_back.enabled = cfg.review_active
    theme.operator(step_back, "sceneray_splat.previous_camera", text="Previous",
                       icon='TRIA_LEFT')
    theme.operator(row, 
        "sceneray_splat.review_cameras",
        text="Exit" if cfg.review_active else "Preview",
        icon='CAMERA_DATA',
    )
    step_forward = row.row(align=True)
    step_forward.enabled = cfg.review_active
    theme.operator(step_forward, "sceneray_splat.next_camera", text="Next",
                          icon='TRIA_RIGHT')

    delete = box.row()
    delete.enabled = not busy and cfg.review_active
    theme.operator(delete, "sceneray_splat.delete_active_camera",
                    text="Delete Current Camera", icon='TRASH')


class SCENERAY_SPLAT_PT_global_camera_settings(bpy.types.Panel):
    bl_idname = "SCENERAY_SPLAT_PT_global_camera_settings"
    bl_parent_id = "SCENERAY_SPLAT_PT_panel"
    bl_label = "Global Camera Settings"
    bl_space_type = 'OUTLINER'
    bl_region_type = 'WINDOW'
    bl_category = "SplatGen"
    bl_order = 2

    def draw(self, context):
        layout, cfg = self.layout, context.scene.SCENERAY_SPLAT
        layout.enabled = not (cfg.is_rendering or cfg.is_generating_points)

        # Three collapsed groups rather than one long column. These values are
        # set once for a project and rarely touched again, so none of them
        # deserves permanent vertical space above the controls in daily use.
        clipping = _sr_settings_group(
            layout, cfg, "show_clipping_settings", "Clipping Settings",
            'CON_CAMERASOLVER')
        if clipping is not None:
            _sr_draw_clipping_settings(clipping, context, cfg)

        lens = _sr_settings_group(
            layout, cfg, "show_lens_settings", "Lens & Sensor", 'CAMERA_DATA')
        if lens is not None:
            lens.prop(cfg, "global_focal_length")
            lens.prop(cfg, "global_sensor_width")
            lens.prop(cfg, "global_sensor_height")
            lens.prop(cfg, "global_sensor_fit")
            lens.prop(cfg, "global_shift_x")
            lens.prop(cfg, "global_shift_y")

        display = _sr_settings_group(
            layout, cfg, "show_display_settings", "Viewport Display",
            'RESTRICT_VIEW_OFF')
        if display is not None:
            display.prop(cfg, "global_display_size")
            display.prop(cfg, "global_use_passepartout")
            row = display.row()
            row.enabled = cfg.global_use_passepartout
            row.prop(cfg, "global_passepartout_opacity")


def _sr_settings_group(layout, cfg, property_name, label, icon):
    """A collapsed disclosure row; returns its body only when it is open."""
    column = layout.column(align=True)
    expanded = bool(getattr(cfg, property_name, False))
    header = column.row(align=True)
    header.use_property_split = False
    header.alignment = 'LEFT'
    header.label(text="", icon=icon)
    header.prop(
        cfg,
        property_name,
        text=label,
        icon='DISCLOSURE_TRI_DOWN' if expanded else 'DISCLOSURE_TRI_RIGHT',
        emboss=False,
    )
    if not expanded:
        return None
    column.separator(factor=.25)
    return theme.form(column)


def _sr_draw_clipping_settings(layout, context, cfg):
    """Live clipping for the whole rig, or the selected individual camera."""
    layout.prop(cfg, "use_global_clipping")
    if cfg.use_global_clipping:
        # Editing either value writes it onto every actual camera datablock,
        # so Camera Preview shows the same clipping before any build starts.
        layout.prop(cfg, "global_clip_start")
        layout.prop(cfg, "global_clip_end")
        return

    active = None
    if 0 <= cfg.active_camera_index < len(cfg.camera_queue):
        active = cfg.camera_queue[cfg.active_camera_index].camera
    if active is None or active.type != 'CAMERA':
        layout.label(text="Select a camera to set its clipping.", icon='INFO')
        return
    layout.label(text=f"Clipping — {active.name}", icon='CON_CAMERASOLVER')
    layout.prop(active.data, "clip_start", text="Clip Start")
    layout.prop(active.data, "clip_end", text="Clip End")


class SCENERAY_SPLAT_PT_camera_placement(bpy.types.Panel):
    """All camera creation, rig, queue, review, and calibration controls."""

    bl_idname = "SCENERAY_SPLAT_PT_camera_placement"
    bl_parent_id = "SCENERAY_SPLAT_PT_panel"
    bl_label = "1. Camera Placement"
    bl_space_type = 'OUTLINER'
    bl_region_type = 'WINDOW'
    bl_category = "SplatGen"
    bl_order = 0

    def draw(self, context):
        # Drawn flat. Camera Placement is already a collapsible section, so
        # wrapping its contents in a second one only nested a box inside a box
        # and pushed the rig list a level deeper for nothing.
        SCENERAY_SPLAT_PT_rigs.draw(
            _SplatGenDrawProxy(self.layout),
            context,
        )

        # The camera list is not repeated here. There is one, permanently
        # visible at the top of Building Dataset, and a second copy was only
        # ever another place for the same list to be scrolled past.
        #
        # Global Camera Settings moved to Building Data: they describe the
        # whole project rather than where any one camera goes.
        #
        # Coverage validation is not drawn here either. It has one entry
        # point, directly above Build Dataset, so the workflow reads as
        # validate then build with no second place to start it from.




@persistent
def _sr_block_manual_render(scene, _depsgraph=None):
    """Refuse a manual render while the add-on is already working.

    Validation, dataset generation, the point cloud and training all saturate
    the same CPU and GPU. Starting an interactive render on top of them fights
    for those resources and, during a dataset build, competes for the very
    render pipeline the build is driving.

    Raising from ``render_init`` is how an add-on cancels a render before it
    starts; Blender shows the message and nothing is rendered.
    """
    from .building_data import stages

    # A render the add-on started itself must obviously be allowed through:
    # the dataset build drives Blender's renderer to make its images.
    cfg = getattr(scene, "SCENERAY_SPLAT", None)
    if _render_job_state.get("owner") is not None:
        return
    if raw_stage.is_running():
        return
    if cfg is not None and cfg.is_rendering:
        return
    if not stages.anything_running():
        return
    raise RuntimeError(
        "SplatGen is busy - wait for the current operation to finish, or "
        "stop it, before rendering."
    )


def _sr_enforce_panel_order():
    """Restore the add-on's section order after workspace panel dragging."""
    if SCENERAY_SPLAT_PT_camera_placement.bl_order != 0:
        SCENERAY_SPLAT_PT_camera_placement.bl_order = 0


# 19. PREFERENCES + REGISTRATION

class SCENERAY_SPLAT_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or __name__

    interface_location: EnumProperty(
        name="Interface Location",
        description="Choose the one place where the SplatGen workflow appears",
        items=(
            (
                'PROPERTIES',
                "Scene Properties",
                "Show SplatGen as a normal panel in Scene Properties",
            ),
            (
                'SIDEBAR',
                "3D View Sidebar",
                "Show SplatGen in the conventional 3D View N-panel",
            ),
        ),
        default='SIDEBAR',
        update=lambda self, context: _tag_redraw_sceneray_splat(context),
    )
    rig_library_dir: StringProperty(
        name="Camera Rig Library",
        description="Custom rigs shared by all projects. Empty uses the automatic SplatGen user folder. Bundled rigs are always available; changing this path does not move existing files",
        default=str(camera_rigs.default_user_dir()),
        subtype='DIR_PATH')
    camera_list_rows: IntProperty(
        name="Maximum Camera Rows", default=5, min=4, max=24,
        description="Maximum visible rows; shorter lists shrink to fit")
    def draw(self, context):
        _sr_draw_preferences(self.layout, context, self)


def _sr_draw_preferences(layout, context, preferences):
    from . import version

    theme.hero(
        layout,
        title="SplatGen preferences",
        subtitle="Workspace & shortcuts",
        version_text=f"{edition.label()} · v{version.addon_stringversion}",
    )
    layout.use_property_split = True
    layout.use_property_decorate = False
    support = layout.row(align=True)
    theme.operator(support, "splatgen.copy_diagnostics",
                   text="Copy bug report & save logs", custom='ui_bug', icon='INFO')
    interface = layout.box()
    interface.use_property_split = True
    interface.use_property_decorate = False
    theme.heading(
        interface,
        "INTERFACE",
        custom="stage_project",
        fallback='PREFERENCES',
    )
    interface.prop(preferences, "interface_location")
    interface.prop(preferences, "rig_library_dir")
    interface.prop(preferences, "camera_list_rows")

    layout.separator()
    shortcuts = theme.card(layout, "Keyboard shortcuts", 'EVENT_A',
        "Left / Right: review · Backspace: delete")
    layout = shortcuts
    wm = context.window_manager
    keyconfig = wm.keyconfigs.user
    keymap = keyconfig.keymaps.get("3D View") if keyconfig else None
    if keymap is None:
        layout.label(text="Edit it in Preferences ▸ Keymap ▸ 3D View.",
                     icon='INFO')
        return
    try:
        import rna_keymap_ui
    except ImportError:
        layout.label(text="Edit it in Preferences ▸ Keymap ▸ 3D View.",
                     icon='INFO')
        return
    found = False
    for kmi in keymap.keymap_items:
        if kmi.idname in {"sceneray_splat.add_camera_from_view", "sceneray_splat.previous_camera",
                          "sceneray_splat.next_camera", "sceneray_splat.delete_active_camera"}:
            rna_keymap_ui.draw_kmi([], keyconfig, keymap, kmi, layout, 0)
            found = True
    if not found:
        layout.label(text="No shortcut assigned. Add one under "
                          "Preferences ▸ Keymap ▸ 3D View.", icon='INFO')


class SPLATGEN_OT_workflow_step(bpy.types.Operator):
    bl_idname = 'sceneray_splat.workflow_step'
    bl_label = 'Open workflow step'
    bl_description = 'Switch workspace; the running job and its controls remain visible'
    step: EnumProperty(items=[('PREPARE','Prepare','')])
    def execute(self, context):
        context.scene.SCENERAY_SPLAT.workspace_step = self.step
        return {'FINISHED'}


class SPLATGEN_OT_camera_method(bpy.types.Operator):
    bl_idname = 'sceneray_splat.camera_method'
    bl_label = 'Choose camera method'
    method: EnumProperty(items=[('SMART', 'Smart rig', ''), ('MANUAL', 'Manual', ''), ('SAVED', 'Saved', '')])

    @classmethod
    def description(cls, context, properties):
        return {'SMART': 'Smart camera rig: calculate cameras for rooms, buildings and objects',
                'MANUAL': 'Manual cameras: use the current view and its shortcut, or mesh faces',
                'SAVED': 'Saved camera rigs: load a bundled rig or reuse your own arrangement'}.get(properties.method, 'Choose how to add cameras')

    def execute(self, context):
        context.scene.splatgen_auto_rig.camera_method = self.method
        return {'FINISHED'}


class SPLATGEN_OT_dismiss_task(bpy.types.Operator):
    bl_idname = 'sceneray_splat.dismiss_task'
    bl_label = 'Dismiss finished task'
    token: StringProperty()
    def execute(self, context):
        if self.token == 'PROGRESS':
            if not progress.is_active(): progress.reset()
        else:
            context.scene.SCENERAY_SPLAT.dismissed_job = self.token
            if not progress.is_active():
                progress.reset()
        return {'FINISHED'}


class SPLATGEN_PT_camera_tools(bpy.types.Panel):
    bl_idname = 'SPLATGEN_PT_camera_tools'
    bl_label = 'Saved camera rigs'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_ui_units_x = 18
    def draw(self, context):
        from . import workspace_ui
        workspace_ui.draw_camera_tools(self.layout, context)


class SPLATGEN_PT_camera_settings(bpy.types.Panel):
    bl_idname = 'SPLATGEN_PT_camera_settings'
    bl_label = 'Camera settings'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_ui_units_x = 20
    def draw(self, context):
        SCENERAY_SPLAT_PT_global_camera_settings.draw(self, context)


class SPLATGEN_MT_queue_actions(bpy.types.Menu):
    bl_idname = 'SPLATGEN_MT_queue_actions'
    bl_label = 'Camera queue actions'

    def draw(self, context):
        from .building_data import stages
        cfg = context.scene.SCENERAY_SPLAT
        layout = self.layout
        layout.enabled = not stages.anything_running(context)
        theme.operator(layout, 'sceneray_splat.queue_add_selected', text='Add selected cameras', icon='ADD')
        theme.operator(layout, 'sceneray_splat.queue_add_all', text='Add all scene cameras', icon='CAMERA_DATA')
        layout.separator()
        layout.prop(cfg, 'camera_status_colors')
        edit = layout.column()
        edit.enabled = bool(cfg.camera_queue)
        theme.operator(edit, 'sceneray_splat.preset_save', text='Save queue as rig', icon='FILE_TICK')
        theme.operator(edit, 'sceneray_splat.queue_clean', text='Clean missing / duplicates', custom='action_delete', icon='FILE_REFRESH')
        theme.operator(edit, 'sceneray_splat.queue_clear', text='Clear queue', icon='TRASH')
        if cfg.review_active:
            layout.separator()
            theme.operator(layout, 'sceneray_splat.delete_active_camera', text='Delete preview camera', icon='TRASH')


class SPLATGEN_PT_dataset_settings(bpy.types.Panel):
    bl_idname = 'SPLATGEN_PT_dataset_settings'
    bl_label = 'Dataset settings'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_ui_units_x = 20

    def draw(self, context):
        from . import workspace_ui
        workspace_ui.draw_dataset_settings(self.layout, context)





_classes = (
    SPLATGEN_OT_workflow_step, SPLATGEN_OT_camera_method, SPLATGEN_OT_dismiss_task, SPLATGEN_PT_camera_tools,
    SPLATGEN_PT_camera_settings,
    SPLATGEN_MT_queue_actions, SPLATGEN_PT_dataset_settings,
    SceneRaySplatQueueItem, SceneRaySplatRigCamera, SceneRaySplatRigPreset,
    SceneRaySplatProperties,
    SCENERAY_SPLAT_OT_add_camera_from_view,
    SCENERAY_SPLAT_OT_create_cameras_from_faces,
    SCENERAY_SPLAT_OT_queue_add_selected,
    SCENERAY_SPLAT_OT_queue_add_all,
    SCENERAY_SPLAT_OT_review_cameras, SCENERAY_SPLAT_OT_previous_camera, SCENERAY_SPLAT_OT_next_camera,
    SCENERAY_SPLAT_OT_delete_active_camera, SCENERAY_SPLAT_OT_mark_pending,
    SCENERAY_SPLAT_OT_queue_remove, SCENERAY_SPLAT_OT_queue_clean, SCENERAY_SPLAT_OT_queue_clear,
    SCENERAY_SPLAT_OT_preset_save, SCENERAY_SPLAT_OT_preset_rename,
    SCENERAY_SPLAT_OT_preset_delete, SCENERAY_SPLAT_OT_preset_load,
    SCENERAY_SPLAT_OT_render_images,
    SCENERAY_SPLAT_OT_calculate_cameras, SCENERAY_SPLAT_OT_generate_points3d,
    SCENERAY_SPLAT_OT_stop_render,
    SCENERAY_SPLAT_OT_open_output,
    SCENERAY_SPLAT_OT_open_manual,
    SCENERAY_SPLAT_OT_open_preferences,
    SCENERAY_SPLAT_OT_point_defaults,
    SCENERAY_SPLAT_OT_clean_temp,
    SCENERAY_SPLAT_UL_camera_queue, SCENERAY_SPLAT_UL_rig_presets,
    SCENERAY_SPLAT_PT_blender_render_progress,
    SPLATKIT_PT_properties,
    SPLATKIT_PT_sidebar,
)


_addon_keymaps = []


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    _sr_enforce_panel_order()
    try:
        bpy.utils.register_class(SCENERAY_SPLAT_AddonPreferences)
    except Exception as exc:
        print(f"[sceneray_splat] add-on preferences unavailable: {exc}")

    bpy.types.Scene.SCENERAY_SPLAT = PointerProperty(type=SceneRaySplatProperties)
    # Blender replaces bpy.data with _RestrictData while an add-on is being
    # installed and enabled. Do not enumerate scenes here: newly registered
    # process-only properties already have clean defaults, and the load_post
    # handler below resets saved state after every File > New/Open operation.
    if _sr_reset_build_state_after_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_sr_reset_build_state_after_load)
    if _sr_depsgraph_cache_invalidate not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(
            _sr_depsgraph_cache_invalidate)
    # The guard goes on first, so it can refuse a render before any of the
    # add-on's own bookkeeping runs for it.
    if _sr_block_manual_render not in bpy.app.handlers.render_init:
        bpy.app.handlers.render_init.insert(0, _sr_block_manual_render)
    for handler, collection in (
            (_sr_render_job_init, bpy.app.handlers.render_init),
            (_sr_render_job_complete, bpy.app.handlers.render_complete),
            (_sr_render_job_cancel, bpy.app.handlers.render_cancel)):
        if handler not in collection:
            collection.append(handler)

    # A keymap entry — not a hard-coded hotkey. It shows up under
    # Preferences ▸ Keymap ▸ 3D View, where any key, mouse button or
    # modifier combination can be assigned instead.
    try:
        keyconfig = bpy.context.window_manager.keyconfigs.addon
        if keyconfig is not None:
            keymap = keyconfig.keymaps.new(name="3D View",
                                           space_type='VIEW_3D')
            for idname, key, kwargs in (
                ("sceneray_splat.add_camera_from_view", 'F', {"shift": True}),
                ("sceneray_splat.previous_camera", 'LEFT_ARROW', {}),
                ("sceneray_splat.next_camera", 'RIGHT_ARROW', {}),
                ("sceneray_splat.delete_active_camera", 'BACK_SPACE', {}),
            ):
                item = keymap.keymap_items.new(idname, key, 'PRESS', **kwargs)
                _addon_keymaps.append((keymap, item))
    except Exception as exc:
        print(f"[sceneray_splat] could not register the Add Camera from View "
              f"shortcut: {exc}")


def unregister():
    _sr_abort_render_batch("SplatGen add-on disabled.")
    _sr_clear_render_job()
    # Never leave a preference we borrowed in the borrowed state.
    restore_render_display()
    # A draw handler left behind survives the add-on and crashes on the next
    # redraw, so it comes off first.
    from . import camera_overlay

    camera_overlay.disable()
    for handler, collection in (
            (_sr_block_manual_render, bpy.app.handlers.render_init),
            (_sr_render_job_init, bpy.app.handlers.render_init),
            (_sr_render_job_complete, bpy.app.handlers.render_complete),
            (_sr_render_job_cancel, bpy.app.handlers.render_cancel),
            (_sr_reset_build_state_after_load, bpy.app.handlers.load_post),
            (_sr_depsgraph_cache_invalidate,
             bpy.app.handlers.depsgraph_update_post)):
        try:
            collection.remove(handler)
        except ValueError:
            pass
    for timer_fn in (_sr_run_pending_render_status_sync,
                     _sr_run_pending_queue_sync,
                     _sr_run_pending_rig_library_refresh,
                     _sr_render_batch_tick,
                     _sr_run_dataset_continuation,
                     _sr_dispatch_render_modal_wakeup):
        try:
            if bpy.app.timers.is_registered(timer_fn):
                bpy.app.timers.unregister(timer_fn)
        except (ReferenceError, RuntimeError, ValueError):
            pass
    _ui_scene_cache["revisions"].clear()
    _render_status_sync["pending"].clear()
    _render_status_sync["signatures"].clear()
    _render_status_sync["last_checks"].clear()
    _render_status_sync["timer"] = False
    _render_modal_wakeup["timer_registered"] = False
    _render_batch_runtime["batch"] = None
    _render_batch_runtime["timer_registered"] = False
    _render_dataset_continuation["scene_pointer"] = None
    _render_dataset_continuation["window_pointer"] = None
    _render_dataset_continuation["timer_registered"] = False
    _queue_sync["pending"].clear()
    _queue_sync["checks"].clear()
    _queue_sync["timer_registered"] = False
    _rig_library_cache["pending"].clear()
    _rig_library_cache["last_checks"].clear()
    _rig_library_cache["timer_registered"] = False
    _rig_library_cache["key"] = None

    for keymap, item in _addon_keymaps:
        try:
            keymap.keymap_items.remove(item)
        except Exception:
            from . import diagnostics as _diag
            _diag.swallowed("sceneray_splat.py")
    _addon_keymaps.clear()

    if hasattr(bpy.types.Scene, "SCENERAY_SPLAT"):
        del bpy.types.Scene.SCENERAY_SPLAT

    try:
        bpy.utils.unregister_class(SCENERAY_SPLAT_AddonPreferences)
    except Exception:
        from . import diagnostics as _diag
        _diag.swallowed("sceneray_splat.py")
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
