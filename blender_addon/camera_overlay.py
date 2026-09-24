# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Draw each queued camera in its status colour, in the 3D View.

``Object.color`` is the only per-object colour Blender exposes, and it reaches
a mesh's wireframe only when the viewport's Wire Color is set to Object. Camera
objects are drawn as *extras*, not as object geometry, so that setting does not
reliably recolour them - which is why setting ``Object.color`` alone left the
viewport looking unchanged.

So the status colour is drawn here instead: a frustum outline over each queued
camera, in the same orange and blue the Camera List uses. It is an overlay, so
it never touches the camera objects, never appears in a render, and disappears
completely when the feature is switched off.
"""

import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

import bpy

from . import theme

#: The handle returned by ``draw_handler_add``, or None while switched off.
_handle = None


def _pose(camera_object):
    """Camera-to-world with the parent's scale taken out.

    A camera parented to a mesh inherits that mesh's scale, so drawing from
    ``matrix_world`` directly would stretch the status box exactly as it
    stretches the camera. Blender's own camera gizmo ignores scale; this
    matches it.
    """
    location, rotation, _scale = camera_object.matrix_world.decompose()
    matrix = rotation.to_matrix().to_4x4()
    matrix.translation = location
    return matrix


def _frustum_lines(camera_object):
    """Line segments tracing this camera's view frustum, in world space.

    Sized from the camera's own Viewport Display size, so the status box is
    always the same size as the camera gizmo it wraps - change one in the
    camera settings and the other follows.
    """
    data = camera_object.data
    matrix = _pose(camera_object)
    # Blender draws its camera cone this far along -Z; matching it is what
    # keeps the two the same size at any display setting.
    depth = max(float(getattr(data, "display_size", 1.0)), 1e-4)

    scene = bpy.context.scene
    render = scene.render if scene is not None else None
    width = float(getattr(render, "resolution_x", 1920) or 1920)
    height = float(getattr(render, "resolution_y", 1080) or 1080)
    aspect = width / height if height else 1.0

    if data.type == 'ORTHO':
        half_x = data.ortho_scale * 0.5
        half_y = half_x / aspect if aspect else half_x
    else:
        # Blender fits the sensor to the larger image dimension.
        sensor = data.sensor_width if aspect >= 1.0 else data.sensor_height
        half = (sensor * 0.5) / max(data.lens, 1e-6) * depth
        half_x = half if aspect >= 1.0 else half * aspect
        half_y = half / aspect if aspect >= 1.0 else half

    corners = [
        matrix @ Vector((-half_x, -half_y, -depth)),
        matrix @ Vector((half_x, -half_y, -depth)),
        matrix @ Vector((half_x, half_y, -depth)),
        matrix @ Vector((-half_x, half_y, -depth)),
    ]
    apex = matrix @ Vector((0.0, 0.0, 0.0))

    lines = []
    for index in range(4):
        lines.extend((corners[index], corners[(index + 1) % 4]))
        lines.extend((apex, corners[index]))
    return lines


def _draw():
    """Called by Blender for every 3D View redraw while the overlay is on."""
    context = bpy.context
    cfg = getattr(context.scene, "SCENERAY_SPLAT", None)
    if cfg is None or not getattr(cfg, "camera_status_colors", False):
        return

    pending, rendered = [], []
    for item in cfg.camera_queue:
        camera = item.camera
        if camera is None or camera.type != 'CAMERA':
            continue
        try:
            if camera.hide_get() or not camera.visible_get():
                continue
        except (RuntimeError, ReferenceError):
            continue
        target = (rendered if item.render_state == 'RENDERED' else pending)
        try:
            target.extend(_frustum_lines(camera))
        except (AttributeError, ReferenceError, ValueError, ZeroDivisionError):
            continue

    if not (pending or rendered):
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    state = gpu.state
    state.blend_set('ALPHA')
    state.line_width_set(2.0)
    # Depth test on, so a camera behind geometry reads as behind it rather
    # than floating in front of the scene.
    state.depth_test_set('LESS_EQUAL')
    try:
        for points, colour in (
            (pending, theme.CAMERA_PENDING_RGBA),
            (rendered, theme.CAMERA_RENDERED_RGBA),
        ):
            if not points:
                continue
            batch = batch_for_shader(shader, 'LINES', {"pos": points})
            shader.uniform_float("color", colour)
            batch.draw(shader)
    finally:
        state.depth_test_set('NONE')
        state.line_width_set(1.0)
        state.blend_set('NONE')


def tag_redraw():
    window_manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def is_active():
    return _handle is not None


def enable():
    """Start drawing. Safe to call when already drawing."""
    global _handle
    if _handle is not None:
        return
    _handle = bpy.types.SpaceView3D.draw_handler_add(
        _draw, (), 'WINDOW', 'POST_VIEW')
    tag_redraw()


def disable():
    """Stop drawing and remove every trace of the overlay."""
    global _handle
    if _handle is None:
        return
    try:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
    except (ValueError, RuntimeError):
        pass
    _handle = None
    tag_redraw()


def sync(cfg):
    """Keep the camera-state layer alive whatever the colour switch does."""
    # _draw() already checks camera_status_colors. Keeping this registered as
    # a no-op while disabled means the handler is added exactly once per
    # session, so its draw order relative to the Splat layer never shifts.
    enable()


def register():
    pass


def unregister():
    disable()
