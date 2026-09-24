# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Live viewport preview of the Smart camera rig while it calculates.

What the planner actually does, drawn as it happens:

    surfaces     faint points on every surface it read
    free space   each system's space flooding out from its blob - orange for
                 an interior, blue for an exterior, yellow around objects
    candidates   white dots at every viewpoint being tested
    cameras      each chosen camera pops in as a frustum, newest highlighted
    coverage     at the end, every surface from red (unseen) to green
                 (fully covered), then the preview fades away

The planner thread only drops small, subsampled snapshots into a locked
buffer. Batches are built on the main thread when something changed and
cached between redraws, so watching does not slow the calculation down.
Overlays only: nothing is added to the scene and nothing can render.
"""

import threading
import time

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from .. import palette

COLOURS = {
    'INTERIOR': palette.rgb(palette.ORANGE),
    'EXTERIOR': palette.rgb(palette.BLUE),
    'OBJECT': palette.rgb(palette.YELLOW),
}


def _linear(rgb):
    """Display colour to the linear colour the viewport overlay expects."""
    return tuple(float(c) ** 2.2 for c in rgb)
REVEAL_SECONDS = 1.1     # how long a flood takes to spread on screen
HOLD_SECONDS = 3.5       # how long the finished result stays
FADE_SECONDS = 1.2

_lock = threading.Lock()
_handles = {"view": None, "pixel": None}
_cache = {}


def _fresh():
    return {
        "active": False, "version": 0, "started": 0.0, "finished": 0.0,
        "surfaces": [], "regions": [], "candidates": None, "cameras": [],
        "coverage": None, "scale": 1.0, "label": "Reading the scene",
    }


_state = _fresh()


# ---------------------------------------------------------------------------
# Feeding (any thread)
# ---------------------------------------------------------------------------

def feed(kind, **data):
    """Receive one snapshot from the planner. Cheap: no drawing here."""
    with _lock:
        state = _state
        if not state["active"]:
            return
        if kind == 'surface':
            points = data["points"]
            state["surfaces"].append(points)
            if len(points):
                extent = float(np.linalg.norm(np.ptp(points, axis=0)))
                state["scale"] = max(state["scale"], extent)
            state["label"] = "Reading surfaces"
        elif kind == 'region':
            points = data["points"]
            seed = data["seed"]
            # Nearest first, so revealing a growing prefix looks like the
            # flood spreading out from the blob.
            order = np.argsort(np.sum((points - seed) ** 2, axis=1))
            state["regions"].append({
                "points": np.ascontiguousarray(points[order]),
                "colour": COLOURS.get(data["space"], (1.0, 1.0, 1.0)),
                "arrived": time.time(), "name": data["name"],
            })
            state["label"] = f"Filling the space of {data['name']}"
        elif kind == 'candidates':
            state["candidates"] = data["points"]
            state["label"] = f"Testing {len(data['points']):,} viewpoints"
        elif kind == 'camera':
            state["cameras"].append((data["position"], data["forward"], data["up"]))
            state["label"] = f"Placing cameras · {len(state['cameras'])}"
        elif kind == 'coverage':
            state["coverage"] = (data["points"], data["level"])
            state["label"] = f"Done · {len(state['cameras'])} cameras"
        state["version"] += 1


def camera_count():
    with _lock:
        return len(_state["cameras"])


# ---------------------------------------------------------------------------
# Lifecycle (main thread)
# ---------------------------------------------------------------------------

def begin():
    clear()
    with _lock:
        _state.update(_fresh())
        _state["active"] = True
        _state["started"] = time.time()
    if _handles["view"] is None:
        _handles["view"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_view, (), 'WINDOW', 'POST_VIEW')
    if _handles["pixel"] is None:
        _handles["pixel"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_label, (), 'WINDOW', 'POST_PIXEL')
    redraw()


def finish():
    """Keep the finished result on screen briefly, then fade it out."""
    with _lock:
        if not _state["active"]:
            return
        _state["finished"] = time.time()
    if not bpy.app.timers.is_registered(_fade_tick):
        bpy.app.timers.register(_fade_tick, first_interval=0.05)


def _fade_tick():
    with _lock:
        finished = _state["finished"]
    if not finished:
        return None
    if time.time() - finished > HOLD_SECONDS + FADE_SECONDS:
        clear()
        return None
    redraw()
    return 0.05


def clear():
    for key, space in (("view", 'WINDOW'), ("pixel", 'WINDOW')):
        handle = _handles[key]
        if handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handle, space)
            except (ValueError, RuntimeError):
                pass
            _handles[key] = None
    with _lock:
        _state.update(_fresh())
    _cache.clear()
    try:
        if bpy.app.timers.is_registered(_fade_tick):
            bpy.app.timers.unregister(_fade_tick)
    except (ValueError, RuntimeError):
        pass
    redraw()


def redraw():
    window_manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# ---------------------------------------------------------------------------
# Drawing (main thread)
# ---------------------------------------------------------------------------

def _shader(name, fallback):
    try:
        return gpu.shader.from_builtin(name)
    except (ValueError, SystemError):
        return gpu.shader.from_builtin(fallback)


def _points_batch(key, points, rgba):
    """Cached point batch; ``key`` changes whenever its content should."""
    cached = _cache.get(key)
    if cached is not None:
        return cached
    # Drop the superseded batch of the same layer (an earlier reveal step).
    for old in [k for k in _cache if k[:2] == key[:2] and k != key]:
        del _cache[old]
    shader = _shader('POINT_FLAT_COLOR', 'FLAT_COLOR')
    colours = np.empty((len(points), 4), dtype=np.float32)
    colours[:] = (*_linear(rgba[:3]), rgba[3])
    batch = batch_for_shader(shader, 'POINTS', {"pos": points, "color": colours})
    _cache[key] = (shader, batch)
    return shader, batch


def _draw_points(key, points, rgba, size):
    if points is None or not len(points):
        return
    shader, batch = _points_batch(key, points, rgba)
    gpu.state.point_size_set(size)
    batch.draw(shader)


def _frustum_segments(cameras, scale):
    """Line segments of small frustums; the newest camera stands out."""
    depth = float(np.clip(scale * 0.022, 0.08, 3.0))
    half_w, half_h = 0.55 * depth, 0.37 * depth
    coords, colours = [], []
    last = len(cameras) - 1
    for index, (position, forward, up) in enumerate(cameras):
        right = np.cross(forward, up)
        centre = position + forward * depth
        corners = [centre + right * sx * half_w + up * sy * half_h
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        colour = ((*_linear(palette.rgb(palette.YELLOW)), 1.0) if index == last
                  else (*_linear(palette.rgb(palette.ORANGE)), 0.95))
        for k in range(4):
            coords += [position, corners[k], corners[k], corners[(k + 1) % 4]]
            colours += [colour] * 4
        # A short "up" tick, like Blender's camera triangle.
        tip = centre + up * half_h * 1.6
        coords += [corners[2], tip, tip, corners[3]]
        colours += [colour] * 4
    return np.asarray(coords, np.float32), np.asarray(colours, np.float32)


def _draw_view():
    with _lock:
        state = dict(_state)
        surfaces = list(_state["surfaces"])
        regions = list(_state["regions"])
        cameras = list(_state["cameras"])
    if not state["active"]:
        return
    now = time.time()
    fade = 1.0
    if state["finished"]:
        fade = float(np.clip(1.0 - (now - state["finished"] - HOLD_SECONDS) / FADE_SECONDS,
                             0.0, 1.0))
        if fade <= 0.0:
            return
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    have_candidates = state["candidates"] is not None
    have_coverage = state["coverage"] is not None

    # Surfaces: a faint ghost of what was read.
    ghost = 0.10 if regions else 0.22
    for index, points in enumerate(surfaces):
        _draw_points(("surface", index), points, (0.78, 0.84, 0.95, ghost * fade), 2.0)

    # Free space: each flood spreads from its blob, then settles back.
    for index, region in enumerate(regions):
        points = region["points"]
        grow = float(np.clip((now - region["arrived"]) / REVEAL_SECONDS, 0.0, 1.0))
        grow = 1.0 - (1.0 - grow) ** 3
        steps = 40
        shown = int(len(points) * round(grow * steps) / steps)
        if shown <= 0:
            continue
        alpha = 0.08 if have_coverage else (0.14 if have_candidates else 0.32)
        r, g, b = region["colour"]
        _draw_points(("region", index, shown), points[:shown], (r, g, b, alpha * fade), 2.5)

    if have_candidates and not have_coverage:
        alpha = 0.30 if cameras else 0.85
        _draw_points(("candidates", len(state["candidates"]), bool(cameras)),
                     state["candidates"], (1.0, 1.0, 1.0, alpha * fade), 5.0)

    if have_coverage:
        points, level = state["coverage"]
        key = ("coverage", len(points), round(fade, 2))
        if key not in _cache:
            _cache.pop(next((k for k in _cache if k[0] == "coverage"), None), None)
            shader = _shader('POINT_FLAT_COLOR', 'FLAT_COLOR')
            colours = np.empty((len(points), 4), np.float32)
            # Discrete exact brand colors instead of a blended pastel ramp.
            table = np.array([palette.rgb(c) for c in (palette.RED, palette.ORANGE,
                                                       palette.YELLOW, palette.GREEN)], dtype=np.float32)
            band = np.minimum((np.clip(level, 0.0, 1.0) * 4).astype(np.int32), 3)
            colours[:, :3] = table[band]
            colours[:, :3] = np.power(colours[:, :3], 2.2)
            colours[:, 3] = 0.9 * fade
            _cache[key] = (shader, batch_for_shader(shader, 'POINTS',
                                                    {"pos": points, "color": colours}))
        shader, batch = _cache[key]
        gpu.state.point_size_set(4.0)
        batch.draw(shader)

    if cameras:
        key = ("cameras", len(cameras), round(fade, 2))
        if key not in _cache:
            for old in [k for k in _cache if k[0] == "cameras"]:
                del _cache[old]
            coords, colours = _frustum_segments(cameras, state["scale"])
            colours[:, 3] *= fade
            shader = _shader('POLYLINE_FLAT_COLOR', 'FLAT_COLOR')
            _cache[key] = (shader, batch_for_shader(shader, 'LINES',
                                                    {"pos": coords, "color": colours}))
        shader, batch = _cache[key]
        region = bpy.context.region
        try:
            shader.uniform_float("viewportSize", (region.width, region.height))
            shader.uniform_float("lineWidth", 1.6)
        except (ValueError, AttributeError):
            pass
        batch.draw(shader)

    gpu.state.point_size_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_label():
    """One line at the bottom of the viewport saying what is happening."""
    import blf
    with _lock:
        active, label, finished = _state["active"], _state["label"], _state["finished"]
        regions = [(r["name"], r["colour"]) for r in _state["regions"]]
    if not active:
        return
    fade = 1.0
    if finished:
        fade = float(np.clip(1.0 - (time.time() - finished - HOLD_SECONDS) / FADE_SECONDS, 0, 1))
        if fade <= 0:
            return
    scale = bpy.context.preferences.system.ui_scale
    font = 0
    blf.size(font, 13 * scale)
    x, y = 18 * scale, 44 * scale
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 5, 0.0, 0.0, 0.0, 0.8 * fade)
    blf.color(font, 1.0, 1.0, 1.0, fade)
    blf.position(font, x, y, 0)
    blf.draw(font, f"Smart camera rig  ·  {label}")
    # A small legend of the systems found so far, in their preview colours.
    blf.size(font, 11 * scale)
    offset = x
    for name, (r, g, b) in regions:
        blf.color(font, r, g, b, fade)
        blf.position(font, offset, y - 18 * scale, 0)
        text = f"● {name}"
        blf.draw(font, text)
        offset += blf.dimensions(font, text)[0] + 14 * scale
    blf.disable(font, blf.SHADOW)
