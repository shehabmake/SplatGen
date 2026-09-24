# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""One progress report for every long-running operation.

Whatever is running - rendering, point sampling, training, or something added
later - reports through here, and it appears in Blender's status bar and,
while rendering, in the render window too.

The state lives in module globals rather than in the scene, so it cannot be
saved into a .blend and reappear in an unrelated project. A load handler
clears it as well, because a file opened mid-run must not inherit the bar of
whatever was running before.

Usage::

    progress.begin("Build Dataset", ("Rendering Images", "Point Cloud"))
    progress.set_stage("Rendering Images")
    progress.update(fraction=0.4, detail="12/30 images", eta="1m 20s")
    progress.set_stage("Point Cloud")
    ...
    progress.end("Dataset complete")
"""

import time
from pathlib import Path

import bpy

from . import theme

_IDLE = {
    "active": False,
    "operation": "",
    "stages": (),
    "stage": "",
    "fraction": 0.0,
    "message": "",
    "detail": "",
    "eta": "",
    "started_at": 0.0,
    "finished_message": "",
    "finished_state": "DONE",
    #: One record per phase, so a finished phase keeps its own result and a
    #: pending one can be shown before it starts. Without this the interface
    #: could only ever describe the single thing running right now.
    "phases": [],
}

_state = dict(_IDLE)
_state["phases"] = []

PENDING = "PENDING"
RUNNING = "RUNNING"
DONE = "DONE"
FAILED = "FAILED"


def _new_phase(name):
    return {
        "name": str(name),
        "state": PENDING,
        "fraction": 0.0,
        "message": "",
        "detail": "",
        "eta": "",
        "started_at": 0.0,
        "finished_at": 0.0,
        "result": "",
    }


def _phase(name):
    for entry in _state["phases"]:
        if entry["name"] == name:
            return entry
    return None


def current_phase():
    return _phase(_state["stage"])


def phase_elapsed(entry):
    if not entry["started_at"]:
        return ""
    end = entry["finished_at"] or time.time()
    return format_seconds(end - entry["started_at"])


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def begin(operation, stages=()):
    """Start reporting. ``stages`` names the phases this operation runs."""
    names = tuple(str(stage) for stage in stages) or (str(operation),)
    _state.update(_IDLE)
    _state["active"] = True
    _state["operation"] = str(operation)
    _state["stages"] = names
    _state["phases"] = [_new_phase(name) for name in names]
    _state["started_at"] = time.time()
    # Nothing is running until a phase is entered, so the whole plan is
    # visible up front rather than appearing one step at a time.
    _state["stage"] = ""
    set_stage(names[0])


def set_stage(stage, message=""):
    """Enter a phase, completing whichever one was running.

    Phases are sequential, so entering one is also the signal that the
    previous finished - which is what lets the interface show a completed
    phase with its own result instead of discarding it.
    """
    if not _state["active"]:
        begin(stage, (stage,))
        return
    name = str(stage)
    previous = current_phase()
    if previous is not None and previous["name"] != name:
        if previous["state"] == RUNNING:
            previous["state"] = DONE
            previous["fraction"] = 1.0
            previous["finished_at"] = time.time()
            if not previous["result"]:
                previous["result"] = previous["message"] or "Finished"

    entry = _phase(name)
    if entry is None:
        entry = _new_phase(name)
        _state["phases"].append(entry)
        _state["stages"] = tuple(_state["stages"]) + (name,)
    entry["state"] = RUNNING
    entry["started_at"] = entry["started_at"] or time.time()
    entry["fraction"] = 0.0
    entry["detail"] = ""
    entry["eta"] = ""
    if message:
        entry["message"] = str(message)

    _state["stage"] = name
    _state["fraction"] = 0.0
    _state["detail"] = ""
    _state["eta"] = ""
    if message:
        _state["message"] = str(message)
    _redraw()


def update(fraction=None, message=None, detail=None, eta=None):
    if not _state["active"]:
        return
    entry = current_phase()
    if fraction is not None:
        value = max(0.0, min(1.0, float(fraction)))
        _state["fraction"] = value
        if entry is not None:
            entry["fraction"] = value
    if message is not None:
        _state["message"] = str(message)
        if entry is not None:
            entry["message"] = str(message)
    if detail is not None:
        _state["detail"] = str(detail)
        if entry is not None:
            entry["detail"] = str(detail)
    if eta is not None:
        _state["eta"] = str(eta)
        if entry is not None:
            entry["eta"] = str(eta)
    _redraw()


def fail(message=""):
    """Mark the running phase as failed and stop, keeping the reason."""
    entry = current_phase()
    if entry is not None:
        entry["state"] = FAILED
        entry["finished_at"] = time.time()
        entry["result"] = str(message or entry["message"] or "Failed")
    end(message)


def end(message="", outcome=None):
    """Finish. The last message stays visible until something else starts."""
    finished = str(message or _state.get("message", ""))
    entry = current_phase()
    outcome = outcome or (FAILED if entry is not None and entry['state'] == FAILED else DONE)
    if entry is not None and entry["state"] == RUNNING:
        entry["state"] = DONE
        entry["fraction"] = 1.0
        entry["finished_at"] = time.time()
        entry["result"] = entry["message"] or "Finished"
    _state.update(_IDLE)
    _state["phases"] = []
    _state["finished_message"] = finished
    _state["finished_state"] = outcome
    _redraw()


def reset():
    """Drop everything, including the finished message."""
    _state.update(_IDLE)
    _state["phases"] = []
    _redraw()


def is_active():
    return bool(_state["active"])


def current():
    return dict(_state)


# --------------------------------------------------------------------------
# Derived values
# --------------------------------------------------------------------------

def overall_fraction():
    """Progress across every stage of the current operation.

    A single-stage operation fills the whole bar; Build Dataset's two stages
    each fill half, so the bar never restarts part-way through.
    """
    stages = _state["stages"] or ()
    if not stages:
        return 0.0
    try:
        index = stages.index(_state["stage"])
    except ValueError:
        index = 0
    return (index + _state["fraction"]) / max(1, len(stages))


def format_seconds(seconds):
    """A duration a person can read at a glance."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def elapsed_text():
    if not _state["active"] or not _state["started_at"]:
        return ""
    return format_seconds(time.time() - _state["started_at"])


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def _draw_bar(layout, factor, text, scale_x=1.0, scale_y=1.0):
    row = layout.row(align=True)
    row.scale_x = scale_x
    row.scale_y = scale_y
    try:
        row.progress(factor=max(0.0, min(1.0, float(factor))), text=text)
    except (AttributeError, TypeError):
        # Older UI API: a label still carries the same information.
        row.label(text=text)


def stage_position():
    stages = _state['stages']
    try:
        return stages.index(_state['stage']) + 1, len(stages)
    except ValueError:
        return 0, len(stages)


def status_line():
    """Operation, stage and percentage, in one compact line."""
    if not _state["active"]:
        return _state["finished_message"]
    index, total = stage_position()
    stage = _state["stage"]
    if total > 1:
        stage = f"{stage} ({index}/{total})"
    return (
        f"{_state['operation']} — {stage} — "
        f"{overall_fraction() * 100.0:.0f}%"
    )


def detail_line():
    """What is happening right now, plus what is left."""
    parts = []
    if _state["detail"]:
        parts.append(_state["detail"])
    elapsed = elapsed_text()
    if elapsed:
        parts.append(f"{elapsed} elapsed")
    if _state["eta"]:
        parts.append(f"~{_state['eta']} left")
    return " · ".join(parts)


def _draw_status_bar(self, context):
    """Appended to Blender's bottom status bar.

    The status bar is one row high, so the bar is widened rather than made
    taller and the text beside it carries the operation, stage, percentage and
    remaining time instead of a bare number.
    """
    if not _state["active"]:
        message = _state["finished_message"]
        if message:
            self.layout.label(text=f"SplatGen: {message[:90]}", icon='CHECKMARK' if _state['finished_state'] == DONE else 'INFO')
        return
    layout = self.layout
    layout.separator_spacer()
    row = layout.row(align=True)
    row.label(text="SplatGen", icon="SEQ_STRIP_META")
    # A wide bar: the default is a few characters across and unreadable at a
    # glance, which is what made progress look like it was not moving.
    _draw_bar(row, overall_fraction(), status_line()[:80], scale_x=3.0)
    detail = detail_line()
    if detail:
        row.label(text=detail[:70])
    if _state["message"]:
        row.label(text=_state["message"][:60])


def _draw_render_window(self, context):
    """Appended to the Image editor header, i.e. Blender's render window.

    Rendering hides the status bar behind the render window, so the same
    report is mirrored here rather than leaving that window silent.
    """
    if not _state["active"]:
        return
    space = getattr(context, "space_data", None)
    image = getattr(space, "image", None)
    if image is None or image.type != "RENDER_RESULT":
        return
    layout = self.layout
    layout.separator_spacer()
    row = layout.row(align=True)
    row.label(text="SplatGen", icon="SEQ_STRIP_META")
    _draw_bar(row, overall_fraction(), status_line()[:80], scale_x=3.0)
    detail = detail_line()
    if detail:
        row.label(text=detail[:60])


#: A coloured swatch per state, so the phase list can be read at a glance
#: without parsing any of the words on it.
_PHASE_ICONS = {
    PENDING: theme.STATUS_IDLE,
    RUNNING: theme.STATUS_ACTIVE,
    DONE: theme.STATUS_OK,
    FAILED: theme.STATUS_ERROR,
}


def _draw_phase(layout, entry, index, total):
    state = entry['state']
    row = layout.row(align=True)
    row.label(text=f"{index:02d}  {entry['name']}", icon=_PHASE_ICONS.get(state, 'RADIOBUT_OFF'))
    if state == RUNNING:
        _draw_bar(layout, entry['fraction'], f"{entry['fraction']:.0%}", scale_y=1.25)
        if entry['message']:
            theme.message(layout, entry['message'])
        if entry['detail']:
            theme.muted(layout, entry['detail'])
        theme.stat(layout, [('Elapsed', phase_elapsed(entry) or '0s'),
                            ('Remaining', entry['eta'] or 'Estimating…')])
    elif state == PENDING:
        theme.muted(layout, 'Queued')
    else:
        if entry['result']:
            theme.message(layout, entry['result'])
        if phase_elapsed(entry):
            theme.muted(layout, phase_elapsed(entry))


def draw_panel(layout):
    """The task, the active phase and one overall progress bar."""
    if not _state['active']:
        if _state['finished_message']:
            theme.status(layout, _state['finished_message'], theme.STATUS_OK if _state['finished_state'] == DONE else theme.STATUS_ERROR)
        return
    box = layout.box()
    theme.icon_label(box, _state['operation'] or 'SplatGen', custom='status_active')
    entry = current_phase()
    _draw_bar(box, overall_fraction(), f"Overall {overall_fraction():.0%} · {_state['stage']}", scale_y=1.3)
    if entry and entry['message']:
        theme.message(box, entry['message'])
    theme.stat(box, [('Elapsed', elapsed_text() or '0s'), ('Remaining', _state['eta'] or 'Estimating…')])
    cfg = getattr(bpy.context.scene, 'SCENERAY_SPLAT', None)
    details = theme.disclosure(box, cfg, 'show_job_details', 'Details') if cfg else box
    if details is not None:
        for index, phase in enumerate(_state['phases'],1):
            theme.icon_label(details, f"{index}. {phase['name']}", fallback=_PHASE_ICONS[phase['state']])
            if phase['detail']:theme.muted(details, phase['detail'])
            if phase['result']:theme.muted(details, phase['result'])


#: Which button owns which operation names, so only the running one draws a
#: bar. A build reports under three different names as it moves from
#: rendering to metadata to points, and all three belong to Build Dataset.
OWNER_COVERAGE = ("Camera Coverage",)
OWNER_BUILD = ("Build Dataset", "Render Images", "Generate Point Cloud")
OWNER_AUTO_RIG = ("Smart Camera Rig",)

#: Every operation this add-on considers exclusive. One at a time.
ALL_OWNERS = OWNER_COVERAGE + OWNER_BUILD + OWNER_AUTO_RIG


def owner():
    """The operation currently running, or "" when nothing is."""
    return _state["operation"] if _state["active"] else ""


def is_owner(names):
    """Whether the operation running right now is one of ``names``."""
    current = owner()
    if not current:
        return False
    if isinstance(names, str):
        names = (names,)
    return current in names


def control(row, kind, idname, text, icon):
    """One operation control, sized and coloured to be hard to miss.

    Blender gives an add-on exactly one colour for a button - ``alert``, which
    draws it in the theme's red - so red is spent on the one control that
    ends the work. The rest are separated by size, icon and label instead;
    there is no API for arbitrary button colours.
    """
    cell = row.row(align=True)
    cell.scale_x = 1.0
    cell.scale_y = 1.2
    if kind == 'STOP':
        cell.alert = True
    elif kind == 'PRIMARY':
        cell.active_default = True
    return theme.operator(cell, idname, text=text.title(), icon=icon)





def draw_active(layout, names, controls=None, context=None):
    """The one progress bar, drawn only by the operation that is running.

    ``controls`` is a callable that adds Pause/Stop beneath the bar, so every
    control for the running task sits with it rather than somewhere else in
    the panel.
    """
    if not is_owner(names):
        return False
    draw_panel(layout)
    if controls is not None:
        # Under the bar, at full width: these are the controls the user
        # reaches for while watching it, so they get room rather than a
        # cramped corner of the header.
        layout.separator()
        buttons = layout.row(align=True)
        controls(buttons)
    return True


def _redraw():
    window_manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {
                "STATUSBAR", "IMAGE_EDITOR", "VIEW_3D", "OUTLINER", "PROPERTIES"
            }:
                area.tag_redraw()


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

@bpy.app.handlers.persistent
def _reset_on_file_load(*_args):
    """A different project must never inherit the previous one's progress."""
    reset()


def register():
    bpy.types.STATUSBAR_HT_header.append(_draw_status_bar)
    bpy.types.IMAGE_HT_header.append(_draw_render_window)
    if _reset_on_file_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_reset_on_file_load)
    reset()


def unregister():
    while _reset_on_file_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reset_on_file_load)
    for header, function in (
        (bpy.types.STATUSBAR_HT_header, _draw_status_bar),
        (bpy.types.IMAGE_HT_header, _draw_render_window),
    ):
        try:
            header.remove(function)
        except (ValueError, RuntimeError):
            pass
    reset()
