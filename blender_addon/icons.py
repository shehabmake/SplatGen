# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""SplatGen's transparent, semantic colour icon set.

Blender deliberately leaves add-on panels in the user's theme and most of its
built-in icons are monochrome. Preview icons are the supported way to add a
small amount of brand colour without repainting Blender. Routine glyphs use
off-white for contrast on gray controls. The artwork is generated from source in
``icons/make_ui_icons.py`` with editable SVG source beside each raster asset.
The same symbols identify headings, actions, visible layers, and progress.
There are no theme colour-set swatches or emoji in this family.
"""

from pathlib import Path

import bpy
import bpy.utils.previews

LOGO = "splatgen_logo"
PROJECT = "stage_project"
CAMERAS = "stage_cameras"
DATASET = "stage_dataset"
POINTS = "stage_points"
TRAINING = "stage_training"
VIEWER = "stage_viewer"
EXPORT = "stage_export"
CAMERA_PENDING = "camera_pending"
CAMERA_RENDERED = "camera_rendered"
BUILD = "action_build"
TRAIN = "action_train"
IMPORT = "action_import"
EXPORT_ACTION = "action_export"
PAUSE = "action_pause"
STOP = "action_stop"
RESUME = "action_resume"
ADD = "action_add"
REMOVE = "action_remove"
FRAME = "action_frame"
SETTINGS = "ui_settings"
HELP = "ui_help"
SEARCH = "ui_search"
CHANGED = "ui_changed"
RESET = "ui_reset"
SAVE = "ui_save"
ACTIVE = "status_active"
OK = "status_ok"
WAIT = "status_wait"
ERROR = "status_error"
IDLE = "status_idle"
NEW = "badge_new"
SPLATS = "layer_splats"
COVERAGE = "layer_coverage"
RELIGHT = "viewer_relight"
SCENE = "layer_scene"
AUTO_RIG = "method_auto_rig"
BLOB = "method_blob"
BUG = "ui_bug"
OPEN_FOLDER = "action_open_folder"

ASSETS = {
    LOGO: "splatgen_logo.png",
    PROJECT: "stage_project.png",
    CAMERAS: "stage_cameras.png",
    DATASET: "stage_dataset.png",
    POINTS: "stage_points.png",
    TRAINING: "stage_training.png",
    VIEWER: "stage_viewer.png",
    EXPORT: "stage_export.png",
    CAMERA_PENDING: "camera_pending.png",
    CAMERA_RENDERED: "camera_rendered.png",
}
ASSETS.update({name: name + ".png" for name in (
    'section_cameras', 'section_coverage', 'section_dataset',
    'action_close', 'action_delete', 'action_hide',
    BUILD, TRAIN, IMPORT, EXPORT_ACTION, PAUSE, STOP, RESUME, ADD, REMOVE,
    FRAME, SETTINGS, HELP, SEARCH, CHANGED, RESET, SAVE, ACTIVE, OK, WAIT,
    ERROR, IDLE, NEW, SPLATS, COVERAGE, SCENE, RELIGHT, AUTO_RIG, BLOB, BUG, OPEN_FOLDER, 'action_calculate', 'action_coverage', 'action_save', 'coverage_fair', 'coverage_weak',
)})

#: The one preview collection this add-on owns, or None while unregistered.
_previews = None


def _asset_path(filename):
    return Path(__file__).parent / "icons" / filename


def icon_id(name):
    """Return a preview id, or zero when an optional asset is unavailable."""
    if _previews is None:
        return 0
    entry = _previews.get(name)
    return entry.icon_id if entry is not None else 0


def logo_icon_id():
    """The icon id for ``icon_value=``, or 0 when the logo is unavailable.

    Zero is Blender's "no icon" value, so a missing or unreadable file simply
    draws nothing rather than breaking every panel that asks for it.
    """
    return icon_id(LOGO)


def register():
    global _previews
    if _previews is not None:
        return
    collection = bpy.utils.previews.new()
    loaded = 0
    for name, filename in ASSETS.items():
        path = _asset_path(filename)
        if not path.is_file():
            continue
        try:
            collection.load(name, str(path), 'IMAGE')
            loaded += 1
        except Exception as exc:  # artwork must never stop the add-on
            print(f"[splatgen] could not load icon {filename}: {exc}")
    if not loaded:
        bpy.utils.previews.remove(collection)
        return
    _previews = collection


def unregister():
    global _previews
    if _previews is None:
        return
    try:
        bpy.utils.previews.remove(_previews)
    except Exception:
        from . import diagnostics as _diag
        _diag.swallowed("icons.py")
    _previews = None
