# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""The few places the legacy build calls into the raw export.

The beauty passes ride along with the legacy render - one render per camera
produces both the legacy RGB and every raw beauty pass - so the most
expensive render is never repeated. Everything here is best-effort: a raw
failure is reported and left for the raw stage to fill in, and never stops
or alters the legacy dataset.
"""

import shutil
from pathlib import Path

import bpy

from . import layout, properties, sessions


def _log(message):
    print(f"[splatgen raw] {message}")


def batch_begin(batch):
    """Called by the legacy render setup, after its own capture exists."""
    batch._raw_capture = None
    if not getattr(batch, "queue", None) or not properties.enabled(batch.scene):
        return
    try:
        batch._raw_capture = sessions.begin_beauty(
            batch.scene, bpy.context.view_layer,
            layout.raw_root(batch.output_dir), own_tree=False,
        )
    except Exception as exc:
        batch._raw_capture = None
        _log(f"beauty passes are not captured with this render ({exc}); "
             "the raw stage will render them separately")


def batch_prepare(batch, frame_index):
    session = getattr(batch, "_raw_capture", None)
    if session is not None:
        session.prepare_view(layout.frame_stem(frame_index))


def batch_finish(batch, camera_name, frame_index):
    session = getattr(batch, "_raw_capture", None)
    if session is None:
        return
    try:
        _published, missing = session.finish_view(layout.frame_stem(frame_index))
    except Exception as exc:
        missing = [str(exc)]
    if missing:
        _log(f"'{camera_name}' is missing raw beauty passes {missing}; "
             "the raw stage will render them separately")


def batch_restore(batch):
    """Undo every raw scene change; runs before the legacy restore."""
    session = getattr(batch, "_raw_capture", None)
    batch._raw_capture = None
    if session is not None:
        session.restore()


def copy_view(previous_build, target_build, frame_index):
    """Carry a reused camera's raw per-view files into the new build."""
    if previous_build is None:
        return
    source_root = layout.raw_root(previous_build)
    if not source_root.is_dir():
        return
    target_root = layout.raw_root(target_build)
    try:
        id_map = source_root / layout.ID_MAP
        if id_map.is_file() and not (target_root / layout.ID_MAP).is_file():
            (target_root / layout.ID_MAP).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(id_map, target_root / layout.ID_MAP)
        for key in layout.PASSES:
            source = layout.pass_file(source_root, key, frame_index)
            if source.is_file():
                destination = layout.pass_file(target_root, key, frame_index)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    except OSError as exc:
        _log(f"could not carry raw files for frame {frame_index} forward: {exc}")
