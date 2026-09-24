# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Raw data export: every signal Blender can give a splat trainer.

Writes ``Dataset(Raw)`` beside the legacy ``Dataset(Default)`` of a build,
leaving the legacy files exactly as they were. See ``layout`` for the folder
contract and ``docs/RAW_DATASET.md`` for the format.

Module map:
    layout      folder contract and the per-view pass registry
    properties  settings (Scene.splatgen_raw)
    capture     reversible scene changes: passes, AOVs, ids, clay, outputs
    sessions    beauty / clay / world / probe render recipes
    hooks       the calls the legacy render batch makes
    geometry    scene mesh and collision voxels
    metadata    cameras, scene description, manifest
    stage       the post-build runner
    ui          settings panel and the Export Raw Data operator

Nothing here imports ``sceneray_splat`` at module level; it imports this
package, so those imports happen inside functions.
"""

import bpy

if "layout" in locals():
    from importlib import reload

    layout = reload(layout)
    properties = reload(properties)
    capture = reload(capture)
    sessions = reload(sessions)
    hooks = reload(hooks)
    geometry = reload(geometry)
    metadata = reload(metadata)
    stage = reload(stage)
    ui = reload(ui)
else:
    from . import layout
    from . import properties
    from . import capture
    from . import sessions
    from . import hooks
    from . import geometry
    from . import metadata
    from . import stage
    from . import ui


def _purge_after_register():
    try:
        capture.purge_leftovers()
    except Exception:
        pass
    return None


def register():
    properties.register()
    ui.register()
    stage.register()
    # bpy.data is restricted while an add-on is being enabled.
    bpy.app.timers.register(_purge_after_register, first_interval=0.5)


def unregister():
    stage.unregister()
    ui.unregister()
    properties.unregister()
