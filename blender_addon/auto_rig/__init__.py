# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Smart camera rig: camera placement for rooms, exteriors and objects.

Module map:
    planner      the view-planning algorithm - pure numpy, runs on a worker
    geometry     unique meshes plus instance transforms, sampled by area
    scene_proxy  reads the render-visible scene into ``geometry`` (main thread)
    properties   the systems list, settings, and the blob marker on Empties
    preview      live viewport drawing of the calculation
    operators    add/remove systems, place blobs, calculate, remove cameras
    ui           the Smart camera rig card, systems list and settings popover
    coverage     grading how well queued cameras cover the scene (numpy)
    coverage_ops Check coverage: operators, viewport overlay and card

``planner`` and ``geometry`` import nothing from Blender, so the algorithm
can be exercised without it.
"""

import bpy

if "planner" in locals():
    from importlib import reload

    planner = reload(planner)
    geometry = reload(geometry)
    scene_proxy = reload(scene_proxy)
    properties = reload(properties)
    preview = reload(preview)
    operators = reload(operators)
    ui = reload(ui)
    coverage = reload(coverage)
    coverage_ops = reload(coverage_ops)
else:
    from . import planner
    from . import geometry
    from . import scene_proxy
    from . import properties
    from . import preview
    from . import operators
    from . import ui
    from . import coverage
    from . import coverage_ops

_CLASSES = (*operators.CLASSES, *ui.CLASSES)


def register():
    properties.register()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    coverage_ops.register()


def unregister():
    operators.request_cancel()
    preview.clear()
    coverage_ops.unregister()
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    properties.unregister()
