# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Building Data: turning a camera setup into a training dataset.

Two independent operations, so each can grow on its own:

    Rendering Images -> RGB + masks + metric depth + cameras.txt + images.txt
    Point Cloud      -> RGB-D back-projection + fusion -> points3D.txt

Module map:
    paths     the on-disk shape, build folders and recovery rotation
    dataset_export  canonical masks/depth and optional Blender numerical passes
    pointcloud  per-camera back-projection and multi-view fusion (numpy only)
    manifest  what each image was rendered with, and what changed since
    stages    the two build operators
    ui        the Building Data panel

None of these may import ``sceneray_splat`` at import time: that module
imports ``paths`` and ``manifest`` directly, and a module-level import back
would close the cycle. Import it inside functions instead.
"""

import bpy

if "paths" in locals():
    from importlib import reload

    paths = reload(paths)
    manifest = reload(manifest)
    if "dataset_export" in locals():
        dataset_export = reload(dataset_export)
    else:
        from . import dataset_export
    if "pointcloud" in locals():
        pointcloud = reload(pointcloud)
    else:
        from . import pointcloud
    raycast = reload(raycast)
    stages = reload(stages)
    ui = reload(ui)
else:
    from . import paths
    from . import manifest
    from . import dataset_export
    from . import pointcloud
    from . import raycast
    from . import stages
    from . import ui

# Public API used by the rest of the add-on.
from .ui import SPLATRAY_PT_building_data  # noqa: F401
from .ui import draw_build_dataset  # noqa: F401
from .ui import draw_pointcloud_settings  # noqa: F401
from .ui import draw_render_settings  # noqa: F401

# ui.SPLATRAY_PT_building_data is deliberately absent: like the other workflow
# sections it is drawn through a proxy layout, not owned by Blender.
_CLASSES = stages.CLASSES


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    raycast.register()


def unregister():
    raycast.unregister()
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
