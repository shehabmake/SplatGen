# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

"""SplatGen Prepare: camera placement and dataset generation only."""
import bpy
if bpy.app.version[:2] != (5, 3):
    raise RuntimeError("SplatGen Prepare 5.3 requires Blender 5.3.x.")

from . import version
from . import edition

if "icons" in locals():
    from importlib import reload
    reload(version)
    reload(edition)
    reload(icons)
    reload(diagnostics)
    reload(theme)
    reload(workspace_ui)
    reload(progress)
    reload(sceneray_splat)
    reload(building_data)
    reload(raw_export)
    reload(auto_rig)
else:
    from . import icons
    from . import diagnostics
    from . import theme
    from . import workspace_ui
    from . import progress
    from . import sceneray_splat
    from . import building_data
    from . import raw_export
    from . import auto_rig

import bpy


bl_info = {
    # Keep these values literal: Blender discovers legacy add-ons by parsing
    # bl_info with ast.literal_eval before importing the package, so nothing
    # here may be computed - build_editions.py rewrites the two lines below
    # for the Standard package instead.
    "name": "SplatGen Prepare",
    "description": "Place cameras and build image, camera and point-cloud datasets.",
    "author": "JELLY FISH STUDIO <shehabmekkyb010@gmail.com>",
    "blender": (5, 3, 0),
    "version": (5, 3, 0),
    "support": "COMMUNITY",
    "category": "Scene",
    "location": "Scene Properties > SplatGen or 3D View > Sidebar > SplatGen",
}


def register():
    from . import camera_overlay

    # Undo even a partially registered module on failure. Otherwise Blender
    # leaves working panel classes behind with missing settings/operators.
    steps = [icons, diagnostics, progress, sceneray_splat, building_data,
             raw_export, auto_rig]
    attempted = []
    try:
        for module in steps:
            attempted.append(module)
            module.register()
        camera_overlay.enable()
    except Exception:
        import traceback
        camera_overlay.disable()
        for module in reversed(attempted):
            try:
                module.unregister()
            except Exception:
                traceback.print_exc()
        raise


def unregister():
    from . import camera_overlay
    camera_overlay.disable()
    auto_rig.unregister()
    raw_export.unregister()
    building_data.unregister()
    sceneray_splat.unregister()
    progress.unregister()
    diagnostics.unregister()
    icons.unregister()


if __name__ == "__main__":
    register()
