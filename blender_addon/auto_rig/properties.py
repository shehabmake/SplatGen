# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Smart camera rig settings: the list of systems and the calculation options.

A *system* is one thing the rig must capture. Interior and Exterior systems
own a *blob* - a sphere Empty the user moves into the space. Object /
Collection systems point at an object or a collection and have no blob.
Systems are planned together, so an exterior, the rooms inside and a few
objects that deserve extra views can all be one calculation.
"""

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

SYSTEM_KINDS = [
    ('INTERIOR', "Interior", "Capture the enclosed space around a blob: a room, a hall, "
     "a corridor. Move the blob inside it", 'HOME', 0),
    ('EXTERIOR', "Exterior", "Capture the outside around a blob: the facades and "
     "surroundings of a building. Move the blob outside it", 'WORLD', 1),
    ('OBJECT', "Object / Collection", "Orbit chosen objects from every side. On top "
     "of another system it gives those objects extra views", 'OBJECT_DATA', 2),
]
KIND_ICONS = {kind: icon for kind, _label, _tip, icon, _n in SYSTEM_KINDS}
KIND_LABELS = {kind: label for kind, label, _tip, _icon, _n in SYSTEM_KINDS}


def _redraw(self, context):
    from .. import sceneray_splat
    sceneray_splat._tag_redraw_sceneray_splat(context)


def _kind_changed(self, context):
    """Interior/Exterior need a blob; an Object system must not keep one."""
    from . import operators
    operators.sync_system_blob(context, self)
    _redraw(self, context)


def _name_changed(self, context):
    from . import operators
    operators.sync_system_blob(context, self, rename=True)


def _custom_changed(system, context):
    """Start custom placement from the current system defaults, not from zero."""
    if not system.use_custom or context is None or context.scene is None:
        return
    defaults = context.scene.splatgen_auto_rig
    for name in ("clearance", "seal_size", "limit_height", "height_min",
                 "height_max", "allow_below"):
        setattr(system, name, getattr(defaults, name))
    _redraw(system, context)


def _active_changed(self, context):
    """Picking a system in the list selects its blob in the viewport."""
    from . import operators
    operators.select_active_system(context)


class SplatGenBlobSettings(PropertyGroup):
    """Stored on an Empty: marks it as the blob of a rig system."""

    is_blob: BoolProperty(name="Scan Blob", default=False)
    mode: EnumProperty(
        name="Space",
        items=[('AUTO', "Auto", ""), ('INTERIOR', "Interior", ""),
               ('EXTERIOR', "Exterior", "")],
        default='AUTO',
    )
    bounded: BoolProperty(name="Limit to Radius", default=False)


class SplatGenRigSystem(PropertyGroup):
    """One entry of the systems list."""

    name: StringProperty(name="Name", default="System", update=_name_changed)
    enabled: BoolProperty(
        name="Use", default=True,
        description="Include this system when calculating cameras")
    kind: EnumProperty(
        name="Type", items=SYSTEM_KINDS, default='INTERIOR', update=_kind_changed)
    blob: PointerProperty(
        name="Blob", type=bpy.types.Object,
        description="The sphere Empty marking this space. Move it inside the room "
                    "(Interior) or outside the building (Exterior)")
    source: EnumProperty(
        name="Source",
        items=[('OBJECT', "Object", "Orbit one object"),
               ('COLLECTION', "Collection", "Orbit every object in a collection")],
        default='OBJECT', update=_redraw)
    target_object: PointerProperty(
        name="Object", type=bpy.types.Object,
        description="The object to capture from every side")
    target_collection: PointerProperty(
        name="Collection", type=bpy.types.Collection,
        description="The objects of this collection are captured together")
    views: IntProperty(
        name="Views per Surface",
        description=(
            "How many distinct, good views every surface of this system should get. "
            "3 trains clean Splats; raise it for reflective or detailed areas"
        ),
        default=3, min=1, max=8)
    bounded: BoolProperty(
        name="Limit to Radius",
        description=(
            "Only capture inside the blob's sphere. Scale the blob to set it. "
            "Use it for a room without a ceiling or to focus on part of a huge scene"
        ),
        default=False, update=_redraw)
    reach: EnumProperty(
        name="Reach",
        items=[('NEARBY', "Nearby", "Capture the buildings and objects around the "
                "blob, at a finer resolution"),
               ('SCENE', "Whole scene", "Capture everything outside, across the scene")],
        default='NEARBY')
    max_cameras: IntProperty(
        name="Cameras",
        description=(
            "How many cameras this system gets. 0 is automatic: as many as its surfaces "
            "need. With a number, Stop When Covered decides whether it is a limit or an "
            "exact count. All systems together stay within Max Cameras (total)"
        ),
        default=0, min=0, max=5000, update=_redraw)
    stop_when_covered: BoolProperty(
        name="Stop When Covered",
        description=(
            "Finish this system early once its surfaces have all their views, even before "
            "it has used all its cameras. Off: it always gets exactly its Cameras count"
        ),
        default=True)
    # ---- own placement, overriding the defaults in the settings popover ----
    use_custom: BoolProperty(
        name="Custom Placement",
        description=(
            "Give this system its own distance from surfaces, doorway width and camera "
            "heights instead of the system defaults in the calculation settings"
        ),
        default=False, update=lambda self, context: _custom_changed(self, context))
    clearance: FloatProperty(
        name="Distance from Surfaces",
        description="How close cameras may come to walls, furniture and objects. More keeps cameras out in the open for wide views; less lets them into tight spots. Passages too narrow for this distance still get cameras along their middle, so tunnels and hallways are never skipped. 0 picks a distance from the size of each space",
        default=0.3, min=0.0, soft_max=2.0, subtype='DISTANCE', unit='LENGTH')
    seal_size: FloatProperty(
        name="Doorway Width",
        description="The widest opening that counts as a door or window. Closing openings up to this width is how a room is told apart from the outside and from the next room: an Interior system stays in its room, an Exterior system stays out of the rooms. Narrower passages - tunnels, corridors, closets - stay part of the space. Raise it for wide openings such as garage doors or glass fronts",
        default=1.5, min=0.0, soft_max=4.0, subtype='DISTANCE', unit='LENGTH')
    limit_height: BoolProperty(
        name="Height Range",
        description="Keep this system's cameras within a height range above the floor",
        default=False)
    height_min: FloatProperty(
        name="Lowest", default=0.5, min=0.0, subtype='DISTANCE', unit='LENGTH')
    height_max: FloatProperty(
        name="Highest", default=2.2, min=0.0, subtype='DISTANCE', unit='LENGTH')
    allow_below: BoolProperty(
        name="Cameras Below Ground",
        description="Let this exterior's cameras go below the lowest surface",
        default=False)


class SplatGenAutoRigSettings(PropertyGroup):
    """Scene-wide settings of the Smart camera rig."""

    systems: CollectionProperty(type=SplatGenRigSystem)
    active_system: IntProperty(default=0, min=0, update=_active_changed)

    quality: EnumProperty(
        name="Quality",
        items=[
            ('DRAFT', "Draft", "Coarse scene analysis in a few seconds"),
            ('STANDARD', "Standard", "Balanced analysis for most scenes"),
            ('HIGH', "High", "Fine analysis: small rooms, narrow passages, clutter. Slower"),
        ],
        default='STANDARD',
    )
    max_cameras: IntProperty(
        name="Max Cameras (total)",
        description=(
            "Upper limit on the number of cameras across every system together. Systems "
            "given a camera count are served first"
        ),
        default=500, min=4, max=10000, update=_redraw,
    )
    clearance: FloatProperty(
        name="Distance from Surfaces",
        description="Default for systems. How close cameras may come to walls, furniture and objects. More keeps cameras out in the open for wide views; less lets them into tight spots. Passages too narrow for this distance still get cameras along their middle, so tunnels and hallways are never skipped. 0 picks a distance from the size of each space",
        default=0.3, min=0.0, soft_max=2.0, subtype='DISTANCE', unit='LENGTH',
    )
    seal_size: FloatProperty(
        name="Doorway Width",
        description="Default for systems. The widest opening that counts as a door or window. Closing openings up to this width is how a room is told apart from the outside and from the next room: an Interior system stays in its room, an Exterior system stays out of the rooms. Narrower passages - tunnels, corridors, closets - stay part of the space. Raise it for wide openings such as garage doors or glass fronts",
        default=1.5, min=0.0, soft_max=4.0, subtype='DISTANCE', unit='LENGTH',
    )
    limit_height: BoolProperty(
        name="Height Range",
        description="Default for systems: keep cameras within a height range above the floor",
        default=False,
    )
    height_min: FloatProperty(
        name="Lowest", default=0.5, min=0.0, subtype='DISTANCE', unit='LENGTH',
        description="Lowest camera height above the floor")
    height_max: FloatProperty(
        name="Highest", default=2.2, min=0.0, subtype='DISTANCE', unit='LENGTH',
        description="Highest camera height above the floor")
    allow_below: BoolProperty(
        name="Cameras Below Ground",
        description="Default for systems: let exterior cameras go below the lowest surface",
        default=False,
    )
    geometry_collection: PointerProperty(
        name="Scene Geometry",
        type=bpy.types.Collection,
        description="Only this collection's objects are analysed and block views. "
                    "Empty uses every render-visible object",
    )
    replace_previous: BoolProperty(
        name="Replace Previous Cameras",
        description="Delete the cameras of the previous calculation before creating new ones. "
                    "Cameras you placed yourself are never deleted",
        default=True,
    )
    live_preview: BoolProperty(
        name="Live Preview",
        description="Show the analysis in the 3D Viewport while it runs: surfaces, the space "
                    "each system fills, candidate viewpoints and each chosen camera",
        default=True,
    )
    seed: IntProperty(
        name="Seed", default=0, min=0,
        description="Change for a different, equally good arrangement",
    )
    camera_method: EnumProperty(
        name="Camera method",
        description="Choose how to add cameras to the shared queue",
        items=[('SMART', "Smart rig", "Calculate viewpoints for rooms, buildings or objects"),
               ('MANUAL', "Manual", "Add the current view with its shortcut, or create cameras from mesh faces"),
               ('SAVED', "Saved rigs", "Reuse a bundled rig or save your own camera arrangement")],
        default='SMART', update=_redraw,
    )
    show_manual: BoolProperty(name="Add cameras", default=True)
    show_saved: BoolProperty(name="Saved camera rigs", default=True)
    # ---- last result, shown under the button ---------------------------
    last_summary: StringProperty(default="", options={'SKIP_SAVE'})
    last_warning: StringProperty(default="", options={'SKIP_SAVE'})


CLASSES = (SplatGenBlobSettings, SplatGenRigSystem, SplatGenAutoRigSettings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.splatgen_blob = PointerProperty(type=SplatGenBlobSettings)
    bpy.types.Scene.splatgen_auto_rig = PointerProperty(type=SplatGenAutoRigSettings)


def unregister():
    for owner, name in ((bpy.types.Scene, "splatgen_auto_rig"),
                        (bpy.types.Object, "splatgen_blob")):
        if hasattr(owner, name):
            delattr(owner, name)
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
