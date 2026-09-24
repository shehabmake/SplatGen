# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""SplatGen's palette and compact native UI components.

Exact palette colours are carried by custom transparent icons. Buttons and
progress fills honour the user's Blender theme; no global theme is modified.
"""

STATUS_OK = 'CHECKMARK'
STATUS_WAIT = 'TIME'
STATUS_ERROR = 'ERROR'
STATUS_IDLE = 'RADIOBUT_OFF'
STATUS_ACTIVE = 'PLAY'
from . import palette

ACCENT = palette.BLUE
WARM = palette.ORANGE
ERROR = palette.RED
CAMERA_PENDING = 'TIME'
CAMERA_RENDERED = 'CHECKMARK'
CAMERA_PENDING_RGBA = (*palette.rgb(palette.ORANGE), 1.0)
CAMERA_RENDERED_RGBA = (*palette.rgb(palette.BLUE), 1.0)

ICON_MAP = {
    'CHECKMARK': 'status_ok', 'TIME': 'status_wait', 'ERROR': 'status_error',
    'RADIOBUT_OFF': 'status_idle', 'PLAY': 'action_resume', 'PAUSE': 'action_pause',
    'CANCEL': 'action_stop', 'X': 'action_close', 'TRASH': 'action_delete',
    'PREFERENCES': 'ui_settings', 'HELP': 'ui_help',
    'VIEWZOOM': 'ui_search', 'FILTER': 'ui_changed', 'IMPORT': 'action_import',
    'EXPORT': 'action_export', 'CAMERA_DATA': 'stage_cameras',
    'OUTLINER_OB_CAMERA': 'stage_cameras', 'OUTLINER_OB_POINTCLOUD': 'stage_points',
    'SHADING_RENDERED': 'stage_viewer', 'FILE_FOLDER': 'stage_project',
    'ADD': 'action_add', 'REMOVE': 'action_remove', 'ZOOM_SELECTED': 'action_frame',
    'FILE_TICK': 'ui_save', 'SCENE_DATA': 'layer_scene', 'SEQ_STRIP_META': 'stage_dataset', 'RENDER_STILL': 'stage_dataset',
    'SHADERFX': 'stage_training', 'FILE_REFRESH': 'ui_reset',
}


def icon_args(icon='NONE', custom=None):
    value = custom_icon(custom or ICON_MAP.get(icon))
    return {'icon_value': value} if value else {'icon': icon}


def operator(layout, idname, *, text='', icon='NONE', custom=None, **kwargs):
    return layout.operator(idname, text=text, **icon_args(icon, custom), **kwargs)


def primary(layout, idname, text, custom, *, enabled=True):
    row = layout.row(align=True)
    row.enabled = enabled
    row.scale_y = 1.7
    row.active_default = enabled
    return operator(row, idname, text=text, custom=custom, icon='PLAY')


def disclosure(layout, data, prop, title, icon='NONE'):
    row = layout.row(align=True)
    row.use_property_split = False
    row.alignment = 'LEFT'
    row.prop(data, prop, text=title, emboss=False,
             icon='TRIA_DOWN' if getattr(data, prop) else 'TRIA_RIGHT')
    if not getattr(data, prop):
        return None
    return form(layout)


def custom_icon(name):
    """Resolve one optional brand icon without introducing an import cycle."""
    from . import icons

    return icons.icon_id(name)


def icon_label(layout, text, custom=None, fallback='NONE'):
    """Draw a label with colour artwork and a native-icon fallback."""
    layout.label(text=text, **icon_args(fallback, custom))


def muted(layout, text, icon=None):
    """Supporting copy with normal contrast, separated by layout rather than dimming.

    ``active=False`` also attenuates custom icon artwork and makes useful hints
    look disabled. Reserve that treatment for genuinely unavailable controls.
    """
    row = layout.row()
    message(row, text, icon=icon)
    return row


def heading(layout, title, *, custom=None, fallback='NONE', detail=None,
            status_state=None):
    """A consistent card heading with optional detail and semantic state."""
    header = layout.row(align=True)
    header.scale_y = 1.15
    icon_label(header, title.title() if title.isupper() else title, custom=custom, fallback=fallback)
    if status_state:
        state = header.row(align=True)
        state.alignment = 'RIGHT'
        icon_label(state, "", fallback=status_state)
    if detail:
        muted(layout, detail)
    return header


def message(layout, text, icon=None):
    """Wrap supporting copy in narrow sidebars instead of clipping the fix."""
    import textwrap
    import bpy
    region = getattr(bpy.context, 'region', None)
    scale = getattr(bpy.context.preferences.system, 'ui_scale', 1.0) or 1.0
    width = max(24, min(100, int((getattr(region, 'width', 380) / scale - 90) / 7)))
    column = layout.column(align=True)
    for i, line in enumerate(textwrap.wrap(str(text), width=width) or ['']):
        icon_label(column, line, fallback=icon if icon and i == 0 else 'NONE')
    return column


def form(layout):
    """Consistent spacing and labels for every editable settings group."""
    body = layout.column()
    body.use_property_split = True
    body.use_property_decorate = False
    body.scale_y = 1.0
    return body


def card(layout, title, icon='NONE', detail=None):
    box = layout.box()
    heading(box, title, fallback=icon)
    if detail:
        muted(box, detail)
    box.separator(factor=.3)
    return box


def hero(layout, *, title, subtitle="", version_text=""):
    """Draw the branded masthead shared by both supported panel locations."""
    from . import icons

    card = layout.box()
    row = card.row(align=True)
    row.scale_y = 1.35
    logo = icons.logo_icon_id()
    if logo:
        row.label(text="", icon_value=logo)
    else:
        row.label(text="", icon='OUTLINER_OB_POINTCLOUD')
    copy = row.column(align=True)
    copy.label(text=title)
    if subtitle:
        sub = copy.row(align=True)
        sub.label(text=subtitle)
    if version_text:
        badge = row.row(align=True)
        badge.alignment = 'RIGHT'
        badge.label(text=version_text)
    return card


def camera_icon(rendered):
    """Neutral camera glyph, with a checkmark once rendered."""
    return CAMERA_RENDERED if rendered else CAMERA_PENDING


def camera_rgba(rendered):
    return CAMERA_RENDERED_RGBA if rendered else CAMERA_PENDING_RGBA


def status(layout, text, state=None, alert=False):
    """One line of status, styled the same everywhere in the add-on.

    Every panel used to report itself differently - a titled box here, a
    paragraph there, a bare label somewhere else. This is the single form:
    a coloured glyph saying what state it is in, and one short line saying
    what that means. Nothing else.
    """
    row = layout.row(align=True)
    if alert:
        row.alert = True
    message(row, text, icon=state or STATUS_IDLE)
    return row


def stat(layout, pairs):
    """A compact two-column read-out of name/value facts.

    For the numbers worth showing - point estimates, camera counts, losses -
    where a sentence per fact would cost five rows to say what a grid says in
    two.
    """
    grid = layout.column(align=True)
    for name, value in pairs:
        row = grid.split(factor=.48, align=True)
        left = row.row()
        left.label(text=str(name))
        right = row.row()
        right.alignment = 'RIGHT'
        right.label(text=str(value))
    return grid


def accent(layout, on=True):
    """Mark ``layout`` as holding the next action worth taking.

    Blender draws an ``active_default`` button in the theme accent colour. Only
    ever applied to one button per section, and only while pressing it would
    actually do something - an accent that is always on stops meaning anything.
    """
    layout.active_default = bool(on)
    return layout
