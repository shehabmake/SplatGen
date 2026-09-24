"""Camera placement and dataset preparation workspace."""
from . import theme


def draw_header(layout, context):
    from . import icons, version
    row = layout.row(align=True)
    row.scale_y = 1.15
    row.label(text='SplatGen Prepare', icon_value=icons.logo_icon_id())
    version_row = row.row()
    version_row.alignment = 'RIGHT'
    version_row.label(text=version.addon_stringversion)
    theme.operator(row, 'splatray.open_preferences', icon='PREFERENCES')
    theme.operator(row, 'sceneray_splat.open_manual', text='Guide', icon='HELP')
    theme.operator(row, 'splatgen.copy_diagnostics', text='', custom='ui_bug', icon='INFO')
    layout.separator(factor=.5)



def draw_task(layout, context):
    from . import progress
    if progress.is_active():
        progress.draw_panel(layout)
        theme.operator(layout, 'sceneray_splat.stop_render', text='Stop', custom='action_stop', icon='CANCEL')
        layout.separator(factor=.5)
    elif progress.current()['finished_message']:
        row = layout.row(align=True)
        theme.status(row, progress.current()['finished_message'], theme.STATUS_OK if progress.current()['finished_state']=='DONE' else theme.STATUS_WAIT)
        theme.operator(row, 'sceneray_splat.dismiss_task', icon='X').token = 'PROGRESS'



def draw_camera_methods(layout, context):
    """One selected method, with all three entry points always visible."""
    from .auto_rig import ui as auto_rig_ui
    settings = context.scene.splatgen_auto_rig
    box = layout.box()
    theme.heading(box, '1  ·  Add cameras', custom='section_cameras')
    tabs = box.row(align=True)
    tabs.scale_y = 1.3
    for method, label, icon in (('SMART', 'Smart rig', 'method_auto_rig'),
                                ('MANUAL', 'Manual', 'stage_viewer'),
                                ('SAVED', 'Saved', 'ui_save')):
        op = theme.operator(tabs, 'sceneray_splat.camera_method', text=label, custom=icon,
                            icon='CAMERA_DATA', depress=settings.camera_method == method)
        op.method = method
    box.separator(factor=.5)
    if settings.camera_method == 'SMART':
        auto_rig_ui.draw_panel(box, context)
    elif settings.camera_method == 'MANUAL':
        draw_manual_cameras(box, context)
    else:
        draw_saved_rigs(box, context)


def draw_manual_cameras(layout, context):
    """Full-width actions remain readable in a narrow sidebar."""
    from . import sceneray_splat as sr
    from .building_data import stages
    body = layout.column()
    body.enabled = not stages.anything_running(context)
    shortcut = sr.sr_shortcut_label('sceneray_splat.add_camera_from_view')
    row = body.row(align=True)
    row.scale_y = 1.4
    theme.operator(row, 'sceneray_splat.add_camera_from_view',
                   text='Add from current view', custom='stage_viewer')
    if shortcut:
        hint = body.row()
        hint.alignment = 'RIGHT'
        hint.label(text=shortcut)
    row = body.row(align=True)
    row.scale_y = 1.25
    theme.operator(row, 'sceneray_splat.create_cameras_from_faces',
                   text='Cameras from mesh faces', custom='layer_scene')


def draw_saved_rigs(layout, context):
    from . import sceneray_splat as sr
    from .building_data import stages
    cfg = context.scene.SCENERAY_SPLAT
    body = layout.column()
    body.enabled = not stages.anything_running(context)
    sr._sr_request_rig_library_refresh(cfg)
    if cfg.rig_presets:
        body.template_list('SCENERAY_SPLAT_UL_rig_presets', '', cfg, 'rig_presets', cfg,
                           'active_preset_index', rows=3, maxrows=5)
        preset = cfg.rig_presets[min(max(0, cfg.active_preset_index), len(cfg.rig_presets) - 1)]
        row = body.row(align=True)
        row.scale_y = 1.35
        theme.operator(row, 'sceneray_splat.preset_load', text='Add rig at 3D Cursor', icon='IMPORT')
        row = body.row(align=True)
        theme.operator(row, 'sceneray_splat.preset_save', text='Save current rig', icon='FILE_TICK')
        edit = row.row(align=True)
        edit.enabled = not preset.builtin
        theme.operator(edit, 'sceneray_splat.preset_rename', text='', icon='GREASEPENCIL')
        theme.operator(edit, 'sceneray_splat.preset_delete', text='', icon='TRASH')
    else:
        theme.operator(body, 'sceneray_splat.preset_save', text='Save current rig', icon='FILE_TICK')


def draw_cameras(layout, context):
    from . import sceneray_splat as sr
    from .building_data import stages
    cfg = context.scene.SCENERAY_SPLAT
    busy = stages.anything_running(context)
    sr._sr_request_queue_sync(context.scene, cfg)
    sr._sr_apply_camera_status_colors(cfg)
    box = layout.box()
    count = len(cfg.camera_queue)
    header = box.row(align=True)
    theme.icon_label(header, '', custom='section_cameras')
    header.prop(cfg, 'show_prepare_queue', text=f'Camera queue  ·  {count}',
                emboss=False, icon='TRIA_DOWN' if cfg.show_prepare_queue else 'TRIA_RIGHT')
    tools = header.row(align=True)
    tools.enabled = not busy
    tools.menu('SPLATGEN_MT_queue_actions', text='', icon='DOWNARROW_HLT')
    tools.popover(panel='SPLATGEN_PT_camera_settings', text='', **theme.icon_args('PREFERENCES'))
    if not cfg.show_prepare_queue:
        return
    if count:
        prefs = sr._sr_addon_preferences()
        rows = min(getattr(prefs, 'camera_list_rows', 5), max(2, count))
        listing = box.column()
        listing.enabled = not busy
        listing.template_list('SCENERAY_SPLAT_UL_camera_queue', '', cfg, 'camera_queue', cfg,
                              'active_camera_index', rows=rows, maxrows=rows)
        nav = box.row(align=True)
        nav.enabled = not busy
        nav.scale_y = 1.2
        if cfg.review_active:
            theme.operator(nav, 'sceneray_splat.previous_camera', icon='TRIA_LEFT')
        theme.operator(nav, 'sceneray_splat.review_cameras',
                       text='End preview' if cfg.review_active else 'Preview cameras',
                       custom='action_close' if cfg.review_active else 'stage_viewer')
        if cfg.review_active:
            theme.operator(nav, 'sceneray_splat.next_camera', icon='TRIA_RIGHT')
        rendered = sum(item.render_state == 'RENDERED' for item in cfg.camera_queue if item.camera)
        stats = box.row(align=True)
        theme.icon_label(stats, f'{rendered} rendered', custom='camera_rendered')
        tail = stats.row(); tail.alignment = 'RIGHT'
        theme.icon_label(tail, f'{count-rendered} pending', custom='camera_pending')
    else:
        row = box.row(align=True)
        row.enabled = not busy
        theme.operator(row, 'sceneray_splat.queue_add_selected', text='Add selected', custom='action_add')
        theme.operator(row, 'sceneray_splat.queue_add_all', text='Add scene', custom='stage_cameras')


def draw_prepare(layout, context):
    from .building_data import ui, stages
    from .auto_rig import coverage_ops
    cfg = context.scene.SCENERAY_SPLAT
    busy = stages.anything_running(context)
    draw_cameras(layout, context)
    layout.separator(factor=.4)
    draw_camera_methods(layout, context)
    layout.separator(factor=.4)
    coverage_ops.draw_card(layout, context)
    layout.separator(factor=.4)
    box = layout.box()
    header = theme.heading(box, '3  ·  Build dataset', custom='section_dataset')
    settings = header.row(align=True)
    settings.popover(panel='SPLATGEN_PT_dataset_settings', text='', **theme.icon_args('PREFERENCES'))
    path = box.row(align=True)
    path.enabled = not busy
    path.prop(cfg, 'output_dir', text='')
    theme.operator(path, 'sceneray_splat.open_output', custom='action_open_folder', icon='FILEBROWSER')
    ui.draw_build_dataset(box, context, cfg)


def draw_dataset_settings(layout, context):
    from .building_data import ui, stages
    from . import sceneray_splat as sr
    cfg = context.scene.SCENERAY_SPLAT
    busy = stages.anything_running(context)
    theme.heading(layout, 'Images', custom='stage_dataset')
    ui.draw_image_settings(theme.form(layout), cfg, busy)
    layout.separator(factor=.8)
    theme.heading(layout, 'Point sampling', custom='stage_points')
    body = theme.form(layout)
    body.enabled = not busy
    body.prop(cfg, 'point_quality', text='Quality')
    if cfg.point_quality == 'CUSTOM':
        for name in ('point_camera_usage', 'point_sampling_density', 'point_merging_strength'):
            body.prop(cfg, name)
    if not sr.point_settings_are_default(cfg):
        theme.operator(body, 'sceneray_splat.point_defaults', text='Reset sampling', icon='LOOP_BACK')
    ui._draw_point_estimate(body, cfg)
    layout.separator(factor=.8)
    from .raw_export import ui as raw_ui
    raw_ui.draw_settings(layout, context, busy)


def draw_camera_tools(layout, context):
    from . import sceneray_splat as sr
    cfg = context.scene.SCENERAY_SPLAT
    layout.enabled = not (cfg.is_rendering or cfg.is_generating_points)
    theme.heading(layout, 'Saved camera rigs', custom='stage_cameras')
    sr._sr_request_rig_library_refresh(cfg)
    theme.operator(layout, 'sceneray_splat.preset_save', text='Save current rig', icon='ADD')
    if not cfg.rig_presets:return
    layout.template_list('SCENERAY_SPLAT_UL_rig_presets','',cfg,'rig_presets',cfg,'active_preset_index',rows=5)
    preset = cfg.rig_presets[min(max(0,cfg.active_preset_index),len(cfg.rig_presets)-1)]
    theme.operator(layout,'sceneray_splat.preset_load',text='Add rig at 3D Cursor',icon='IMPORT')
    row=layout.row(align=True)
    row.enabled = not preset.builtin
    theme.operator(row,'sceneray_splat.preset_rename',text='Rename',icon='FILE_REFRESH')
    theme.operator(row,'sceneray_splat.preset_delete',text='Delete',icon='TRASH')


def draw(layout, context):
    layout.use_property_split = False
    layout.use_property_decorate = False
    draw_header(layout, context)
    draw_task(layout, context)
    draw_prepare(layout, context)

