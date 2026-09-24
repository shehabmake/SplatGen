# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""One place where the add-on's quiet failures are written down.

Many steps catch an error so Blender keeps working - a panel that cannot
read a folder, a timer whose scene is gone. Those used to vanish. Now each
is recorded here: kept in memory for the diagnostic report and appended to
a small log file, so a support request comes with what actually happened.
"""

import collections
import os
import platform
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import bpy

LOG_PATH = os.path.join(tempfile.gettempdir(), "splatgen_errors.log")
LOG_LIMIT = 1_000_000          # bytes; the file restarts past this
_recent = collections.deque(maxlen=50)


def swallowed(where):
    """Record the exception being handled; the caller carries on."""
    kind, error, trace = sys.exc_info()
    if kind is None:
        return
    lines = traceback.format_exception(kind, error, trace, limit=4)
    entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {where}: {kind.__name__}: {error}"
    _recent.append(entry)
    try:
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > LOG_LIMIT:
            os.replace(LOG_PATH, LOG_PATH + '.1')
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(entry + "\n" + "".join(lines) + "\n")
    except OSError:
        pass                    # the log must never become the failure


def log_sources(context=None):
    """Known support files only: never crawl scenes, images or model weights."""
    context = context or bpy.context
    sources = [("errors/splatgen_errors.log", Path(LOG_PATH)),
               ("errors/splatgen_errors.previous.log", Path(LOG_PATH + '.1'))]
    return [(name, path) for name, path in sources if path.is_file()]


def _tail(path, limit=65536):
    """Bound clipboard size while preserving the most recent traceback."""
    try:
        with path.open('rb') as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - limit))
            text = handle.read(limit).decode('utf-8', errors='replace')
        return (f'[Last {limit:,} of {size:,} bytes; full file in support ZIP]\n'
                if size > limit else '') + text
    except OSError as exc:
        return f'[Could not read {path.name}: {exc}]'


def report_text(context=None):
    """Everything support needs, as plain text."""
    from . import version, edition
    context = context or bpy.context
    rows = [
        f"SplatGen {version.addon_stringversion} ({edition.label()})",
        f"Blender {bpy.app.version_string} ({bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else bpy.app.build_hash})",
        f"System {platform.system()} {platform.release()} · Python {platform.python_version()}",
    ]
    try:
        import gpu
        rows.append(f"GPU {gpu.platform.vendor_get()} · {gpu.platform.renderer_get()} · "
                    f"{gpu.platform.backend_type_get()}")
    except Exception:
        rows.append("GPU unknown")
    rows.append("Edition: camera placement and dataset preparation only")
    scene = getattr(context, "scene", None)
    if scene is not None:
        rows.append(f"File {bpy.data.filepath or '(unsaved)'} · scenes {len(bpy.data.scenes)} · "
                    f"objects {len(scene.objects)}")
        cfg = getattr(scene, "SCENERAY_SPLAT", None)
        if cfg is not None:
            rows.append(f"Camera queue {len(cfg.camera_queue)} · output {cfg.output_dir}")
            rows.append(f"Workspace {cfg.workspace_step} · rendering {cfg.is_rendering} · "
                        f"sampling {cfg.is_generating_points}")
    try:
        from . import progress
        state = progress.current()
        if state.get("operation"):
            rows.append(f"Last task: {state['operation']} · {state.get('finished_state')} · "
                        f"{state.get('finished_message') or state.get('message')}")
    except Exception:
        pass
    rows.append("")
    rows.append(f"Recent handled errors ({len(_recent)}), full log: {LOG_PATH}")
    rows.extend(_recent or ["none"])
    sources = log_sources(context)
    rows.append('\nLog files included: ' + str(len(sources)))
    for name, path in sources:
        rows.extend(['', f'--- {name} ---', _tail(path)])
    return "\n".join(rows)


def save_bundle(context, report):
    """Snapshot complete known logs locally; no upload or network calls."""
    folder = Path(tempfile.gettempdir()) / 'SplatGen_Support'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (time.strftime('SplatGen_%Y%m%d_%H%M%S_') + str(time.time_ns())[-9:] + '.zip')
    failures = []
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('report.txt', report)
        for name, source in log_sources(context):
            try:
                # Fix the snapshot length so a running worker cannot grow the
                # file faster than this reader and keep the operator blocked.
                with source.open('rb') as src, archive.open(name, 'w', force_zip64=True) as dst:
                    remaining = os.fstat(src.fileno()).st_size
                    while remaining > 0:
                        block = src.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        dst.write(block)
                        remaining -= len(block)
            except OSError as exc:
                failures.append(f'{name}: {exc}')
        if failures:
            archive.writestr('unavailable-files.txt', '\n'.join(failures))
    return path, failures


class SPLATGEN_OT_copy_diagnostics(bpy.types.Operator):
    """Copy a bug report and save complete available logs to a local support ZIP.
    Includes system details, file paths and current run settings; nothing is uploaded"""
    bl_idname = "splatgen.copy_diagnostics"
    bl_label = "Copy Bug Report & Save Logs"
    bl_options = {'REGISTER'}

    def execute(self, context):
        report = report_text(context)
        try:
            path, failures = save_bundle(context, report)
        except (OSError, zipfile.BadZipFile) as exc:
            context.window_manager.clipboard = report + f'\n\nSupport ZIP could not be saved: {exc}'
            self.report({'WARNING'}, 'Bug report copied; support ZIP could not be saved.')
            return {'FINISHED'}
        context.window_manager.clipboard = report + f'\n\nFull support logs: {path}'
        self.report({'WARNING'} if failures else {'INFO'},
                    f'Bug report copied. Logs: {path}' + (' (some files unavailable)' if failures else ''))
        return {'FINISHED'}


CLASSES = (SPLATGEN_OT_copy_diagnostics,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
