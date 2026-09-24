# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""What a rendered image depended on, so a change can invalidate it.

Each manifest entry stores a *signature*: every camera property that changes
the pixels the renderer produces. Comparing the stored signature with the
camera's current one is what marks a camera outdated - no manual "mark
pending" step, and no re-rendering cameras that did not actually move.

Only render-affecting properties belong here. Object colour, viewport display
size, custom properties and the like are deliberately excluded: including them
would force pointless re-renders.
"""

#: Bumped when the set of tracked properties changes, so entries written by an
#: older build are re-rendered once rather than compared against a short list.
SIGNATURE_VERSION = 1

_PRECISION = 6


def _round(value):
    return round(float(value), _PRECISION)


def _matrix(matrix):
    return [_round(value) for row in matrix for value in row]


def _dof_signature(data):
    """Depth of field, only when it is actually enabled."""
    dof = getattr(data, "dof", None)
    if dof is None or not getattr(dof, "use_dof", False):
        return {"use_dof": False}
    focus = getattr(dof, "focus_object", None)
    signature = {
        "use_dof": True,
        "focus_object": getattr(focus, "name", ""),
        "focus_distance": _round(getattr(dof, "focus_distance", 0.0)),
        "aperture_fstop": _round(getattr(dof, "aperture_fstop", 0.0)),
        "aperture_blades": int(getattr(dof, "aperture_blades", 0)),
        "aperture_rotation": _round(getattr(dof, "aperture_rotation", 0.0)),
        "aperture_ratio": _round(getattr(dof, "aperture_ratio", 1.0)),
    }
    if focus is not None:
        # Moving the focus target changes the blur just as moving the camera
        # would, so the target's placement is part of the signature too.
        signature["focus_object_matrix"] = _matrix(focus.matrix_world)
    return signature


def _panorama_signature(data):
    """Cycles panoramic settings, which only exist on a panoramic camera."""
    if getattr(data, "type", "PERSP") != "PANO":
        return {}
    cycles = getattr(data, "cycles", None)
    source = cycles if cycles is not None else data
    signature = {}
    for name in (
        "panorama_type",
        "fisheye_fov",
        "fisheye_lens",
        "fisheye_polynomial_k0",
        "fisheye_polynomial_k1",
        "fisheye_polynomial_k2",
        "fisheye_polynomial_k3",
        "fisheye_polynomial_k4",
        "latitude_min",
        "latitude_max",
        "longitude_min",
        "longitude_max",
    ):
        value = getattr(source, name, None)
        if value is None:
            continue
        signature[name] = (
            value if isinstance(value, str) else _round(value)
        )
    return signature


def effective_clipping(camera_data, cfg):
    """The clipping this camera will actually render with.

    A global override changes the rendered image without touching the values
    stored on the camera, so the signature has to describe what the renderer
    will use - otherwise turning the override on would not re-render anything.
    """
    if cfg is not None and getattr(cfg, "use_global_clipping", False):
        return (
            float(getattr(cfg, "global_clip_start", 0.1)),
            float(getattr(cfg, "global_clip_end", 1000.0)),
        )
    return (
        float(getattr(camera_data, "clip_start", 0.1)),
        float(getattr(camera_data, "clip_end", 100.0)),
    )


def camera_signature(camera_object, cfg=None):
    """Everything about this camera that changes its rendered image."""
    from .. import sceneray_splat

    data = camera_object.data
    # Scale is stripped exactly as the renderer does, so a scaled camera does
    # not read as changed when only its (ignored) scale moved.
    pose, _had_scale = sceneray_splat.camera_pose_world(camera_object)

    signature = {
        "signature_version": SIGNATURE_VERSION,
        "matrix": _matrix(pose),
        "type": str(getattr(data, "type", "PERSP")),
        "lens_unit": str(getattr(data, "lens_unit", "MILLIMETERS")),
        "lens": _round(getattr(data, "lens", 50.0)),
        "angle": _round(getattr(data, "angle", 0.0)),
        "ortho_scale": _round(getattr(data, "ortho_scale", 6.0)),
        "sensor_fit": str(getattr(data, "sensor_fit", "AUTO")),
        "sensor_width": _round(getattr(data, "sensor_width", 36.0)),
        "sensor_height": _round(getattr(data, "sensor_height", 24.0)),
        "shift_x": _round(getattr(data, "shift_x", 0.0)),
        "shift_y": _round(getattr(data, "shift_y", 0.0)),
    }
    clip_start, clip_end = effective_clipping(data, cfg)
    signature["clip_start"] = _round(clip_start)
    signature["clip_end"] = _round(clip_end)
    signature.update(_dof_signature(data))
    panorama = _panorama_signature(data)
    if panorama:
        signature["panorama"] = panorama
    return signature


def signature_of(entry):
    stored = entry.get("signature") if isinstance(entry, dict) else None
    return stored if isinstance(stored, dict) else None


def matches(entry, signature):
    """True when a stored entry was rendered with these exact settings.

    An entry with no signature comes from a build made before signatures
    existed. It is treated as outdated once, which is the safe direction: the
    alternative is trusting an image whose camera may have moved.
    """
    stored = signature_of(entry)
    if stored is None:
        return False
    if int(stored.get("signature_version", 0)) != SIGNATURE_VERSION:
        return False
    return stored == signature


def changed_properties(entry, signature):
    """Human-readable names of the properties that differ, for the UI."""
    stored = signature_of(entry)
    if stored is None:
        return ["not previously tracked"]
    if int(stored.get("signature_version", 0)) != SIGNATURE_VERSION:
        return ["tracking updated"]
    labels = {
        "matrix": "position/rotation",
        "type": "projection type",
        "lens": "focal length",
        "lens_unit": "lens unit",
        "angle": "field of view",
        "ortho_scale": "orthographic scale",
        "sensor_fit": "sensor fit",
        "sensor_width": "sensor width",
        "sensor_height": "sensor height",
        "shift_x": "lens shift X",
        "shift_y": "lens shift Y",
        "clip_start": "clip start",
        "clip_end": "clip end",
        "use_dof": "depth of field",
        "focus_object": "focus object",
        "focus_object_matrix": "focus object position",
        "focus_distance": "focus distance",
        "aperture_fstop": "aperture f-stop",
        "aperture_blades": "aperture blades",
        "aperture_rotation": "aperture rotation",
        "aperture_ratio": "aperture ratio",
        "panorama": "panorama settings",
    }
    differing = []
    for key in sorted(set(stored) | set(signature)):
        if key == "signature_version":
            continue
        if stored.get(key) != signature.get(key):
            differing.append(labels.get(key, key.replace("_", " ")))
    return differing


def apply_clipping_override(cfg, cameras):
    """Set the global clipping for a render, returning what to restore.

    Nothing is written back to the camera afterwards beyond its own original
    values, so the per-camera settings survive the render untouched.
    """
    if cfg is None or not getattr(cfg, "use_global_clipping", False):
        return []
    start = float(getattr(cfg, "global_clip_start", 0.1))
    end = float(getattr(cfg, "global_clip_end", 1000.0))
    restore = []
    for camera in cameras:
        if camera is None or camera.type != "CAMERA":
            continue
        data = camera.data
        restore.append((data, data.clip_start, data.clip_end))
        data.clip_start = start
        data.clip_end = end
    return restore


def restore_clipping(restore):
    """Put every camera's own clipping back after a render."""
    for data, start, end in restore:
        try:
            data.clip_start = start
            data.clip_end = end
        except (AttributeError, ReferenceError, TypeError):
            pass


def outdated_cameras(cfg, entries):
    """Queued cameras whose stored signature no longer matches the scene.

    ``entries`` maps camera name to its manifest entry. Returns a mapping of
    camera name to the list of properties that changed.
    """
    outdated = {}
    for item in getattr(cfg, "camera_queue", ()):
        camera = item.camera
        if camera is None or camera.type != "CAMERA":
            continue
        entry = entries.get(camera.name)
        if entry is None:
            continue
        if str(entry.get("status", "")).upper() != "RENDERED":
            continue
        signature = camera_signature(camera, cfg)
        if not matches(entry, signature):
            outdated[camera.name] = changed_properties(entry, signature)
    return outdated
