# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
# ##### END GPL LICENSE BLOCK #####

"""Canonical dataset paths and build-folder rotation.

New builds keep the portable COLMAP dataset isolated from SplatGen-only data::

    <chosen folder>/SplatGen_<project>/<timestamp>/
        Dataset(Default)/
            images/frame_0000.png
            masks/frame_0000.png
            depth/frame_0000.exr
            cameras.txt
            images.txt
            points3D.txt
        sg_metadata/                     <- optional

The accessors also recognise the flat layout written through SplatGen 1.39 so
existing projects remain readable and resumable.
"""

import re
import shutil
import time
from pathlib import Path

import bpy

#: Prefix for the dedicated project folder created inside whatever folder the
#: user picks.  New projects include the .blend/project name so multiple
#: SplatGen projects can safely share the same parent folder.
ADDON_FOLDER = "SplatGen"

#: Folder names used by earlier releases.  A project built before the rename
#: keeps its original folder rather than silently starting an empty one.
LEGACY_ADDON_FOLDERS = ("SplatRay Studio", "SplatKit", "SceneRay Splat Studio")


def _safe_project_name(cfg=None):
    """Filesystem-safe .blend name, with a useful unsaved-scene fallback."""
    blend_path = str(getattr(bpy.data, "filepath", "") or "").strip()
    source = Path(blend_path).stem if blend_path else ""
    if not source and cfg is not None:
        source = str(getattr(getattr(cfg, "id_data", None), "name", "") or "")
    source = source or "Untitled"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip(" ._")[:80]
    return safe or "Untitled"


def project_folder_name(cfg=None):
    return f"{ADDON_FOLDER}_{_safe_project_name(cfg)}"


def addon_folder_names(cfg=None):
    """The project-aware folder first, then every name still recognised."""
    return (project_folder_name(cfg), ADDON_FOLDER) + LEGACY_ADDON_FOLDERS


def is_addon_folder_name(name, cfg=None):
    value = str(name or "")
    return value in addon_folder_names(cfg) or value.startswith(f"{ADDON_FOLDER}_")


def _addon_folder_in(parent, cfg=None):
    """``parent/SplatGen_<project>`` while preserving existing installations."""
    for name in addon_folder_names(cfg):
        if (parent / name).is_dir():
            return parent / name
    return parent / project_folder_name(cfg)

#: Suffix marking the single retained copy of a superseded unfinished build.
RECOVERY_SUFFIX = "_recovery"

#: Standard dataset folders. RGB is isolated from masks and metadata.
IMAGES_FOLDER = "images"
MASKS_FOLDER = "masks"
DEPTH_FOLDER = "depth"
DEFAULT_DATASET_FOLDER = "Dataset(Default)"
GROUND_TRUTH_FOLDER = "sg_metadata"
SCENE_GUIDANCE_FOLDER = "scene_guidance"
SCENE_GUIDANCE_MANIFEST = "manifest.json"
SCENE_GUIDANCE_SEED = "surface_seed.npz"

#: Written by stage 1 (cameras/images) and stage 2 (points).
STAGE1_FILES = ("cameras.txt", "images.txt")
POINTCLOUD_FILE = "points3D.txt"
DATASET_FILES = STAGE1_FILES + (POINTCLOUD_FILE,)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

def standard_dataset_dir(version_dir):
    """Portable dataset root for a build or an already-selected dataset.

    A freshly-created build contains ``Dataset(Default)``.  A legacy flat
    build has ``images`` directly below the timestamp and is deliberately
    returned unchanged.
    """
    path = Path(version_dir)
    if path.name == DEFAULT_DATASET_FOLDER:
        return path
    nested = path / DEFAULT_DATASET_FOLDER
    if nested.is_dir():
        return nested
    return path


def build_root(version_dir):
    """Timestamped build root belonging to either supported layout."""
    path = Path(version_dir)
    if path.name == DEFAULT_DATASET_FOLDER:
        return path.parent
    if path.parent.name == DEFAULT_DATASET_FOLDER:
        return path.parent.parent
    return path


def data_dir(version_dir):
    """The RGB-only image directory."""
    return standard_dataset_dir(version_dir) / IMAGES_FOLDER


def masks_dir(version_dir):
    """The standard binary-mask directory."""
    return standard_dataset_dir(version_dir) / MASKS_FOLDER


def depth_dir(version_dir):
    """Metric camera-Z maps used to reconstruct the RGB-D point cloud."""
    return standard_dataset_dir(version_dir) / DEPTH_FOLDER


def ground_truth_dir(version_dir):
    """Optional Blender numerical metadata, separate from standard data."""
    return build_root(version_dir) / GROUND_TRUTH_FOLDER


def scene_guidance_dir(version_dir):
    """Compact Blender surface seed used only by the SplatGen trainer."""
    return ground_truth_dir(version_dir) / SCENE_GUIDANCE_FOLDER


def scene_guidance_manifest(version_dir):
    return scene_guidance_dir(version_dir) / SCENE_GUIDANCE_MANIFEST


def scene_guidance_seed(version_dir):
    return scene_guidance_dir(version_dir) / SCENE_GUIDANCE_SEED


def dataset_file(version_dir, name):
    """A canonical COLMAP file at the dataset root."""
    return standard_dataset_dir(version_dir) / name


def has_stage1_output(version_dir):
    return all(
        dataset_file(version_dir, name).is_file() for name in STAGE1_FILES
    )


def has_pointcloud(version_dir):
    return dataset_file(version_dir, POINTCLOUD_FILE).is_file()


def is_complete_build(version_dir):
    """A build with images, stage 1 metadata and a point cloud."""
    version_dir = Path(version_dir)
    return (
        version_dir.is_dir()
        and data_dir(version_dir).is_dir()
        and has_stage1_output(version_dir)
        and has_pointcloud(version_dir)
    )


def has_rendered_images(version_dir):
    images = data_dir(Path(version_dir))
    if not images.is_dir():
        return False
    try:
        return any(
            child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            for child in images.iterdir()
        )
    except OSError:
        return False


def has_reusable_content(version_dir):
    """Work worth carrying into the next build.

    Deliberately weaker than :func:`is_complete_build`: a run interrupted
    part-way has images but no metadata yet, which is exactly the case
    continuing exists for.
    """
    version_dir = Path(version_dir)
    return version_dir.is_dir() and (
        has_rendered_images(version_dir) or has_stage1_output(version_dir)
    )


def is_recovery(path):
    return Path(path).name.endswith(RECOVERY_SUFFIX)


def is_timestamp_folder(path):
    path = Path(path)
    name = path.name
    if is_recovery(path):
        name = name[: -len(RECOVERY_SUFFIX)]
    if len(name) < 19:
        return False
    try:
        time.strptime(name[:19], "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return False
    return path.is_dir()


# --------------------------------------------------------------------------
# Project root
# --------------------------------------------------------------------------

def project_root(cfg):
    """The add-on's own folder inside the chosen output directory.

    Resolves through any build folder the user may have picked by mistake, so
    a build can never be nested inside another build.
    """
    raw = str(getattr(cfg, "output_dir", "") or "").strip()
    if not raw:
        if not bpy.data.filepath:
            return None
        raw = '//'
    if raw.startswith('//') and not bpy.data.filepath:
        return None
    chosen = Path(bpy.path.abspath(raw))
    if chosen.name == DEFAULT_DATASET_FOLDER:
        chosen = chosen.parent
    elif chosen.name.lower() in {IMAGES_FOLDER, MASKS_FOLDER}:
        if chosen.parent.name == DEFAULT_DATASET_FOLDER:
            chosen = chosen.parent.parent
        else:
            chosen = chosen.parent
    elif chosen.name == GROUND_TRUTH_FOLDER:
        chosen = chosen.parent

    # Picking a build folder, or the add-on folder itself, still resolves to
    # the one add-on root rather than nesting another level.
    if is_addon_folder_name(chosen.name, cfg):
        return chosen
    if is_timestamp_folder(chosen) or has_reusable_content(chosen):
        parent = chosen.parent
        if is_addon_folder_name(parent.name, cfg):
            return parent
        return _addon_folder_in(parent, cfg)
    return _addon_folder_in(chosen, cfg)


def latest_build(root, preferred=None):
    """Newest non-recovery build below ``root`` that still holds work."""
    if root is None:
        return None
    root = Path(root)
    candidates = []
    if preferred:
        candidate = Path(bpy.path.abspath(str(preferred)))
        if not is_recovery(candidate) and has_reusable_content(candidate):
            candidates.append(candidate)
    if root.is_dir():
        try:
            candidates.extend(
                child
                for child in root.iterdir()
                if is_timestamp_folder(child)
                and not is_recovery(child)
                and has_reusable_content(child)
            )
        except OSError:
            pass
    if not candidates:
        return None
    unique = {str(path.resolve()): path for path in candidates}
    return max(
        unique.values(),
        key=lambda path: (path.name[:19], path.stat().st_mtime_ns),
    )


def new_build_folder(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    target = root / stamp
    suffix = 2
    while target.exists():
        target = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    standard = target / DEFAULT_DATASET_FOLDER
    (standard / IMAGES_FOLDER).mkdir(parents=True)
    (standard / MASKS_FOLDER).mkdir(parents=True)
    return target


# --------------------------------------------------------------------------
# Recovery rotation
# --------------------------------------------------------------------------

def existing_recovery_folders(root):
    if root is None or not Path(root).is_dir():
        return []
    try:
        found = [child for child in Path(root).iterdir() if is_recovery(child)]
    except OSError:
        return []
    found.sort(key=lambda path: path.name)
    return found


def rotate_recovery(root, superseded):
    """Keep exactly one recovery copy: the build just superseded.

    Continuing an unfinished build copies its work forward, so the original is
    no longer needed as-is - but throwing it away outright would leave nothing
    to fall back on. It becomes *the* recovery folder, and any older recovery
    folder is removed so these can never pile up.

    Returns ``(recovery_path, removed_names)``.
    """
    root = Path(root)
    removed = []
    recovery = None

    if superseded is not None and Path(superseded).is_dir():
        superseded = Path(superseded)
        if is_recovery(superseded):
            recovery = superseded
        else:
            target = superseded.with_name(f"{superseded.name}{RECOVERY_SUFFIX}")
            counter = 2
            while target.exists():
                target = superseded.with_name(
                    f"{superseded.name}_{counter:02d}{RECOVERY_SUFFIX}"
                )
                counter += 1
            try:
                superseded.rename(target)
                recovery = target
            except OSError:
                # A locked folder is not worth failing the build over; it is
                # simply left in place and still counted below.
                recovery = superseded

    for candidate in existing_recovery_folders(root):
        if recovery is not None and candidate == recovery:
            continue
        try:
            shutil.rmtree(candidate)
            removed.append(candidate.name)
        except OSError:
            pass
    return recovery, removed
