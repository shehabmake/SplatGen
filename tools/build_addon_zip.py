"""Package blender_addon/ as an installable Blender extension zip.

    python tools/build_addon_zip.py            -> dist/SplatGen_Prepare_<version>.zip

Install it in Blender with Preferences > Add-ons > Install from Disk.
"""

import sys
import tomllib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADDON = REPO / "blender_addon"
EXCLUDE_DIRS = {"__pycache__", ".git"}


def main():
    manifest = tomllib.loads((ADDON / "blender_manifest.toml").read_text(encoding="utf-8"))
    target = REPO / "dist" / f"SplatGen_Prepare_{manifest['version']}.zip"
    target.parent.mkdir(exist_ok=True)
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ADDON.rglob("*")):
            relative = path.relative_to(ADDON)
            if path.is_dir() or EXCLUDE_DIRS & set(relative.parts) or path.suffix == ".pyc":
                continue
            archive.write(path, relative.as_posix())
            count += 1
    print(f"{target.relative_to(REPO)}: {count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
