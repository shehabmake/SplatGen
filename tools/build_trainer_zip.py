"""Package the trainer app (plus the Blender add-on zip) into one download.

    python tools/build_trainer_zip.py   -> dist/SplatGen_Trainer_<version>.zip
"""

import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAINER = REPO / "trainer"
SKIP_DIRS = {"__pycache__", ".venv", "tests", ".pytest_cache"}


def main():
    version = re.search(r'__version__ = "([^"]+)"',
                        (TRAINER / "splatgen" / "__init__.py").read_text()).group(1)
    subprocess.run([sys.executable, str(REPO / "tools" / "build_addon_zip.py")], check=True)
    addon = sorted((REPO / "dist").glob("SplatGen_Prepare_*.zip"))[-1]
    target = REPO / "dist" / f"SplatGen_Trainer_{version}.zip"
    top = "SplatGen"
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(TRAINER.rglob("*")):
            relative = path.relative_to(TRAINER)
            if path.is_dir() or SKIP_DIRS & set(relative.parts) or path.suffix == ".pyc" \
                    or relative.parts[0].endswith(".egg-info"):
                continue
            info = zipfile.ZipInfo.from_file(path, f"{top}/{relative.as_posix()}")
            if path.suffix == ".sh":
                info.external_attr = 0o755 << 16
            data = path.read_bytes()
            if relative.as_posix() == "README.md":
                # docs/ sits next to the README in the download, one level up in the repo
                data = data.replace(b"../docs/", b"docs/")
            archive.writestr(info, data, zipfile.ZIP_DEFLATED)
            count += 1
        archive.write(addon, f"{top}/Blender add-on/{addon.name}")
        docs = ("RAW_DATASET.md", "CONSTRUCT.md")
        for name in docs:
            archive.write(REPO / "docs" / name, f"{top}/docs/{name}")
    print(f"{target.relative_to(REPO)}: {count + 1 + len(docs)} files, {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
