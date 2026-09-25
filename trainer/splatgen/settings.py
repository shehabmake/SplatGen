"""Per-user app settings in ~/.splatgen/settings.json."""

import json
import os
from pathlib import Path

HOME = Path(os.environ.get("SPLATGEN_HOME", Path.home() / ".splatgen"))
SETTINGS_FILE = HOME / "settings.json"

DEFAULTS = {
    "runs_dir": str(Path.home() / "SplatGen" / "runs"),
    "recent_datasets": [],
    "device": "auto",
    "backend": "auto",
}


def load():
    data = dict(DEFAULTS)
    try:
        data.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return data


def save(data):
    HOME.mkdir(parents=True, exist_ok=True)
    merged = dict(DEFAULTS)
    merged.update(data)
    SETTINGS_FILE.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def remember_dataset(path, limit=12):
    data = load()
    path = str(path)
    recent = [p for p in data.get("recent_datasets", []) if p != path]
    data["recent_datasets"] = [path] + recent[: limit - 1]
    return save(data)
