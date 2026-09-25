"""Run folders: one per training, holding config, metrics, checkpoint and exports.

    <runs_dir>/<run id>/
        run.json          id, name, dataset, config, status, results
        metrics.jsonl     one progress entry per line
        checkpoint.pt     latest checkpoint (for resume)
        splats.ply        exports
        splats.splat
"""

import json
import re
import shutil
import time
from pathlib import Path

from ..train.trainer import write_json


def slug(text):
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", str(text)).strip("-")
    return value[:40] or "run"


class RunStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, run_id):
        path = (self.root / run_id).resolve()
        if self.root.resolve() not in path.parents:
            raise KeyError(run_id)
        return path

    def create(self, name, dataset, config, kind="training"):
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{slug(name)}"
        suffix = 2
        while (self.root / run_id).exists():
            run_id = f"{stamp}-{slug(name)}-{suffix}"
            suffix += 1
        folder = self.root / run_id
        folder.mkdir(parents=True)
        record = {
            "id": run_id, "name": name, "kind": kind, "dataset": str(dataset) if dataset else "",
            "config": config, "status": "queued", "created": time.time(),
            "updated": time.time(), "step": 0, "results": {}, "files": {}, "error": "",
        }
        write_json(folder / "run.json", record)
        return record

    def get(self, run_id):
        return json.loads((self.path(run_id) / "run.json").read_text(encoding="utf-8"))

    def update(self, run_id, **changes):
        record = self.get(run_id)
        record.update(changes)
        record["updated"] = time.time()
        write_json(self.path(run_id) / "run.json", record)
        return record

    def list(self):
        if not self.root.is_dir():
            return []
        runs = []
        for folder in self.root.iterdir():
            meta = folder / "run.json"
            if meta.is_file():
                try:
                    runs.append(json.loads(meta.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
        runs.sort(key=lambda r: r.get("created", 0), reverse=True)
        return runs

    def append_metric(self, run_id, entry):
        with open(self.path(run_id) / "metrics.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def metrics(self, run_id, limit=5000):
        path = self.path(run_id) / "metrics.jsonl"
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    def delete(self, run_id):
        shutil.rmtree(self.path(run_id))

    def files(self, run_id):
        folder = self.path(run_id)
        return {p.name: p.stat().st_size for p in folder.iterdir()
                if p.is_file() and p.suffix in {".ply", ".splat", ".pt"}}
