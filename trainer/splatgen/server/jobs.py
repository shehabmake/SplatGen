"""Runs one job at a time in a background thread: training, a direct build
from the raw dataset, or a build followed by a short polish.

The GPU can only usefully train one scene at a time, so a second start is
refused rather than queued. Everything the UI shows while training comes
from ``live()``; everything after comes from the run folder.
"""

import json
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import torch

from ..config import TrainConfig
from ..construct import ConstructConfig, Stopped
from ..construct.builder import Builder, polish_config
from ..data import load_scene
from ..io import export_model, to_splat_bytes
from ..model import GaussianModel
from ..train.trainer import Trainer


class Job:
    def __init__(self, store, run_id, resume=False):
        self.store = store
        self.run_id = run_id
        self.resume = resume
        self.trainer = None
        self.builder = None
        self.thread = None
        self.status = "queued"
        self.latest = {}
        self.log = deque(maxlen=200)
        self.error = ""
        self._splat_cache = (None, b"")
        self._model_cache = (None, None)

    # -- control ------------------------------------------------------------

    def start(self):
        self.thread = threading.Thread(target=self._run, name=f"train-{self.run_id}", daemon=True)
        self.thread.start()

    def pause(self):
        if self.trainer:
            self.trainer.pause_event.set()
            self._set_status("paused")

    def resume_training(self):
        if self.trainer:
            self.trainer.pause_event.clear()
            self._set_status("training")

    def stop(self):
        if self.builder:
            self.builder.stop_event.set()
        if self.trainer:
            self.trainer.pause_event.clear()
            self.trainer.stop_event.set()
        self._set_status("stopping")

    @property
    def active(self):
        return self.thread is not None and self.thread.is_alive()

    # -- state for the UI --------------------------------------------------------

    def live(self):
        trainer = self.trainer
        state = {"run_id": self.run_id, "status": self.status, "error": self.error,
                 "latest": self.latest, "log": list(self.log)[-30:]}
        if trainer is not None and trainer.model is not None:
            state["steps"] = trainer.cfg.steps
            state["eval"] = trainer.eval_results
            state["device"] = str(getattr(trainer, "device", ""))
            state["backend"] = getattr(getattr(trainer, "backend", None), "NAME", "")
        elif self.builder is not None:
            state["building"] = True
        return state

    def current_params(self):
        """(version, params) of whatever is being made right now, or (None, None)."""
        trainer, builder = self.trainer, self.builder
        if trainer is not None and trainer.model is not None:
            with trainer.lock:
                return ("t", trainer.step), trainer.model.state_dict()
        if builder is not None and builder.params is not None:
            return ("b", len(builder.report["rounds"]), len(builder.params["means"])), builder.state_dict()
        return None, None

    def preview_model(self):
        """A renderable model of the build in progress (None while training:
        the trainer renders itself)."""
        if self.trainer is not None and self.trainer.model is not None:
            return None
        version, params = self.current_params()
        if params is None:
            return None
        if self._model_cache[0] != version:
            self._model_cache = (version, GaussianModel.from_state_dict(params, self.builder.device))
        return self._model_cache[1]

    def splat_bytes(self):
        """Current splats in .splat form, recomputed only when they changed."""
        version, params = self.current_params()
        if params is None:
            return None
        if self._splat_cache[0] == version:
            return self._splat_cache[1]
        data = to_splat_bytes(params)
        self._splat_cache = (version, data)
        return data

    # -- the thread ------------------------------------------------------------

    def _set_status(self, status, **extra):
        self.status = status
        try:
            self.store.update(self.run_id, status=status, **extra)
        except (OSError, KeyError):
            pass

    def _message(self, text):
        self.log.append({"time": time.time(), "text": str(text)})

    def _run(self):
        store, run_id = self.store, self.run_id
        folder = store.path(run_id)
        try:
            record = store.get(run_id)
            method = record.get("method", "train")
            config = TrainConfig.from_dict(record["config"])
            if method in ("construct", "construct_train") and not self.resume:
                if not (folder / "constructed.ply").is_file() and self._build(record, folder) is None:
                    return
                if method == "construct":
                    return
                config = polish_config(config.steps, folder / "constructed.ply",
                                       record["construct"].get("test_every", 8),
                                       sh_degree=record["construct"].get("sh_degree", 3))
                store.update(run_id, config=config.to_dict())
            self._train(record, config, folder)
        except Exception as exc:  # the UI shows the message; the log has the trace
            self.error = str(exc)
            self._message(traceback.format_exc())
            self._set_status("failed", error=str(exc))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build(self, record, folder):
        """Direct construction; returns the report, or None when stopped."""
        store, run_id = self.store, self.run_id
        config = ConstructConfig.from_dict(record.get("construct"))
        self._set_status("building")
        self._message(f"Building splats from the raw data of {record['dataset']}")

        def on_progress(entry):
            self.latest = entry
            store.append_metric(run_id, {"build": True, **entry})

        self.builder = Builder(record.get("raw_dataset") or record["dataset"], config, folder, on_progress=on_progress,
                               log=self._message, stop_event=threading.Event())
        try:
            report = self.builder.run()
        except Stopped:
            self._message("Build stopped")
            self._set_status("stopped")
            return None
        from ..io import write_ply, write_splat
        state = self.builder.state_dict()
        write_ply(state, folder / "constructed.ply")
        (folder / "construct.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        files = {}
        for fmt in ("ply", "splat"):
            export_model(state, fmt, folder / f"splats.{fmt}")
            files[fmt] = f"splats.{fmt}"
        final = {"gaussians": report["splats"], "eval": report.get("evaluation", {}),
                 "construct": {k: report[k] for k in ("rounds", "samples", "time") if k in report}}
        self._set_status("finished" if record.get("method") == "construct" else "built",
                         results=final, files=files)
        return report

    def _train(self, record, config, folder):
        store, run_id = self.store, self.run_id
        self._set_status("loading")
        self._message(f"Loading dataset {record['dataset']}")
        scene = load_scene(record["dataset"])
        for warning in scene.warnings:
            self._message(f"Warning: {warning}")

        def on_progress(entry):
            self.latest = entry
            store.append_metric(run_id, entry)

        self.trainer = Trainer(scene, config, folder, on_progress=on_progress, log=self._message)
        checkpoint = folder / "checkpoint.pt"
        self._message("Preparing images and Gaussians")
        self.trainer.setup(checkpoint if self.resume and checkpoint.is_file() else None)
        self.builder = None
        self._set_status("training", device=str(self.trainer.device),
                         backend=self.trainer.backend.NAME)
        outcome = self.trainer.run()
        self._set_status("saving")
        self._message("Evaluating held-out views")
        results = self.trainer.evaluate() if self.trainer.test_cams else {}
        self.trainer.save_checkpoint(checkpoint)
        files = {}
        for fmt in config.export_formats:
            path = self.trainer.export(fmt, folder / f"splats.{fmt}")
            files[fmt] = Path(path).name
        final = dict(store.get(run_id).get("results") or {})
        final.update({"gaussians": len(self.trainer.model), "step": self.trainer.step,
                      "eval": results, "train": self.latest})
        status = "finished" if outcome == "finished" else "stopped"
        self._message(f"Run {status} at step {self.trainer.step:,}")
        self._set_status(status, step=self.trainer.step, results=final, files=files)


class JobManager:
    def __init__(self, store):
        self.store = store
        self.job = None
        self.lock = threading.Lock()

    def busy(self):
        return self.job is not None and self.job.active

    def start(self, run_id, resume=False):
        with self.lock:
            if self.busy():
                raise RuntimeError("A training run is already in progress")
            self.job = Job(self.store, run_id, resume=resume)
            self.job.start()
            return self.job

    def get(self, run_id):
        job = self.job
        return job if job is not None and job.run_id == run_id else None
