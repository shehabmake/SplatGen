"""Runs one training job at a time in a background thread.

The GPU can only usefully train one scene at a time, so a second start is
refused rather than queued. Everything the UI shows while training comes
from ``live()``; everything after comes from the run folder.
"""

import threading
import time
import traceback
from collections import deque
from pathlib import Path

import torch

from ..config import TrainConfig
from ..data import load_scene
from ..io import export_model, to_splat_bytes
from ..train.trainer import Trainer


class Job:
    def __init__(self, store, run_id, resume=False):
        self.store = store
        self.run_id = run_id
        self.resume = resume
        self.trainer = None
        self.thread = None
        self.status = "queued"
        self.latest = {}
        self.log = deque(maxlen=200)
        self.error = ""
        self._splat_cache = (None, b"")

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
        return state

    def splat_bytes(self):
        """Current splats in .splat form, recomputed at most once per step."""
        trainer = self.trainer
        if trainer is None or trainer.model is None:
            return None
        if self._splat_cache[0] == trainer.step:
            return self._splat_cache[1]
        with trainer.lock:
            data = to_splat_bytes(trainer.model.state_dict())
            step = trainer.step
        self._splat_cache = (step, data)
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
            config = TrainConfig.from_dict(record["config"])
            self._set_status("loading")
            self._message(f"Loading dataset {record['dataset']}")
            scene = load_scene(record["dataset"])
            for warning in scene.warnings:
                self._message(f"Warning: {warning}")

            def on_progress(entry):
                self.latest = entry
                store.append_metric(run_id, entry)

            self.trainer = Trainer(scene, config, folder, on_progress=on_progress,
                                   log=self._message)
            checkpoint = folder / "checkpoint.pt"
            self._message("Preparing images and Gaussians")
            self.trainer.setup(checkpoint if self.resume and checkpoint.is_file() else None)
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
            final = {"gaussians": len(self.trainer.model), "step": self.trainer.step,
                     "eval": results, "train": self.latest}
            status = "finished" if outcome == "finished" else "stopped"
            self._message(f"Run {status} at step {self.trainer.step:,}")
            self._set_status(status, step=self.trainer.step, results=final, files=files)
        except Exception as exc:  # the UI shows the message; the log has the trace
            self.error = str(exc)
            self._message(traceback.format_exc())
            self._set_status("failed", error=str(exc))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
