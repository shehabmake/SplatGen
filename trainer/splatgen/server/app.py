"""The local API behind the web UI. Binds to 127.0.0.1 only."""

import io
import os
import shutil
import string
import threading
from pathlib import Path

import numpy as np
import torch
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .. import __version__, render, settings
from ..config import PRESETS, TrainConfig, preset_config
from ..construct import ConstructConfig
from ..data import legacy, load_scene
from ..io import FORMATS, export_model, read_ply, write_ply, write_splat
from ..model import GaussianModel
from .jobs import JobManager
from .runs import RunStore

WEB = Path(__file__).resolve().parent.parent / "web"


class _SceneCache:
    def __init__(self, size=4):
        self.size = size
        self.items = {}
        self.lock = threading.Lock()

    def get(self, path):
        key = str(Path(path).expanduser().resolve())
        with self.lock:
            if key in self.items:
                return self.items[key]
        scene = load_scene(key)
        with self.lock:
            if len(self.items) >= self.size:
                self.items.pop(next(iter(self.items)))
            self.items[key] = scene
        return scene


class _RunRenderer:
    """Renders finished runs from their checkpoint (one cached at a time)."""

    def __init__(self):
        self.key = None
        self.model = None
        self.lock = threading.Lock()

    def model_for(self, folder):
        checkpoint = folder / "checkpoint.pt"
        ply = folder / "splats.ply"
        source = checkpoint if checkpoint.is_file() else ply
        if not source.is_file():
            raise FileNotFoundError("This run has no checkpoint or PLY yet")
        key = (str(source), source.stat().st_mtime_ns)
        with self.lock:
            if self.key != key:
                device = render.pick_device(settings.load().get("device", "auto"))
                if source.suffix == ".pt":
                    state = torch.load(source, map_location="cpu", weights_only=False)["params"]
                else:
                    state = read_ply(source)
                self.model = GaussianModel.from_state_dict(state, device)
                self.key = key
            return self.model


def _default_run_name(scene):
    """``scene`` for SplatGen_scene/<timestamp>/, else the dataset folder."""
    build = scene.extras.get("build_folder")
    if build:
        project = Path(build).parent.name
        return project[len("SplatGen_"):] if project.startswith("SplatGen_") else project
    root = Path(scene.root)
    return root.parent.name if root.name.lower() in {"sparse", "0"} else root.name


METHODS = {
    "train": {"label": "Train", "description": "Standard 3D Gaussian Splatting from the images."},
    "construct": {"label": "Build from raw data",
                  "description": "No training: splats placed on the surface and coloured directly "
                                 "from the raw passes. Seconds to minutes."},
    "construct_train": {"label": "Build + polish",
                        "description": "Build from raw data, then a short training run to polish."},
}


def _raw_root(scene, path):
    from ..data import raw
    for candidate in (scene.extras.get("build_folder"), path):
        if candidate:
            found = raw.find_root(candidate)
            if found is not None:
                return found
    return None


def _png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png")


def _render_model(model, cam, max_size):
    device = model.device
    backend = render.pick_backend(settings.load().get("backend", "auto"), device)
    a = model.activated()
    scale = min(1.0, max_size / max(cam.width, cam.height))
    width, height = max(1, round(cam.width * scale)), max(1, round(cam.height * scale))
    with torch.no_grad():
        rgb, _alpha, _info = backend.rasterize(
            a["means"], a["quats"], a["scales"], a["opacities"], a["sh"], model.max_sh_degree,
            torch.tensor(cam.world_to_camera, dtype=torch.float32, device=device),
            torch.tensor(cam.K(scale), dtype=torch.float32, device=device),
            width, height, torch.zeros(3, device=device))
    return (rgb.clamp(0, 1) * 255).byte().cpu().numpy()


def create_app(runs_dir=None):
    app = FastAPI(title="SplatGen", version=__version__)
    state = {
        "store": RunStore(runs_dir or settings.load()["runs_dir"]),
        "scenes": _SceneCache(),
        "renderer": _RunRenderer(),
    }
    state["jobs"] = JobManager(state["store"])
    app.state.splatgen = state

    def store():
        return state["store"]

    def scene_for(path):
        try:
            return state["scenes"].get(path)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except Exception as exc:
            raise HTTPException(400, f"Could not read dataset: {exc}")

    def run_or_404(run_id):
        try:
            return store().get(run_id)
        except (KeyError, FileNotFoundError, OSError):
            raise HTTPException(404, "Run not found")

    # -- system & settings ---------------------------------------------------------

    @app.get("/api/system")
    def system():
        info = render.describe()
        info["version"] = __version__
        info["settings"] = settings.load()
        info["busy"] = state["jobs"].busy()
        return info

    @app.get("/api/settings")
    def get_settings():
        return settings.load()

    @app.put("/api/settings")
    def put_settings(data: dict = Body(...)):
        allowed = {k: v for k, v in data.items() if k in settings.DEFAULTS}
        saved = settings.save({**settings.load(), **allowed})
        state["store"] = RunStore(saved["runs_dir"])
        state["jobs"].store = state["store"]
        return saved

    # -- file system browsing -------------------------------------------------------

    @app.get("/api/fs/roots")
    def fs_roots():
        roots = [{"name": "Home", "path": str(Path.home())}]
        if os.name == "nt":
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    roots.append({"name": drive, "path": drive})
        else:
            roots.append({"name": "/", "path": "/"})
        return {"roots": roots, "recent": settings.load().get("recent_datasets", [])}

    @app.get("/api/fs/list")
    def fs_list(path: str = Query(...)):
        folder = Path(path).expanduser()
        if not folder.is_dir():
            raise HTTPException(404, "Folder not found")
        folder = folder.resolve()
        entries = []
        try:
            children = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            raise HTTPException(403, "Permission denied")
        for child in children:
            if child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            entry = {"name": child.name, "path": str(child), "dir": is_dir}
            if is_dir:
                try:
                    entry["dataset"] = legacy.can_load(child)
                except OSError:
                    entry["dataset"] = False
            elif child.suffix.lower() not in {".ply", ".splat"}:
                continue
            entries.append(entry)
        return {"path": str(folder), "parent": str(folder.parent),
                "dataset": legacy.can_load(folder), "entries": entries[:1000]}

    # -- datasets -------------------------------------------------------------------

    @app.get("/api/dataset")
    def dataset(path: str = Query(...)):
        scene = scene_for(path)
        settings.remember_dataset(path)
        summary = scene.summary()
        summary["path"] = path
        summary["cameras"] = [c.to_dict() for c in scene.cameras]
        return summary

    @app.get("/api/dataset/points")
    def dataset_points(path: str = Query(...), limit: int = 200_000):
        scene = scene_for(path)
        xyz, rgb = scene.points_xyz, scene.points_rgb
        if len(xyz) > limit:
            index = np.random.default_rng(0).choice(len(xyz), limit, replace=False)
            xyz, rgb = xyz[index], rgb[index]
        data = np.concatenate([xyz.astype(np.float32), rgb.astype(np.float32) / 255.0], axis=1)
        return Response(data.astype("<f4").tobytes(), media_type="application/octet-stream")

    @app.get("/api/dataset/image")
    def dataset_image(path: str = Query(...), index: int = 0, size: int = 320):
        scene = scene_for(path)
        if not 0 <= index < len(scene.cameras):
            raise HTTPException(404, "No such image")
        with Image.open(scene.cameras[index].image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((size, size))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
        return Response(buffer.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    # -- presets & runs -----------------------------------------------------------------

    @app.get("/api/presets")
    def presets():
        return {"presets": {name: {**p, "values": preset_config(name).to_dict()}
                            for name, p in PRESETS.items()},
                "defaults": TrainConfig().to_dict(), "formats": FORMATS,
                "construct": ConstructConfig().to_dict(), "methods": METHODS}

    @app.get("/api/runs")
    def runs():
        items = store().list()
        job = state["jobs"].job
        for run in items:
            if job is not None and job.run_id == run["id"] and job.active:
                run["live"] = job.live()
        return {"runs": items, "busy": state["jobs"].busy()}

    @app.post("/api/runs")
    def create_run(data: dict = Body(...)):
        path = data.get("dataset")
        if not path:
            raise HTTPException(400, "Choose a dataset first")
        scene = scene_for(path)
        if state["jobs"].busy():
            raise HTTPException(409, "A training run is already in progress")
        preset = data.get("preset", "standard")
        method = data.get("method", "train")
        if method not in METHODS:
            raise HTTPException(400, "Unknown method")
        config = preset_config(preset if preset in PRESETS else "standard", data.get("config") or {})
        name = data.get("name") or _default_run_name(scene)
        extra = {"preset": preset, "method": method}
        if method != "train":
            raw = _raw_root(scene, path)
            if raw is None:
                raise HTTPException(400, "This dataset has no Dataset(Raw) folder. Export it with the raw "
                                         "data option in the Blender add-on to build splats directly.")
            construct = ConstructConfig.from_dict(data.get("construct") or {})
            construct.test_every = config.test_every
            extra.update(raw_dataset=str(raw), construct=construct.to_dict())
        record = store().create(name, path, config.to_dict(), kind="training" if method == "train" else "construct")
        store().update(record["id"], **extra)
        state["jobs"].start(record["id"])
        return store().get(record["id"])

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        record = run_or_404(run_id)
        job = state["jobs"].get(run_id)
        if job is not None and job.active:
            record["live"] = job.live()
        record["file_sizes"] = store().files(run_id)
        return record

    @app.get("/api/runs/{run_id}/metrics")
    def run_metrics(run_id: str):
        run_or_404(run_id)
        return {"metrics": store().metrics(run_id)}

    def _job(run_id):
        job = state["jobs"].get(run_id)
        if job is None or not job.active:
            raise HTTPException(409, "This run is not training")
        return job

    @app.post("/api/runs/{run_id}/pause")
    def pause(run_id: str):
        _job(run_id).pause()
        return {"ok": True}

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str, data: dict = Body(default={})):
        job = state["jobs"].get(run_id)
        if job is not None and job.active:
            job.resume_training()
            return {"ok": True}
        record = run_or_404(run_id)
        folder = store().path(run_id)
        if not (folder / "checkpoint.pt").is_file() and (folder / "constructed.ply").is_file():
            # A finished build: polish it with a short training run.
            config = dict(record["config"])
            config["steps"] = int(data.get("extra_steps") or 0) or 2000
            store().update(run_id, config=config, method="construct_train", status="queued", error="")
            try:
                state["jobs"].start(run_id)
            except RuntimeError as exc:
                raise HTTPException(409, str(exc))
            return {"ok": True}
        if not (folder / "checkpoint.pt").is_file():
            raise HTTPException(409, "No checkpoint to continue from")
        config = dict(record["config"])
        extra = int(data.get("extra_steps") or 0)
        if extra:
            config["steps"] = max(int(record.get("step", 0)), int(config["steps"])) + extra
        store().update(run_id, config=config, status="queued", error="")
        try:
            state["jobs"].start(run_id, resume=True)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"ok": True}

    @app.post("/api/runs/{run_id}/stop")
    def stop(run_id: str):
        _job(run_id).stop()
        return {"ok": True}

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        run_or_404(run_id)
        job = state["jobs"].get(run_id)
        if job is not None and job.active:
            raise HTTPException(409, "Stop the run before deleting it")
        store().delete(run_id)
        return {"ok": True}

    @app.patch("/api/runs/{run_id}")
    def rename_run(run_id: str, data: dict = Body(...)):
        run_or_404(run_id)
        name = str(data.get("name", "")).strip()
        if not name:
            raise HTTPException(400, "Name cannot be empty")
        return store().update(run_id, name=name)

    @app.get("/api/runs/{run_id}/splat")
    def run_splat(run_id: str):
        run_or_404(run_id)
        job = state["jobs"].get(run_id)
        if job is not None and job.active:
            data = job.splat_bytes()
            if data is not None:
                return Response(data, media_type="application/octet-stream",
                                headers={"Cache-Control": "no-store"})
        path = store().path(run_id) / "splats.splat"
        if path.is_file():
            return FileResponse(path, media_type="application/octet-stream",
                                headers={"Cache-Control": "no-store"})
        raise HTTPException(404, "No splats yet")

    @app.get("/api/runs/{run_id}/preview")
    def run_preview(run_id: str, camera: int = 0, size: int = 720):
        record = run_or_404(run_id)
        scene = scene_for(record["dataset"])
        if not 0 <= camera < len(scene.cameras):
            raise HTTPException(404, "No such camera")
        cam = scene.cameras[camera]
        job = state["jobs"].get(run_id)
        if job is not None and job.active and job.trainer is not None and job.trainer.model is not None:
            return _png(job.trainer.preview(cam, size))
        building = job.preview_model() if job is not None and job.active else None
        if building is not None:
            return _png(_render_model(building, cam, size))
        try:
            model = state["renderer"].model_for(store().path(run_id))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        return _png(_render_model(model, cam, size))

    @app.post("/api/runs/{run_id}/export")
    def run_export(run_id: str, data: dict = Body(...)):
        run_or_404(run_id)
        fmt = data.get("format", "ply")
        if fmt not in FORMATS:
            raise HTTPException(400, "Unknown format")
        folder = store().path(run_id)
        job = state["jobs"].get(run_id)
        name = f"splats.{fmt}"
        if job is not None and job.active and job.trainer is not None and job.trainer.model is not None:
            job.trainer.export(fmt, folder / name)
        elif job is not None and job.active and job.preview_model() is not None:
            export_model(job.preview_model(), fmt, folder / name)
        else:
            model = state["renderer"].model_for(folder)
            export_model(model, fmt, folder / name)
        target = data.get("destination")
        if target:
            destination = Path(target).expanduser()
            extension = FORMATS[fmt]["extension"]
            if destination.is_dir() or destination.suffix.lower() != extension:
                destination = destination / f"{run_or_404(run_id)['name']}{extension}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / name, destination)
            return {"path": str(destination), "file": name}
        return {"path": str(folder / name), "file": name}

    @app.get("/api/runs/{run_id}/files/{name}")
    def run_file(run_id: str, name: str):
        record = run_or_404(run_id)
        path = (store().path(run_id) / name).resolve()
        if path.parent != store().path(run_id) or not path.is_file():
            raise HTTPException(404, "File not found")
        download = f"{record['name']}{path.suffix}" if path.suffix in {".ply", ".splat"} else name
        return FileResponse(path, filename=download)

    @app.post("/api/import")
    def import_splats(data: dict = Body(...)):
        """Import a .ply as a run so it can be viewed, exported or trained further."""
        source = Path(str(data.get("path", ""))).expanduser()
        if not source.is_file() or source.suffix.lower() != ".ply":
            raise HTTPException(400, "Choose a Gaussian splat .ply file")
        try:
            params = read_ply(source)
        except Exception as exc:
            raise HTTPException(400, f"Could not read PLY: {exc}")
        record = store().create(data.get("name") or source.stem, data.get("dataset", ""),
                                TrainConfig().to_dict(), kind="imported")
        folder = store().path(record["id"])
        write_ply(params, folder / "splats.ply")
        write_splat(params, folder / "splats.splat")
        return store().update(record["id"], status="imported",
                              results={"gaussians": int(params["means"].shape[0])},
                              files={"ply": "splats.ply", "splat": "splats.splat"})

    # -- the web UI ---------------------------------------------------------------------

    @app.exception_handler(RuntimeError)
    def runtime_error(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app
