# SplatGen trainer

A desktop app that trains **3D Gaussian Splatting** scenes from datasets made
by the SplatGen Blender add-on (or any COLMAP dataset), with a modern local
interface, a live training monitor and a built-in WebGL splat viewer.

This first version trains the standard 3DGS way from the **legacy dataset**
(`Dataset(Default)`: images + COLMAP `cameras.txt` / `images.txt` /
`points3D.txt`). Training from the raw data (`Dataset(Raw)`) comes next; the
code is laid out for it (see *Extending*).

## Install

**Windows (NVIDIA GPU):** double-click `install_windows.bat`, then start the
app with `SplatGen.bat`. Needs Python 3.10–3.12 and a current NVIDIA driver.
The script installs PyTorch 2.4.1 for CUDA 12.4 and a matching prebuilt
**gsplat**. If your setup needs another CUDA version, edit the two index URLs
at the top of the script.

**Linux / macOS:** `./install.sh` (for an NVIDIA GPU on Linux:
`TORCH_INDEX=https://download.pytorch.org/whl/cu124 ./install.sh`), then
`./splatgen.sh`.

**By hand:** `pip install -e .` (plus `pip install gsplat` on CUDA machines),
then `splatgen`.

The app runs a small server on `127.0.0.1` and opens your browser; nothing
leaves your computer. `splatgen app --native` opens a native window instead
if `pywebview` is installed.

Without an NVIDIA GPU the app still works, using a pure-PyTorch renderer. That
is 50–100× slower and only practical for small scenes at reduced resolution.

## Using it

1. **Home → Open dataset.** Pick the timestamped SplatGen build folder (the one
   with `Dataset(Default)` inside), the `Dataset(Default)` folder itself, or any
   COLMAP dataset. Folders the app can read are marked **Dataset**.
2. **Dataset** shows the images, the sparse points and every camera in 3D.
   Click an image to look through its camera.
3. **Train** — pick a preset (**Preview** 2k steps at half resolution,
   **Standard** 7k, **High quality** 30k), adjust anything under the advanced
   groups, and start. The monitor shows progress, loss/PSNR charts, the splat
   count and a live render next to the ground truth. **Pause**, **Stop & save**,
   **Continue** and **Train more** are all available.
4. **Viewer** — orbit the result in real time, look through dataset cameras,
   open any `.ply` / `.splat` by drag-and-drop, and take screenshots.
5. **Runs** — every run with its exports: download the **PLY** (standard 3DGS
   format: SuperSplat, Unity/Unreal plugins, most tools) or **.splat**, save to
   a folder, import an existing PLY, rename or delete.

Runs live in `~/SplatGen/runs/<run>/` (change it in **Settings**): `run.json`,
`metrics.jsonl`, `checkpoint.pt`, `splats.ply`, `splats.splat`.

## Command line

```
splatgen                                  # the app
splatgen train <dataset> --preset standard [--steps N] [--out DIR] [--set key=value ...]
splatgen train <dataset> --resume out/checkpoint.pt --steps 30000
splatgen export out/checkpoint.pt --format splat --out scene.splat
splatgen info [<dataset>]                 # GPU, renderer, dataset summary
```

## How training works

Standard 3D Gaussian Splatting (Kerbl et al. 2023), matching the reference and
gsplat's `simple_trainer` defaults: Gaussians start at the dataset's sparse
points with sizes from their nearest neighbours; each step renders one training
camera and minimises `0.8·L1 + 0.2·(1 − SSIM)` with Adam; spherical-harmonic
degree rises every 1,000 steps; between `densify_start` and `densify_stop`
Gaussians with large screen-space gradients are cloned (small) or split
(large), transparent ones are pruned, and opacities are reset every 3,000
steps. Every 8th image is held out and reported as **Test PSNR/SSIM**.

Coordinates stay in the dataset's frame (Blender world space, Z up, for
SplatGen datasets), so the PLY lines up with the Blender scene.

## Extending

| To add | Where |
|---|---|
| A dataset format (e.g. the raw dataset) | `splatgen/data/` — a module with `can_load` / `load` returning a `Scene`, registered in `data/__init__.py` |
| A way to create the first splats | `Trainer._initial_params` (`splatgen/train/trainer.py`) + `TrainConfig.init` |
| Extra loss terms / supervision | `Trainer.train_step` + fields in `splatgen/config.py` |
| A renderer | `splatgen/render/` — same `rasterize(...)` contract, registered in `render/__init__.py` |
| An export format | `splatgen/io/` + `FORMATS` |
| UI | `splatgen/web/` — plain ES modules, no build step |

## Development

```
pip install -e ".[dev]"
pytest                      # builds a synthetic dataset and trains on it (CPU)
```

The tests cover COLMAP text/binary reading, the rasterizer, densification with
Adam state, training improvement on held-out views, checkpoint resume, PLY /
.splat round trips and the full API flow.

## Status

Tested on Linux (CPU, PyTorch renderer): unit and API tests, training on a
16-camera Blender dataset, and the complete UI flow in headless Chromium. The
gsplat path follows gsplat 1.5's documented API but has not been run on a GPU
yet, and the Windows scripts have not been run on Windows yet — please report
any error message.
