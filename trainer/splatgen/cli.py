"""Command line: ``splatgen [app|train|build|export|info] ...``."""

import argparse
import json
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__


def _free_port(preferred):
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found")


def cmd_app(args):
    import uvicorn

    from .server.app import create_app

    port = _free_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    app = create_app(args.runs)
    print(f"SplatGen {__version__} running at {url}  (Ctrl+C to quit)")
    if args.native:
        try:
            import webview
        except ImportError:
            print("pywebview is not installed; opening the browser instead.")
            args.native = False
    if args.native:
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        time.sleep(1.0)
        webview.create_window("SplatGen", url, width=1440, height=900)
        webview.start()
        return 0
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


def _parse_overrides(pairs):
    values = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            values[key] = json.loads(raw)
        except ValueError:
            values[key] = raw
    return values


def cmd_train(args):
    from .config import preset_config
    from .data import load_scene
    from .io import export_model
    from .train.trainer import Trainer

    overrides = _parse_overrides(args.set)
    if args.steps:
        overrides["steps"] = args.steps
    config = preset_config(args.preset, overrides)
    scene = load_scene(args.dataset)
    out = Path(args.out or f"splatgen_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    last = [0.0]

    def progress(entry):
        if time.time() - last[0] > 5 or entry["step"] == config.steps:
            last[0] = time.time()
            print(f"step {entry['step']:>6}/{config.steps}  loss {entry['loss']:.4f}  "
                  f"psnr {entry['psnr']:.2f}  splats {entry['gaussians']:,}  eta {entry['eta']:.0f}s")

    trainer = Trainer(scene, config, out, on_progress=progress)
    resume = Path(args.resume) if args.resume else None
    trainer.setup(resume)
    try:
        trainer.run()
    except KeyboardInterrupt:
        print("Interrupted; saving what was trained so far.")
    results = trainer.evaluate()
    if results:
        print(f"test views: PSNR {results['psnr']:.2f}  SSIM {results['ssim']:.4f}")
    trainer.save_checkpoint(out / "checkpoint.pt")
    for fmt in config.export_formats:
        print("wrote", export_model(trainer.model, fmt, out / f"splats.{fmt}"))
    return 0


def cmd_build(args):
    from .construct import ConstructConfig, build_splats
    from .construct.builder import polish_config
    from .data import load_scene
    from .train.trainer import Trainer

    config = ConstructConfig.from_dict(_parse_overrides(args.set))
    out = Path(args.out or f"splatgen_build_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "construct_config.json").write_text(json.dumps(config.to_dict(), indent=2))
    builder, report = build_splats(args.dataset, config, out, formats=("ply", "splat"))
    for path in report["files"].values():
        print("wrote", path)
    if not args.polish:
        return 0
    train_config = polish_config(args.polish, report["files"]["ply"], config.test_every,
                                 _parse_overrides(args.polish_set))
    print(f"polishing for {train_config.steps:,} steps")
    trainer = Trainer(load_scene(args.dataset), train_config, out)
    trainer.setup()
    try:
        trainer.run()
    except KeyboardInterrupt:
        print("Interrupted; saving what was polished so far.")
    results = trainer.evaluate()
    if results:
        print(f"test views after polish: PSNR {results['psnr']:.2f}  SSIM {results['ssim']:.4f}")
    trainer.save_checkpoint(out / "checkpoint.pt")
    for fmt in train_config.export_formats:
        print("wrote", trainer.export(fmt, out / f"polished.{fmt}"))
    return 0


def cmd_export(args):
    import torch

    from .io import export_model, read_ply

    source = Path(args.source)
    if source.suffix == ".pt":
        params = torch.load(source, map_location="cpu", weights_only=False)["params"]
    else:
        params = read_ply(source)
    print("wrote", export_model(params, args.format, args.out))
    return 0


def cmd_info(args):
    from . import render
    from .data import load_scene

    info = {"system": render.describe()}
    if args.dataset:
        info["dataset"] = load_scene(args.dataset).summary()
    print(json.dumps(info, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="splatgen", description="SplatGen Gaussian splat trainer")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    app = sub.add_parser("app", help="open the SplatGen app (default)")
    app.add_argument("--port", type=int, default=7870)
    app.add_argument("--runs", help="folder for training runs")
    app.add_argument("--no-browser", action="store_true")
    app.add_argument("--native", action="store_true", help="native window (needs pywebview)")
    app.set_defaults(func=cmd_app)

    train = sub.add_parser("train", help="train without the UI")
    train.add_argument("dataset")
    train.add_argument("--preset", default="standard", choices=["preview", "standard", "high"])
    train.add_argument("--steps", type=int)
    train.add_argument("--out")
    train.add_argument("--resume", help="checkpoint.pt to continue from")
    train.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config fields")
    train.set_defaults(func=cmd_train)

    build = sub.add_parser("build", help="build splats directly from the raw dataset (no training)")
    build.add_argument("dataset", help="the build folder, Dataset(Raw) or Dataset(Default)")
    build.add_argument("--out")
    build.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override construct settings")
    build.add_argument("--polish", type=int, default=0, metavar="STEPS",
                       help="then train this many steps starting from the built splats")
    build.add_argument("--polish-set", nargs="*", metavar="KEY=VALUE", help="override polish training settings")
    build.set_defaults(func=cmd_build)

    export = sub.add_parser("export", help="convert a checkpoint or PLY")
    export.add_argument("source")
    export.add_argument("--format", default="ply", choices=["ply", "splat"])
    export.add_argument("--out", required=True)
    export.set_defaults(func=cmd_export)

    info = sub.add_parser("info", help="show hardware and dataset information")
    info.add_argument("dataset", nargs="?")
    info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["app"] + (argv or sys.argv[1:]))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
