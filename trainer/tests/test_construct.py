"""The direct build from raw data, on a synthetic textured plane."""

import time

import numpy as np
import pytest
import torch

from splatgen.construct import ConstructConfig, build_splats
from splatgen.data import raw as rawdata
from splatgen.io import read_ply


def test_raw_dataset_reads(raw_build):
    ds = rawdata.RawDataset(raw_build)
    assert len(ds.views) == 12 and ds.has("depth") and not ds.has("true_normal")
    view = ds.views[1]
    assert ds.load_pass("position", view).shape == (72, 96, 3)
    assert ds.load_pass("depth", view).shape == (72, 96)
    assert rawdata.find_root(raw_build / "Dataset(Default)") == ds.root


def test_build_is_close_to_the_images(raw_build, tmp_path):
    config = ConstructConfig(rounds=2, test_every=4, background_resolution=16)
    builder, report = build_splats(raw_build, config, tmp_path, formats=("ply", "splat"))
    ev = report["evaluation"]
    assert ev["views"] == 3
    assert ev["psnr"] > 24, report
    # splats lie on the plane and are flat along its normal
    state = builder.state_dict()
    surface = slice(0, builder.n_surface)
    assert state["means"][surface, 2].abs().max() < 1e-3
    scales = state["scales"][surface].exp()
    assert (scales[:, 2] < 0.2 * scales[:, 1]).all()
    # the stripes need small, edge-aligned splats; the plain border does not
    assert report["rounds"][-1]["edge_splats"] > 0
    assert scales[:, 0].max() > 4 * scales[:, 0].min()
    params = read_ply(tmp_path / "point_cloud.ply")
    assert params["means"].shape[0] == report["splats"]
    assert (tmp_path / "model.splat").stat().st_size == 32 * report["splats"]


def test_build_can_be_stopped(raw_build):
    import threading
    from splatgen.construct import Builder, Stopped
    event = threading.Event()
    event.set()
    with pytest.raises(Stopped):
        Builder(raw_build, ConstructConfig(), stop_event=event).run()
