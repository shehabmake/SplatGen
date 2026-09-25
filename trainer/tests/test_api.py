import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATGEN_HOME", str(tmp_path / "home"))
    import importlib
    import splatgen.settings as settings
    importlib.reload(settings)
    import splatgen.server.app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.create_app(tmp_path / "runs"))


def wait(client, run_id, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in {"finished", "failed", "stopped"}:
            return run
        time.sleep(0.2)
    raise TimeoutError(run["status"])


def test_full_flow(client, dataset, tmp_path):
    assert client.get("/").status_code == 200
    info = client.get("/api/dataset", params={"path": str(dataset)}).json()
    assert info["camera_count"] == 8 and len(info["cameras"]) == 8
    assert client.get("/api/dataset/image", params={"path": str(dataset), "index": 1}).status_code == 200
    assert len(client.get("/api/dataset/points", params={"path": str(dataset)}).content) == 300 * 24
    listing = client.get("/api/fs/list", params={"path": str(dataset.parent)}).json()
    assert any(entry["dataset"] for entry in listing["entries"])

    run = client.post("/api/runs", json={"dataset": str(dataset), "preset": "preview",
                                         "config": {"steps": 25, "downscale": 1}}).json()
    assert client.post("/api/runs", json={"dataset": str(dataset)}).status_code == 409
    done = wait(client, run["id"])
    assert done["status"] == "finished", done["error"]
    assert "live" not in done
    assert done["files"] == {"ply": "splats.ply", "splat": "splats.splat"}
    assert done["results"]["eval"]["views"] == 1
    assert client.get(f"/api/runs/{run['id']}/preview", params={"camera": 2}).headers["content-type"] == "image/png"
    assert len(client.get(f"/api/runs/{run['id']}/splat").content) % 32 == 0
    assert len(client.get(f"/api/runs/{run['id']}/metrics").json()["metrics"]) >= 1

    out = client.post(f"/api/runs/{run['id']}/export", json={"format": "ply", "destination": str(tmp_path / "out")}).json()
    assert out["path"].endswith(".ply")
    download = client.get(f"/api/runs/{run['id']}/files/splats.ply")
    assert download.status_code == 200 and "attachment" in download.headers["content-disposition"]
    assert client.get(f"/api/runs/{run['id']}/files/../run.json").status_code == 404

    assert client.post(f"/api/runs/{run['id']}/resume", json={"extra_steps": 5}).json()["ok"]
    assert wait(client, run["id"])["step"] == 30

    imported = client.post("/api/import", json={"path": out["path"]}).json()
    assert imported["status"] == "imported" and imported["results"]["gaussians"] > 0
    assert client.delete(f"/api/runs/{imported['id']}").json()["ok"]
    assert len(client.get("/api/runs").json()["runs"]) == 1


def test_bad_dataset_is_reported(client, tmp_path):
    response = client.get("/api/dataset", params={"path": str(tmp_path)})
    assert response.status_code == 404
