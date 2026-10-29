"""Experiment 01's scene viewer plus the camera views of the NICE-SLAM Replica renderings.

    uv run python experiments/11_render_views/viewer/server.py     -> http://localhost:8011

Renderings: Thesis/data/replica_nice_slam/Replica/<room0|office0|...>/results/{frame,depth}NNNNNN.{jpg,png}, poses in
traj.txt (one camera-to-world 4x4 per line, OpenCV axes: x right, y down, z forward), same world frame as the Replica
meshes (checked: points project onto the rendered depth with a median error of 2-3 mm).
"""

from functools import lru_cache
from pathlib import Path
import importlib.util
import json

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import uvicorn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("viewer01", HERE.parents[1] / "01_scene_viewer" / "viewer" / "server.py")
viewer01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer01)
from scenes import DATA  # noqa: E402  (01's code dir, on the path via viewer01)

RENDERS = DATA / "replica_nice_slam" / "Replica"
CAM = json.loads((RENDERS / "cam_params.json").read_text())["camera"]
DEPTH_TOL = 0.02  # a point is visible in a view if it projects into the image within 2 cm of the rendered depth
LABEL_TOL = 0.02  # a pixel takes the instance of the nearest point if it is within 2 cm of its 3D surface point

app = viewer01.app
app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]


def render_dir(scene: str) -> Path:
    """room_0 -> .../Replica/room0"""
    d = RENDERS / scene.replace("_", "")
    if not (d / "traj.txt").exists():
        raise HTTPException(404, f"no renderings for {scene}")
    return d


@lru_cache(maxsize=8)
def poses(scene: str) -> np.ndarray:
    return np.loadtxt(render_dir(scene) / "traj.txt").reshape(-1, 4, 4)


@lru_cache(maxsize=2)
def positions(scene: str) -> np.ndarray:
    """Point positions in the order of 01's /api/points."""
    return viewer01.load_points(scene)["pos"].astype(np.float64)


@lru_cache(maxsize=2)
def label_tree(scene: str) -> tuple[cKDTree, np.ndarray]:
    return cKDTree(positions(scene)), viewer01.load_points(scene)["instance"]


def depth(scene: str, frame: int) -> np.ndarray:
    return np.asarray(Image.open(render_dir(scene) / "results" / f"depth{frame:06d}.png")) / CAM["scale"]


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/views")
def views(scene: str):
    try:
        T = poses(scene)
    except HTTPException:
        return {"c2w": [], "camera": CAM}
    return {"c2w": T.round(5).reshape(len(T), 16).tolist(), "camera": CAM}


@app.get("/api/view_image")
def view_image(scene: str, frame: int):
    return FileResponse(render_dir(scene) / "results" / f"frame{frame:06d}.jpg")


@app.get("/api/view_points")
def view_points(scene: str, frame: int):
    """Indices (uint32) of the points visible in the view: in front of the camera, inside the image, and within
    DEPTH_TOL of the rendered depth."""
    D = depth(scene, frame)
    w2c = np.linalg.inv(poses(scene)[frame])
    x = positions(scene) @ w2c[:3, :3].T + w2c[:3, 3]
    z = np.maximum(x[:, 2], 1e-6)
    u = np.round(CAM["fx"] * x[:, 0] / z + CAM["cx"]).astype(int)
    v = np.round(CAM["fy"] * x[:, 1] / z + CAM["cy"]).astype(int)
    ok = (x[:, 2] > 0.05) & (u >= 0) & (u < CAM["w"]) & (v >= 0) & (v < CAM["h"])
    idx = np.nonzero(ok)[0]
    idx = idx[np.abs(z[idx] - D[v[idx], u[idx]]) < DEPTH_TOL]
    return Response(idx.astype("<u4").tobytes(), media_type="application/octet-stream", headers={"X-Count": str(len(idx))})


@app.get("/api/view_labels")
def view_labels(scene: str, frame: int):
    """GT instance per pixel (uint16, row-major h x w; 0 = no label): every pixel is back-projected with the rendered
    depth and takes the instance of the nearest point, if that is within LABEL_TOL (else: glass, unscanned)."""
    D = depth(scene, frame)
    v, u = np.nonzero(D > 0)
    d = D[v, u]
    cam = np.stack([d * (u - CAM["cx"]) / CAM["fx"], d * (v - CAM["cy"]) / CAM["fy"], d], 1)
    T = poses(scene)[frame]
    tree, inst = label_tree(scene)
    dist, nn = tree.query(cam @ T[:3, :3].T + T[:3, 3], distance_upper_bound=LABEL_TOL, workers=-1)
    out = np.zeros(D.shape, "<u2")
    hit = np.isfinite(dist)
    out[v[hit], u[hit]] = inst[nn[hit]]
    return Response(out.tobytes(), media_type="application/octet-stream",
                    headers={"X-Width": str(D.shape[1]), "X-Height": str(D.shape[0])})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8011)
