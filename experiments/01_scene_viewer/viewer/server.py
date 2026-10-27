"""FastAPI point cloud viewer for the Replica scenes (1 cm voxel point clouds).

Run from the repo root:
    uv run python experiments/01_scene_viewer/viewer/server.py
then open http://localhost:8001
"""

from functools import lru_cache
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
import numpy as np
import uvicorn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "code"))
from scenes import VOXEL, list_scenes, load_points, load_semantics  # noqa: E402

app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/scenes")
def scenes():
    return {"scenes": list_scenes(), "voxel": VOXEL, "up": [0, 0, 1]}


@app.get("/api/semantics")
def semantics(scene: str):
    if scene not in list_scenes():
        raise HTTPException(404, "unknown scene")
    return load_semantics(scene)


@lru_cache(maxsize=4)
def packed(scene: str) -> tuple[bytes, dict]:
    """uint16 blocks first (positions quantized within the bounding box, instance, segment), then uint8 (rgb, normal)."""
    p = load_points(scene)
    pos = p["pos"]
    lo, hi = pos.min(0), pos.max(0)
    ext = np.maximum(hi - lo, 1e-6)
    q = np.round((pos - lo) / ext * 65535).astype("<u2")
    nrm = np.round((p["nrm"].astype(np.float32) * 0.5 + 0.5) * 255).astype(np.uint8)
    blocks = [q, p["instance"].astype("<u2"), p["segment"].astype("<u2"), p["col"], nrm]
    headers = {
        "X-Count": str(len(pos)),
        "X-Min": ",".join(f"{x:.6f}" for x in lo),
        "X-Extent": ",".join(f"{x:.6f}" for x in ext),
    }
    return b"".join(np.ascontiguousarray(b).tobytes() for b in blocks), headers


@app.get("/api/points")
def points(scene: str):
    if scene not in list_scenes():
        raise HTTPException(404, "unknown scene")
    body, headers = packed(scene)
    return Response(body, media_type="application/octet-stream", headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
