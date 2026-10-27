"""FastAPI viewer for a random crowd of superquadrics.

Run from the repo root:
    uv run python experiments/00_sq_viewer/viewer/server.py
then open http://localhost:8000
"""

import colorsys
from pathlib import Path
import sys
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "code"))
from superquadric import grid_faces, random_crowd, sample_surface_grid  # noqa: E402
from supersq import components, decompose, mesh_union  # noqa: E402

app = FastAPI()


def node_colors(n: int) -> list[tuple[float, float, float]]:
    golden = 0.618033988749895
    return [colorsys.hsv_to_rgb((i * golden) % 1.0, 0.65, 0.95) for i in range(n)]


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/crowd")
def crowd(n: int = 40, seed: int = 0, n_eta: int = 32, n_omega: int = 64):
    sq, names = random_crowd(n=n, seed=seed)
    verts = sample_surface_grid(sq, n_eta, n_omega).reshape(len(sq), -1, 3)
    faces = grid_faces(n_eta, n_omega).flatten().tolist()
    colors = node_colors(len(sq))
    return {
        "faces": faces,  # shared by all primitives
        "primitives": [
            {
                "name": names[i],
                "vertices": verts[i].flatten().round(decimals=4).tolist(),
                "color": colors[i],
                "scale": sq.scale[i].tolist(),
                "eps": sq.eps[i].tolist(),
                "trans": sq.trans[i].tolist(),
            }
            for i in range(len(sq))
        ],
    }


class MergeRequest(BaseModel):
    n: int = 40
    seed: int = 0
    groups: list[list[int]] = []  # k-sets of node ids, k >= 2
    mode: Literal["blend", "simplex"] = "blend"
    voxel: float = 0.03


@app.post("/api/merge")
def merge(req: MergeRequest):
    """Mesh every connected component of the SQ hypergraph as one SuperSuperquadric."""
    sq, _ = random_crowd(n=req.n, seed=req.seed)
    groups = sorted({tuple(sorted(set(g))) for g in req.groups if len(set(g)) >= 2})
    colors = node_colors(len(sq))
    centers = sq.trans.numpy()
    out = []
    for comp in components(len(sq), [list(g) for g in groups]):
        if req.mode == "blend":
            pieces = [list(groups[gi]) for gi in comp]
        else:
            pieces = [p for gi in comp for p in decompose(list(groups[gi]), centers[list(groups[gi])])]
        verts, faces, vcols = mesh_union(sq, pieces, colors, voxel=req.voxel)
        out.append({
            "nodes": sorted({v for gi in comp for v in groups[gi]}),
            "pieces": pieces,
            "vertices": verts.flatten().round(decimals=4).tolist(),
            "faces": faces.flatten().tolist(),
            "colors": vcols.flatten().round(decimals=3).tolist(),
        })
    return {"components": out}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
