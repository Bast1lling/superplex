"""Viewer for the error of superquadric SDF approximations on a voxel grid around one SQ.

Run from the repo root:
    uv run python experiments/02_sq_sdf/viewer/server.py
then open http://localhost:8002
"""

import base64
from pathlib import Path
import sys
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel
import torch
import uvicorn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "code"))
from sdf import METHODS, reference  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


class FieldRequest(BaseModel):
    scale: list[float] = [1.0, 0.6, 0.4]
    eps: list[float] = [0.5, 0.5]
    res: int = 32  # voxels along the longest axis
    pad: float = 0.5  # margin around the SQ, as a fraction of its largest half-extent


def b64(x: torch.Tensor) -> str:
    return base64.b64encode(x.float().cpu().numpy().astype("<f4").tobytes()).decode()


def stats(d: torch.Tensor, gt: torch.Tensor) -> dict:
    err = d - gt
    out = {
        "mae": err.abs().mean().item(),
        "rmse": err.pow(2).mean().sqrt().item(),
        "max": err.abs().max().item(),
        "bias": err.mean().item(),
        "sign_err": ((d < 0) != (gt < 0)).float().mean().item(),  # inside/outside disagreement
        "lower_bound": (d.abs() <= gt.abs() + 1e-5).float().mean().item(),  # |d| never overestimates
    }
    return out


@app.post("/api/field")
@torch.no_grad()
def field(req: FieldRequest):
    a = torch.tensor(req.scale, device=DEVICE)
    half = a + req.pad * a.max()
    h = 2 * half.max().item() / req.res
    dims = torch.ceil(2 * half / h).long()
    axes = [(torch.arange(n, device=DEVICE) + 0.5) * h - n * h / 2 for n in dims.tolist()]
    q = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(1, -1, 3)  # x slowest, z fastest
    args = (a[None, None], torch.tensor([[req.eps[0]]], device=DEVICE), torch.tensor([[req.eps[1]]], device=DEVICE))

    torch.cuda.synchronize()
    t = time.perf_counter()
    gt = reference(q, *args)[0]
    torch.cuda.synchronize()
    times = {"reference": 1000 * (time.perf_counter() - t)}
    values = {}
    for name, fn in METHODS.items():
        t = time.perf_counter()
        values[name] = fn(q, *args)[0]
        torch.cuda.synchronize()
        times[name] = 1000 * (time.perf_counter() - t)

    near = gt.abs() < 0.1 * a.max()
    regions = {"all": torch.ones_like(near), "near": near, "inside": gt < 0, "outside": gt >= 0}
    return {
        "dims": dims.tolist(),
        "voxel": h,
        "origin": [ax[0].item() for ax in axes],
        "gt": b64(gt),
        "methods": {k: b64(v) for k, v in values.items()},
        "stats": {r: {k: stats(v[m], gt[m]) for k, v in values.items()} for r, m in regions.items() if m.any()},
        "times_ms": times,
        "near_band": 0.1 * a.max().item(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
