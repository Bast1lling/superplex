"""Viewer for PTv3 / Sonata features on Replica scenes.

Run from the experiment folder (own uv env):
    cd experiments/04_ptv3_replica && uv run python viewer/server.py
then open http://localhost:8004
"""

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
import numpy as np
from pydantic import BaseModel
import torch
import uvicorn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "code"))
from encoder import encode, list_scenes, load_model  # noqa: E402

app = FastAPI()
MODELS = {}  # name -> model (loaded lazily, kept on the GPU)
CURRENT = {}  # the one encoded scene kept in memory: key, encoding


def get_model(name: str):
    if name not in MODELS:
        MODELS[name] = load_model(pretrained=name == "sonata").cuda().eval()
    return MODELS[name]


def get_encoding(scene: str, model: str):
    key = (scene, model)
    if CURRENT.get("key") != key:
        CURRENT.clear()
        torch.cuda.empty_cache()
        CURRENT.update(key=key, enc=encode(get_model(model), scene))
    return CURRENT["enc"]


def stage_feat(enc, stage: str) -> torch.Tensor:
    """Per input point features of a stage ("0".."4") or Sonata's concatenated "full" feature."""
    if stage == "full":
        return enc["full"]
    st = enc["stages"][int(stage)]
    return st["feat"][st["assign"]]


def pca_rgb(feat: torch.Tensor) -> np.ndarray:
    f = feat - feat.mean(0)
    sub = f[torch.randperm(len(f), device=f.device)[:200_000]]
    _, _, v = torch.pca_lowrank(sub, q=3, center=False)
    p = f @ v[:, :3]
    lo, hi = torch.quantile(p[::10], 0.02, dim=0), torch.quantile(p[::10], 0.98, dim=0)
    return ((p - lo) / (hi - lo)).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/meta")
def meta():
    return {"scenes": list_scenes(), "models": ["sonata", "ptv3_scratch"], "stages": ["full", "0", "1", "2", "3", "4"]}


@app.get("/api/scene")
@torch.no_grad()
def scene(scene: str, model: str = "sonata"):
    """Binary: uint16 quantized positions, uint16 instance, int32 token id per stage (5), then uint8 rgb and
    uint8 PCA colors per stage option (6 x rgb)."""
    if scene not in list_scenes():
        raise HTTPException(404, "unknown scene")
    enc = get_encoding(scene, model)
    pos = enc["coord"].cpu().numpy()
    lo, hi = pos.min(0), pos.max(0)
    ext = np.maximum(hi - lo, 1e-6)
    blocks = [np.round((pos - lo) / ext * 65535).astype("<u2"), enc["instance"].cpu().numpy().astype("<u2")]
    blocks += [st["assign"].cpu().numpy().astype("<i4") for st in enc["stages"]]
    rgb = ((enc["color"] + 1) * 127.5).clamp(0, 255).round().to(torch.uint8).cpu().numpy()  # NormalizeColor inverse
    blocks += [rgb] + [pca_rgb(stage_feat(enc, s)) for s in meta()["stages"]]
    headers = {"X-Count": str(len(pos)), "X-Min": ",".join(f"{x:.6f}" for x in lo),
               "X-Extent": ",".join(f"{x:.6f}" for x in ext),
               "X-Tokens": ",".join(str(len(st["feat"])) for st in enc["stages"])}
    return Response(b"".join(np.ascontiguousarray(b).tobytes() for b in blocks),
                    media_type="application/octet-stream", headers=headers)


class SimilarityRequest(BaseModel):
    scene: str
    model: str = "sonata"
    stage: str = "full"
    index: int


@app.post("/api/similarity")
@torch.no_grad()
def similarity(req: SimilarityRequest):
    """Cosine similarity of every point's feature to the picked point's feature, as uint8 (0 -> -1, 255 -> 1)."""
    enc = get_encoding(req.scene, req.model)
    f = torch.nn.functional.normalize(stage_feat(enc, req.stage), dim=1)
    sim = f @ f[req.index]
    return Response(((sim + 1) * 127.5).clamp(0, 255).round().to(torch.uint8).cpu().numpy().tobytes(),
                    media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
