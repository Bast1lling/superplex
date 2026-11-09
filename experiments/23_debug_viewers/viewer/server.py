"""Debug viewers for exp 22's RGB-D -> SQ predictor (features.html, results.html).

    uv run python experiments/23_debug_viewers/viewer/server.py        -> http://localhost:8023

Feature API (features.html), on the cached frames of exp 22 (Thesis/data/replica_rgbd_frames):
- /api/frames: scenes -> frame ids; the overfit frame, the 16 frames, the frames with exported predictions;
- /api/img/<frame>/<kind>.png?mode=frame|global: rgb, depth, label, or a PCA (3 components -> RGB) of one feature set:
  dino (42 x 75 x 1024), s2 / s3 / s4 (Sonata stages per pixel, 192 / 384 / 512), sonata (the 3 stages concatenated,
  1088 per pixel), sonata_patch (max over the patch's pixels with depth, as DenseModel's input), enc:<run> (the 128-d
  encoder output tokens of an exp 22 checkpoint). mode=frame: PCA of this frame only; mode=global: one PCA basis per
  kind fitted on 2 frames of every scene, so colours are comparable across frames and scenes;
- /api/sim/<frame>/<kind>.png?u=..&v=.. (0..1 image coordinates): cosine similarity of the feature at (u, v) to all
  features of the frame (or of another frame with &other=<frame>).
Results API (results.html): /api/cloud/<frame> (binary: float32 xyz, uint8 rgb of every pixel with depth, camera
frame), /api/gt/<frame>, /api/pred/<run>/<frame>, /api/summary, /api/classes; these read code/predict.py's export in
results/large_files and answer 404 when it is missing.
"""

import functools
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from matplotlib import colormaps
from PIL import Image

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP.parent / "22_rgbd_prediction" / "code"))
from model import CX, CY, FX, FY, GH, GW, P, DenseModel, TopKModel  # noqa: E402

FRAMES = Path("/home/sebastian/Documents/Uni/Thesis/data/replica_rgbd_frames")
RUNS22 = EXP.parent / "22_rgbd_prediction" / "results" / "large_files" / "runs"
LARGE = EXP / "results" / "large_files"
SCENES = sorted(p.name for p in FRAMES.iterdir() if p.is_dir())
CLASSES = json.loads((FRAMES / "classes.json").read_text())
app = FastAPI()


def path(name):
    scene, fid = name.rsplit("_", 1)
    p = FRAMES / scene / fid
    if scene not in SCENES or not Path(f"{p}_view.npz").exists():
        raise HTTPException(404, f"no frame {name}")
    return p


@functools.lru_cache(64)
def view(name):
    v = np.load(f"{path(name)}_view.npz")
    return {k: v[k] for k in ("rgb", "depth", "label")}


@functools.lru_cache(8)
def model(run):
    cfg = json.loads((EXP.parent / "22_rgbd_prediction" / "results" / f"{run}.json").read_text())["args"] \
        if (EXP.parent / "22_rgbd_prediction" / "results" / f"{run}.json").exists() else {"topk": "topk" in run}
    m = (TopKModel if cfg.get("topk") else DenseModel)(len(CLASSES))
    m.load_state_dict(torch.load(RUNS22 / run / "model.pt", map_location="cpu"))
    return m.eval()


@functools.lru_cache(32)
def features(name, kind):
    """-> (H, W, C) float32 and a validity mask (H, W)."""
    p = path(name)
    if kind == "dino":
        f = np.load(f"{p}_dino.npy").astype(np.float32)
        return f, np.ones(f.shape[:2], bool)
    s = np.load(f"{p}_sonata.npz")
    idx = s["idx"]  # (3, H, W), -1 without depth
    valid = idx[0] >= 0
    stages = [s[k].astype(np.float32)[np.maximum(idx[i], 0)] * valid[..., None] for i, k in enumerate(("f2", "f3", "f4"))]
    if kind in ("s2", "s3", "s4"):
        return stages[int(kind[1]) - 2], valid
    cat = np.concatenate(stages, -1)
    if kind == "sonata":
        return cat, valid
    if kind == "sonata_patch":
        g = lambda x: x.reshape(GH, P, GW, P, -1).transpose(0, 2, 1, 3, 4).reshape(GH, GW, P * P, -1)
        v = g(valid[..., None])
        return np.where(v, g(cat), -1e4).max(2) * v.any(2), v.any(2)[..., 0]
    if kind.startswith("enc:"):
        run = kind[4:]
        if not (RUNS22 / run / "model.pt").exists():
            raise HTTPException(404, f"no checkpoint for {run}")
        dino = torch.tensor(np.load(f"{p}_dino.npy")).float()[None]
        feats = [torch.tensor(np.concatenate([s[k], np.zeros((1, s[k].shape[1]), s[k].dtype)]).astype(np.float32))
                 for k in ("f2", "f3", "f4")]
        ix = [torch.tensor(np.where(idx[i] >= 0, idx[i], len(f) - 1))[None] for i, f in enumerate(feats)]
        depth = torch.tensor(view(name)["depth"].astype(np.float32))[None]
        with torch.no_grad():
            _, q = model(run).encode(dino, feats, ix, depth)
        return q[0].reshape(GH, GW, -1).numpy(), np.ones((GH, GW), bool)
    raise HTTPException(400, f"unknown kind {kind}")


def pca_fit(x):
    """x (N, C) -> (mean, basis (C, 3), lo, hi) with lo / hi the 1st / 99th percentiles of the projections."""
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    basis = vt[:3].T
    proj = (x - mu) @ basis
    return mu, basis, np.percentile(proj, 1, 0), np.percentile(proj, 99, 0)


def sample(f, valid, n, rng):
    x = f[valid]
    return x[rng.choice(len(x), min(n, len(x)), replace=False)]


@functools.lru_cache(16)
def global_pca(kind):
    rng = np.random.default_rng(0)
    xs = []
    for scene in SCENES:
        ids = sorted(p.name[:6] for p in (FRAMES / scene).glob("*_view.npz"))
        for fid in (ids[len(ids) // 4], ids[3 * len(ids) // 4]):
            f, v = features(f"{scene}_{fid}", kind)
            xs.append(sample(f, v, 3000, rng))
    return pca_fit(np.concatenate(xs))


def png(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


def colors(n):
    rng = np.random.default_rng(1)
    return (rng.random((n, 3)) * 200 + 40).astype(np.uint8)


@app.get("/api/frames")
def frames():
    out = {s: sorted(p.name[:6] for p in (FRAMES / s).glob("*_view.npz")) for s in SCENES}
    meta = json.loads((EXP / "results" / "frames.json").read_text()) if (EXP / "results" / "frames.json").exists() else {}
    runs = sorted(p.name for p in RUNS22.iterdir() if (p / "model.pt").exists() and ("dense" in p.name or "topk" in p.name))
    return {"scenes": out, "meta": meta, "runs": runs}


@app.get("/api/img/{name}/{kind}.png")
def img(name: str, kind: str, mode: str = "frame"):
    if kind == "rgb":
        return png(view(name)["rgb"])
    if kind == "depth":
        d = view(name)["depth"].astype(np.float32)
        rgb = (colormaps["turbo"](np.clip(d / 6.0, 0, 1))[..., :3] * 255).astype(np.uint8)
        rgb[d <= 0] = 40
        return png(rgb)
    if kind == "label":
        lab = view(name)["label"].astype(int)
        rgb = colors(lab.max() + 2)[lab + 1]
        rgb[lab < 0] = 0
        return png(rgb)
    f, valid = features(name, kind)
    mu, basis, lo, hi = global_pca(kind) if mode == "global" else pca_fit(sample(f, valid, 20000, np.random.default_rng(0)))
    proj = np.clip(((f - mu) @ basis - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb = (proj * 255).astype(np.uint8)
    rgb[~valid] = 40
    return png(rgb)


@app.get("/api/sim/{name}/{kind}.png")
def sim(name: str, kind: str, u: float, v: float, other: str = Query(None)):
    f, valid = features(name, kind)
    y, x = min(int(v * f.shape[0]), f.shape[0] - 1), min(int(u * f.shape[1]), f.shape[1] - 1)
    q = f[y, x] / (np.linalg.norm(f[y, x]) + 1e-8)
    g, gv = features(other, kind) if other else (f, valid)
    s = (g @ q) / (np.linalg.norm(g, axis=-1) + 1e-8)
    rgb = (colormaps["magma"](np.clip(s, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[~gv] = 40
    return png(rgb)


@app.get("/api/cloud/{name}")
def cloud(name: str):
    v = view(name)
    d = v["depth"].astype(np.float32)
    H, W = d.shape
    vv, uu = np.mgrid[0:H, 0:W]
    ok = d > 0
    xyz = np.stack([(uu - CX) / FX * d, (vv - CY) / FY * d, d], -1)[ok].astype(np.float32)
    rgb = v["rgb"][ok].astype(np.uint8)
    return Response(np.int32(len(xyz)).tobytes() + xyz.tobytes() + rgb.tobytes(), media_type="application/octet-stream")


def large(p):
    if not p.exists():
        raise HTTPException(404, f"{p.relative_to(EXP)} missing: run code/predict.py")
    return FileResponse(p, media_type="application/json")


@app.get("/api/gt/{name}")
def gt(name: str):
    return large(LARGE / "gt" / f"{name}.json")


@app.get("/api/pred/{run}/{name}")
def pred(run: str, name: str):
    return large(LARGE / "predictions" / run / f"{name}.json")


@app.get("/api/summary")
def summary():
    return large(EXP / "results" / "predict_summary.json")


@app.get("/api/classes")
def classes():
    return CLASSES


app.mount("/", StaticFiles(directory=HERE, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8023)
