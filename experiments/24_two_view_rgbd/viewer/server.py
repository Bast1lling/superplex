"""Prediction viewer of exps 24 / 25 / 26 (the same file in each experiment's viewer/, it detects which one it is in).

    uv run python experiments/24_two_view_rgbd/viewer/server.py     -> http://localhost:8024
    (25 -> 8025, 26 -> 8026)

Lists the runs with saved predictions (results/<run>.json + results/large_files/runs/<run>/predictions/), their
evaluated frames with per-frame metrics, and per frame in 3D: the RGB point cloud of the view(s) (in view 0's camera
frame), the target SQs (visible / hidden) and the predicted SQs above an existence threshold, matched pairs in the
same color; plus the RGB image(s).
"""

import base64
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "code"))
import common as C  # noqa: E402
import data as D  # noqa: E402

KIND = "chairs" if EXP.name.startswith("26") else "replica"
PORT = 8000 + int(EXP.name[:2])
app = FastAPI()


def jpeg(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def backproject(depth, rgb, T=None, step=2):
    d = depth[::step, ::step]
    yy, xx = np.nonzero(d > 0)
    z = d[yy, xx]
    yy, xx = yy * step, xx * step
    x = np.stack([(xx - C.CX) / C.FX * z, (yy - C.CY) / C.FY * z, z], -1)
    if T is not None:
        x = x @ T[:3, :3].T + T[:3, 3]
    col = rgb[yy, xx] if rgb is not None else np.full((len(z), 3), 180, np.uint8)
    return x.astype(np.float32), col


@lru_cache(maxsize=64)
def frame_data(name):
    """-> targets dict (numpy), point cloud, colors, images."""
    if KIND == "chairs":
        m, v, t = name.rsplit("_", 2)
        v, t = int(v[1:]), int(t[1:])
        ds = D.Chairs.__new__(D.Chairs)
        ds.fit_split, ds.train = "test", False
        it = ds.make(m, v, t, 0)
        rgb = it["rgb"][0][1::4, 1::4]
        x, col = backproject(it["depth"][0], rgb, step=1)
        return it["targets"], x, col, [jpeg(it["rgb"][0])]
    parts = name.split("_")
    scene, f = "_".join(parts[:2]), int(parts[2])
    fr = [f] + ([int(parts[3])] if len(parts) > 3 else [])
    ds = D.Views.__new__(D.Views)
    ds.views, ds.partner = len(fr), "fixed"
    ds.items = [(scene, f, {fr[-1]: 0.5})]
    it = ds[0]
    xs, cols, imgs = [], [], []
    for vi, g in enumerate(fr):
        v = np.load(D.FRAMES / scene / f"{g:06d}_view.npz")
        x, col = backproject(v["depth"].astype(np.float32), v["rgb"], it["T0v"][vi])
        xs.append(x), cols.append(col), imgs.append(jpeg(v["rgb"]))
    return it["targets"], np.concatenate(xs), np.concatenate(cols), imgs


def sq_list(scale, shape, rot, trans, **extra):
    out = []
    for i in range(len(scale)):
        d = {"scale": scale[i].tolist(), "shape": shape[i].tolist(), "rot": rot[i].reshape(-1).tolist(), "trans": trans[i].tolist()}
        out.append(d | {k: (v[i].tolist() if hasattr(v[i], "tolist") else v[i]) for k, v in extra.items()})
    return out


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/info")
def info():
    return {"experiment": EXP.name, "kind": KIND}


@app.get("/api/runs")
def runs():
    out = []
    for p in sorted((EXP / "results").glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if "per_frame" not in r:
            continue
        pd = EXP / "results" / "large_files" / "runs" / p.stem / "predictions"
        if not pd.exists() and "fusion" not in (r.get("args") or {}):
            continue
        keys = ("pair_iou", "occ_iou", "occ_iou@0.3", "dec_recall@0.5", "enc_recall@0.5", "viou", "depth_absrel", "pair_iou_v0", "pair_iou_v0_partial")
        out.append({"name": p.stem, "has_pred": pd.exists(), "metrics": {k: r["test"][k] for k in keys if k in r["test"]}})
    return out


@app.get("/api/frames/{run}")
def frames(run):
    r = json.loads((EXP / "results" / f"{run}.json").read_text())
    return [{k: v for k, v in row.items() if isinstance(v, (int, float, str))} for row in r["per_frame"]]


@app.get("/api/frame/{run}/{name}")
def frame(run, name):
    p = EXP / "results" / "large_files" / "runs" / run / "predictions" / f"{name}.npz"
    if not p.exists():
        raise HTTPException(404, "no saved predictions (results/large_files missing?)")
    z = np.load(p)
    t, x, col, imgs = frame_data(name)
    match = {int(q): int(tt) for q, tt in zip(z["match_q"], z["match_t"])}
    pred = sq_list(z["scale"], z["shape"], z["rotate"], z["trans"], exist=z["exist"],
                   match=np.array([match.get(i, -1) for i in range(len(z["exist"]))]))
    vis = t["vis"].any(1) if len(t["vis"]) else np.zeros(0, bool)
    tg = sq_list(t["scale"], t["exponents"], t["rotation"], t["translation"], vis=vis, instance=t["instance"])
    sel = np.random.RandomState(0).permutation(len(x))[:60000]
    return {"pred": pred, "targets": tg, "points": x[sel].round(4).reshape(-1).tolist(),
            "colors": col[sel].reshape(-1).tolist(), "images": imgs}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
