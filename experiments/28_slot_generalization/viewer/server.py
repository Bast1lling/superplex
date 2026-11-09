"""Viewer of exp 28: any frame of a run's scenes, its split, the run's model on it; train / held-out curves.

    uv run python experiments/28_slot_generalization/viewer/server.py      -> http://localhost:8028

Per run (results/large_files/runs/<run>/: args.json, model.pt, best.pt):
- frames: every cached frame of the run's training scenes and --eval_scenes, tagged train (used for training), block
  (training block, not used), gap, views (held-out views), scene (held-out scene);
- checkpoints: "final" / "best" run the model on the chosen frame on demand (cached as an exp 27-style single-frame
  run in results/large_files/views/<run>__ckpt-<c>__<frame>/, redone when the checkpoint is newer); the --n_show frames
  saved by train.py at each eval (`<run>__<split>__<frame>/steps/`) also offer those steps;
- curves: the eval metrics per split, from the trackio database (live while a run trains).
The per-frame payload (images, per-target table, SQs, masks) is exp 27's viewer (../../27_slot_attention/viewer),
imported unchanged. Needs results/large_files.
"""

import json
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP.parent / "27_slot_attention" / "viewer"))
import server as srv  # noqa: E402  (exp 27's viewer; puts exp 27's code on the path)
import common as C  # noqa: E402
import data as D  # noqa: E402
import train as T27  # noqa: E402

import importlib.util  # noqa: E402  (exp 28's train.py, for its splits; "train" is exp 27's)

_spec = importlib.util.spec_from_file_location("train28", EXP / "code" / "train.py")
T28 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T28)

LF = EXP / "results" / "large_files"
RUNS, VIEWS = LF / "runs", LF / "views"
DB = LF / "trackio" / "slot_generalization.db"
srv.RUNS = VIEWS  # exp 27's frame / mask endpoints read <VIEWS>/<view>/args.json + pred.npz | steps/
srv.EXP = EXP
app = FastAPI()


@app.middleware("http")
async def free_gpu(request, call_next):
    """Release the allocator's cached GPU memory after each request (it held ~10 GB, slowing training runs)."""
    resp = await call_next(request)
    torch.cuda.empty_cache()
    return resp


finite = srv.finite


def run_args(run):
    p = RUNS / run / "args.json"
    if not p.exists():
        raise HTTPException(404, f"no run {run}")
    return json.loads(p.read_text())


def split_of(a):
    """frame id 'scene/id' -> split tag for run args a."""
    train = set(a.get("train_ids", []))
    scenes = sorted({f.split("/")[0] for f in train})
    out = {}
    for s in scenes:
        ids = T28.scene_ids(s)
        tb, vb = set(T28.block(s, "train")), set(T28.block(s, "views"))
        for i in ids:
            f = f"{s}/{i}"
            out[f] = "train" if f in train else "block" if i in tb else "views" if i in vb else "gap"
    for s in [x for x in a.get("eval_scenes", "").split(",") if x]:
        for i in T28.scene_ids(s):
            out[f"{s}/{i}"] = "scene"
    return out


def tag(frame):
    return frame.replace("/", "_")


def saved_views(run):
    """frame -> (view dir name, saved steps) of the frames train.py saved for the viewer."""
    out = {}
    for d in VIEWS.glob(f"{run}__*"):
        parts = d.name.split("__")
        if len(parts) != 3 or parts[1].startswith("ckpt-") or not (d / "args.json").exists():
            continue
        fr = json.loads((d / "args.json").read_text())["frame"]
        out[fr] = (d.name, sorted(int(p.stem) for p in (d / "steps").glob("*.npz")) if (d / "steps").exists() else [])
    return out


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/runs")
def runs():
    out = []
    if RUNS.exists():
        for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not (d / "args.json").exists():
                continue
            a = json.loads((d / "args.json").read_text())
            res = EXP / "results" / "runs" / f"{d.name}.json"
            r = json.loads(res.read_text()) if res.exists() else {}
            out.append({"name": d.name, "args": {k: v for k, v in a.items() if k != "train_ids"}, "n_train_ids": len(a.get("train_ids", [])),
                        "done": res.exists(), "ckpts": [c for c in ("final", "best") if (d / ("model.pt" if c == "final" else "best.pt")).exists()],
                        "best": r.get("best"), "final": r.get("final")})
    return finite(out)


@app.get("/api/frames")
def frames(run: str):
    a = run_args(run)
    sv = saved_views(run)
    return [{"frame": f, "split": s, "steps": sv.get(f, (None, []))[1]} for f, s in split_of(a).items()]


@app.get("/api/curves")
def curves(run: str):
    if not DB.exists():
        return {}
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = c.execute("select step, metrics from metrics where run_name = ? order by step", (run,)).fetchall()
    out = {}
    for step, m in rows:
        m = json.loads(m)
        for k, v in m.items():
            key = k.split("/", 1)[1]
            if key.split("_")[0] in ("train", "views", "scenes") and key != "train_loss":
                out.setdefault(key, []).append([step, v])
        if "main/train_loss" in m and step % 50 == 0:
            out.setdefault("train_loss", []).append([step, m["main/train_loss"]])
    return finite(out)


@lru_cache(maxsize=4)
def model(run, ckpt, mtime):
    a = run_args(run)
    net = T28.build_model(argparse_ns(a))
    net.load_state_dict(torch.load(RUNS / run / ("model.pt" if ckpt == "final" else "best.pt"), map_location=C.DEV))
    return net.eval()


def view_name(run, frame, ckpt):
    """-> exp 27-style view name and step for (run, frame, ckpt); ckpt 'final' / 'best' (inference) or a saved step."""
    if ckpt in ("final", "best"):
        p = RUNS / run / ("model.pt" if ckpt == "final" else "best.pt")
        if not p.exists():
            raise HTTPException(404, f"{run}: no {ckpt} checkpoint yet")
        name = f"{run}__ckpt-{ckpt}__{tag(frame)}"
        d = VIEWS / name
        if not (d / "pred.npz").exists() or (d / "pred.npz").stat().st_mtime < p.stat().st_mtime:
            a = run_args(run)
            C.DISTANCE = "scaled"
            T27.A = argparse_ns(a)
            scene, fid = frame.split("/")
            fr = D.load_frame(scene, fid, a["mask_source"], a["part_targets"])
            if fr["M"] == 0:
                raise HTTPException(404, f"{frame}: no targets")
            net = model(run, ckpt, p.stat().st_mtime)
            fr["R_g"] = T28.gravity_rotation(np.load(D.FRAMES / scene / f"{fid}_view.npz")["c2w"])
            with torch.no_grad():
                pe = T28.forward(net, fr, argparse_ns(a))
                d.mkdir(parents=True, exist_ok=True)
                T27.save(pe, fr, T27.match(pe, fr), d / "pred.npz")
            (d / "args.json").write_text(json.dumps({k: v for k, v in a.items() if k != "train_ids"} | {"name": name, "frame": frame}))
        return name, "final"
    sv = saved_views(run).get(frame)
    if sv is None or int(ckpt) not in sv[1]:
        raise HTTPException(404, f"{run}: step {ckpt} not saved for {frame}")
    return sv[0], ckpt


def argparse_ns(a):
    class NS:
        pass
    ns = NS()
    for k, v in a.items():
        setattr(ns, k, v)
    ns.assign, ns.geo_owned = "gt", False
    return ns


@app.get("/api/frame")
@torch.no_grad()
def frame_view(run: str, frame: str, ckpt: str = "final"):
    name, step = view_name(run, frame, ckpt)
    out = srv.frame_view(run=name, step=step)
    a = run_args(run)
    out["split"] = split_of(a).get(frame, "?")
    out["frame_metrics"] = frame_metrics(name, step, frame, a)
    return finite(out)


@lru_cache(maxsize=256)
def _fm(name, step, frame, mask_source, mtime):
    z = srv.pred_file(name, step)
    fr = srv.frame(frame, mask_source, "scaled")
    keys = ["scale", "shape", "rotate", "trans"]
    pred = {k: torch.as_tensor(z[k], device=C.DEV).float() for k in keys}
    pred["exist_logit"] = torch.logit(torch.as_tensor(z["exist"], device=C.DEV).float().clamp(1e-6, 1 - 1e-6))
    pred["logits"] = torch.as_tensor(z["attn"], device=C.DEV).float().clamp_min(1e-8).log()
    if "p_scale" in z.files:
        pred["parts"] = {k: torch.as_tensor(z["p_" + k], device=C.DEV).float() for k in keys}
        pred["parts"]["exist_logit"] = torch.logit(torch.as_tensor(z["p_exist"], device=C.DEV).float().clamp(1e-6, 1 - 1e-6))
    return T27.metrics(pred, fr)


def frame_metrics(name, step, frame, a):
    p = VIEWS / name / ("pred.npz" if step == "final" else f"steps/{int(step):06d}.npz")
    return _fm(name, step, frame, a["mask_source"], p.stat().st_mtime)


@app.get("/api/mask")
@torch.no_grad()
def mask(run: str, frame: str, ckpt: str = "final", t: int = -1, slot: int = -1):
    name, step = view_name(run, frame, ckpt)
    return srv.mask(run=name, step=step, t=t, slot=slot)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8028)
