"""Exp 22's viewer pages (index.html, frames.html, architecture.html, occlusion.html, render.html, sampling.html) plus a comment store for
architecture.html, occlusion.html's per-frame visibility API (/api/occ/...) and render.html's renderings
(/api/render/...).

    uv run python experiments/22_rgbd_prediction/viewer/server.py      -> http://localhost:8022/architecture.html

Comments: results/comments/<page>.json, a list of threads
{id, section, quote, text, author, created, resolved, replies: [{author, text, created}]}; the page polls it, so
replies written into the file (author "Claude") show up there.
"""

import json
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
STORE = HERE.parent / "results" / "comments"
app = FastAPI()


class Thread(BaseModel):
    section: str = ""
    quote: str = ""
    text: str
    author: str = "Sebastian"


class Reply(BaseModel):
    text: str
    author: str = "Sebastian"


def path(page):
    if not page.isidentifier():
        raise HTTPException(400, "bad page name")
    return STORE / f"{page}.json"


def load(page):
    p = path(page)
    return json.loads(p.read_text()) if p.exists() else []


def save(page, threads):
    STORE.mkdir(parents=True, exist_ok=True)
    tmp = path(page).with_suffix(".tmp")
    tmp.write_text(json.dumps(threads, indent=1, ensure_ascii=False))
    tmp.replace(path(page))


def find(threads, tid):
    for t in threads:
        if t["id"] == tid:
            return t
    raise HTTPException(404, "no such thread")


def now():
    return time.strftime("%Y-%m-%d %H:%M")


@app.get("/api/comments/{page}")
def get(page: str):
    return load(page)


@app.post("/api/comments/{page}")
def add(page: str, c: Thread):
    threads = load(page)
    threads.append(c.model_dump() | {"id": uuid.uuid4().hex[:8], "created": now(), "resolved": False, "replies": []})
    save(page, threads)
    return threads


@app.post("/api/comments/{page}/{tid}/reply")
def reply(page: str, tid: str, r: Reply):
    threads = load(page)
    find(threads, tid)["replies"].append(r.model_dump() | {"created": now()})
    save(page, threads)
    return threads


@app.post("/api/comments/{page}/{tid}/resolve")
def resolve(page: str, tid: str):
    threads = load(page)
    t = find(threads, tid)
    t["resolved"] = not t["resolved"]
    save(page, threads)
    return threads


@app.delete("/api/comments/{page}/{tid}")
def delete(page: str, tid: str):
    threads = load(page)
    find(threads, tid)
    save(page, [t for t in threads if t["id"] != tid])
    return [t for t in threads if t["id"] != tid]


# occlusion.html: visibility of every target SQ of a cached frame, computed on request (`occlusion_frame`)
OCC_N = 24  # surface samples per SQ: OCC_N x OCC_N (eta, omega) grid, the same grid as train.surface
_occ_cache = {}


@app.get("/api/occ/frames")
def occ_frames():
    import sys
    sys.path.insert(0, str(HERE.parent / "code"))
    from train import FRAMES, TEST, TRAIN
    return {sc: sorted(int(p.name[:6]) for p in (FRAMES / sc).glob("*_assign.npz")) for sc in TRAIN + TEST}


def frame_visibility(scene, fid, rule, kind, n):
    """Visibility of the surface samples of every target SQ of a cached frame. rule new: the z-buffer pixel owner +
    the owner / same-instance occlusion tests (train.OWNER_RULE "zbuffer"); old: the smallest-SDF pixel owner and
    the depth test only (both computed here, independent of the cache). kind: train.surface's sampling ("grid" /
    "cube"). -> v (view npz), a (assign npz), t, points (S, N, 3), state (S, N) uint8 (0 back-facing, 1 facing but
    outside the image, 2 facing, in the image but hidden, 3 visible), area (S, N), ratios {name: (S,)}, pixown (H, W)."""
    import sys
    sys.path.insert(0, str(HERE.parent / "code"))
    import numpy as np
    import torch
    import train as T
    v = np.load(T.FRAMES / scene / f"{fid:06d}_view.npz")
    a = np.load(T.FRAMES / scene / f"{fid:06d}_assign.npz")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tt = lambda x, dt=torch.float32: torch.tensor(np.asarray(x), dtype=dt, device=dev)
    t = {"scale": tt(v["scale"]), "exponents": tt(v["exponents"]), "rotation": tt(v["rotation"]),
         "translation": tt(v["translation"]), "instance": tt(v["instance"], torch.long)}
    S = len(v["scale"])
    t["slot"] = torch.arange(S, device=dev)
    depth = tt(v["depth"].astype(np.float32))[None]
    fb = torch.zeros(S, dtype=torch.long, device=dev)
    b = {"depth": depth, "label": tt(v["label"], torch.long)[None]}
    with torch.no_grad():
        T.OWNER_RULE = "zbuffer" if rule == "new" else "sdf"
        po, oz = T.pixel_owner(b, 0, t, with_depth=True)
        if rule == "new":
            vis, facing, p = T.target_visibility(t, fb, po[None], depth, n, ownz=oz[None], others=True, kind=kind)
        else:
            vis, facing, p = T.target_visibility(t, fb, po[None], depth, n, kind=kind)
        T.OWNER_RULE = "zbuffer"
        inimg = T.target_visibility(t, fb, po[None], depth, n, occlusion=False, kind=kind)[0]  # facing & in the image
        area = T.sample_areas(t["scale"], t["exponents"], n, kind)
    state = torch.where(vis, 3, torch.where(inimg, 2, torch.where(facing, 1, 0))).to(torch.uint8)
    A = lambda m: (area * m).sum(1)
    tot, fac = A(torch.ones_like(vis)), A(facing).clamp_min(1e-12)
    ratios = {"vis_total": A(vis) / tot.clamp_min(1e-12), "vis_facing": A(vis) / fac,
              "outside_facing": A(facing & ~inimg) / fac, "hidden_facing": A(inimg & ~vis) / fac}
    return v, a, t, p, state, area, ratios, po


def occlusion_frame(scene, fid, rule="new"):
    """-> (meta dict, binary): per target SQ in the frustum its surface samples (`train.surface`, OCC_N^2) with a
    state per sample (0 back-facing, 1 facing but outside the image, 2 facing, in the image but hidden, 3 visible;
    `train.target_visibility`; + 2 pole points per SQ, state 4) and the area-weighted ratios visible / total, visible / camera-facing (= f of
    `train.visible_fraction`), outside-image / facing, hidden / facing; the frame's RGB, depth points, the pixel owner
    SQ (`train.pixel_owner`). rule new: the z-buffer pixel owner + owner / same-instance occlusion tests
    (train.OWNER_RULE "zbuffer"); old: the smallest-SDF pixel owner and the depth test only (`frame_visibility`)."""
    key = (scene, fid, rule)
    if key in _occ_cache:
        return _occ_cache[key]
    import sys
    sys.path.insert(0, str(HERE.parent / "code"))
    import numpy as np
    import torch
    import train as T
    from model import CX, CY, FX, FY
    n = OCC_N
    v, a, t, p, state, area, ratios, pixown = frame_visibility(scene, fid, rule, "grid", n)
    S, dev = len(v["scale"]), t["scale"].device
    po = pixown.cpu().numpy().astype(np.int16)
    owned = np.bincount(po[po >= 0].ravel(), minlength=S)[:S] if S else np.zeros(0, int)
    classes = json.loads((T.FRAMES / "classes.json").read_text())
    d = v["depth"].astype(np.float32)
    vv, uu = np.indices(d.shape)
    ok = d > 0
    xyz = np.stack([(uu - CX) / FX * d, (vv - CY) / FY * d, d], -1)[ok]
    rgb = v["rgb"][ok]
    sub = np.arange(0, len(xyz), 2)
    xyz, rgb = xyz[sub], rgb[sub]
    import io
    from PIL import Image
    png = io.BytesIO()
    Image.fromarray(v["rgb"]).save(png, format="PNG")
    import base64
    meta = {"n": n, "sqs": S, "points": len(xyz), "H": int(d.shape[0]), "W": int(d.shape[1]),
            "ratios": {k: np.round(x.cpu().numpy(), 4).tolist() for k, x in ratios.items()},
            "f_cached": np.round(a["f"], 4).tolist() if "f" in a else None,  # the cache's rule (new since 2026-11-07)
            "pixels": owned.tolist(), "instance": v["instance"].tolist(), "cls": [classes[c] for c in v["cls"]],
            "scale": np.round(v["scale"], 3).tolist(), "png": base64.b64encode(png.getvalue()).decode()}
    # + the two pole points (eta = -/+ pi / 2) per SQ, which the (eta, omega) grid never samples: the viewer closes
    # the caps with them (state 4: "unsampled")
    poles = torch.stack([torch.zeros(S, 3, device=dev), torch.zeros(S, 3, device=dev)], 1)
    poles[:, 0, 2], poles[:, 1, 2] = -t["scale"][:, 2], t["scale"][:, 2]
    poles = poles @ t["rotation"].transpose(1, 2) + t["translation"][:, None]
    pts_all = torch.cat([p, poles], 1)  # (S, n * n + 2, 3)
    st_all = torch.cat([state, torch.full((S, 2), 4, dtype=torch.uint8, device=dev)], 1)
    blob = (pts_all.reshape(-1, 3).cpu().numpy().astype("<f4").tobytes() + st_all.reshape(-1).cpu().numpy().tobytes()
            + xyz.astype("<f4").tobytes() + rgb.astype("u1").tobytes() + po.astype("<i2").tobytes())
    if len(_occ_cache) > 64:
        _occ_cache.clear()
    _occ_cache[key] = (meta, blob)
    return meta, blob


@app.get("/api/occ/{scene}/{fid}/meta")
def occ_meta(scene: str, fid: int, rule: str = "new"):
    return occlusion_frame(scene, fid, rule)[0]


@app.get("/api/occ/{scene}/{fid}/data")
def occ_data(scene: str, fid: int, rule: str = "new"):
    from fastapi import Response
    return Response(occlusion_frame(scene, fid, rule)[1], media_type="application/octet-stream")


# render.html: the renderings of `code/render.py` (what the projection losses see), for the target SQs and a run's
# predictions (runs/<run>/predictions, saved by its final evaluation)
@app.get("/api/render/runs")
def render_runs():
    import sys
    sys.path.insert(0, str(HERE.parent / "code"))
    from train import OUT
    runs = [d for d in OUT.iterdir() if (d / "predictions").is_dir() and any((d / "predictions").iterdir())]
    runs.sort(key=lambda d: -d.stat().st_mtime)
    return [{"run": d.name, "frames": sorted(p.stem for p in (d / "predictions").glob("*.npz"))} for d in runs]


@app.get("/api/render/{run}/{name}")
def render_frame(run: str, name: str, stride: int = 2, qset: str = "train"):
    """GT (RGB png, instance label, class, depth at 168 x 300) and, for the target SQs (existence 1) and the run's
    predicted SQs (rendered as in training: queries with existence > render.MIN_EXIST or matched, alpha = existence x
    silhouette), per ray: front-most SQ (argmax weight; -1 if coverage < 0.5), its weight, coverage, rendered depth,
    the front SQ's instance (prediction: of its matched target, -1 unmatched) and class; plus render_metrics-style
    numbers. qset: which predicted queries (train: as the projection losses; exist05: existence > 0.5; matched).
    """
    import base64
    import io
    import sys
    sys.path.insert(0, str(HERE.parent / "code"))
    import numpy as np
    import torch
    from PIL import Image
    import render as R
    import train as T
    if not (run.replace("-", "").replace("_", "").replace(".", "").isalnum() and name.replace("_", "").isalnum()):
        raise HTTPException(400, "bad name")
    scene, fid = name.rsplit("_", 1)
    v = np.load(T.FRAMES / scene / f"{fid}_view.npz")
    pr = np.load(T.OUT / run / "predictions" / f"{name}.npz")
    classes = json.loads((T.FRAMES / "classes.json").read_text())
    dev = T.DEV
    tt = lambda x, dt=torch.float32: torch.tensor(np.asarray(x), dtype=dt, device=dev)
    H, W = v["depth"].shape
    lab = v["label"].astype(np.int64)
    lut = np.full(max(int(lab.max()), int(v["instance"].max(initial=0))) + 2, -1)
    lut[v["instance"]] = v["cls"]
    gcls = np.where(lab >= 0, lut[np.clip(lab, 0, None)], -1)
    dgt = v["depth"].astype(np.float32)

    def one(sq, exist, inst, cls):
        with torch.no_grad():
            o = R.render(sq, exist, H, W, stride=stride, grad=False)
        w = o["w"]
        cover = o["cover"]
        front = torch.where(cover > 0.5, w.argmax(1), -1) if w.shape[1] else torch.full_like(cover, -1, dtype=torch.long)
        fw = w.max(1).values if w.shape[1] else cover
        f = front.cpu().numpy()
        vv, uu = o["v"].cpu().numpy(), o["u"].cpu().numpy()
        finst = np.where(f >= 0, inst[np.clip(f, 0, None)], -1) if len(inst) else np.full_like(f, -1)
        fcls = np.where(f >= 0, cls[np.clip(f, 0, None)], -1) if len(cls) else np.full_like(f, -1)
        depth = np.where(cover.cpu().numpy() > 0.5, o["depth"].cpu().numpy(), 0)
        l, gc, dg = lab[vv, uu], gcls[vv, uu], dgt[vv, uu]
        okl, okd = l >= 0, (dg > 0) & (f >= 0)
        m = {"cover": float((f >= 0)[okl].mean()) if okl.any() else None,
             "inst_acc": float(((finst == l) & (f >= 0))[okl].mean()) if okl.any() else None,
             "cls_acc": float(((fcls == gc) & (f >= 0))[gc >= 0].mean()) if (gc >= 0).any() else None,
             "depth_absrel": float((np.abs(depth - dg) / np.maximum(dg, 1e-6))[okd].mean()) if okd.any() else None,
             "sqs": int(len(exist))}
        r2 = lambda x, k=3: np.round(x, k).tolist()
        return {"front": f.tolist(), "weight": r2(fw.cpu().numpy(), 2), "cover": r2(cover.cpu().numpy(), 2),
                "depth": r2(depth), "inst": finst.tolist(), "cls": fcls.tolist(), "metrics": m}

    tgt = {"scale": tt(v["scale"]), "shape": tt(v["exponents"]), "rotate": tt(v["rotation"]), "trans": tt(v["translation"])}
    out_t = one(tgt, torch.ones(len(v["scale"]), device=dev), v["instance"], v["cls"])
    ex = tt(pr["exist"])
    mq = tt(pr["match_q"], torch.long)
    if qset == "train":  # as the projection losses: existence > MIN_EXIST or matched
        q = R.frame_queries({"exist": ex[None]}, 0, mq).cpu().numpy()
    elif qset == "exist05":  # what evaluation counts as predicted
        q = np.flatnonzero(pr["exist"] > 0.5)
    else:  # matched queries only
        q = np.sort(pr["match_q"])
    qinst = np.full(len(pr["exist"]), -1)
    qinst[pr["match_q"]] = v["instance"][pr["match_t"]]
    sq = {k: tt(pr[k][q]) for k in ("scale", "shape", "rotate", "trans")}
    out_p = one(sq, ex[q], qinst[q], pr["cls"][q])
    out_p["query"] = q.tolist()
    out_p["exist"] = np.round(pr["exist"][q], 3).tolist()
    png = io.BytesIO()
    Image.fromarray(v["rgb"]).save(png, format="PNG")
    grid = [len(range(stride // 2, H, stride)), len(range(stride // 2, W, stride))]
    return {"H": H, "W": W, "grid": grid, "stride": stride, "classes": classes, "png": base64.b64encode(png.getvalue()).decode(),
            "label": lab.astype(int).ravel().tolist(), "gcls": gcls.astype(int).ravel().tolist(),
            "depth": np.round(dgt, 3).ravel().tolist(), "target": out_t, "pred": out_p,
            "target_instance": v["instance"].tolist(), "target_cls": v["cls"].tolist()}


# sampling.html: train.surface's two samplings side by side (the (eta, omega) grid and the radial cube grid)
SAMP_REF_M = 12  # cube grid side of the reference surface mesh
_samp_cache = {}


def sampling_payload(sq, n, vis=None, extra_meta=None, pts=None):
    """Both samplings of the SQs sq (dict scale / exponents / rotation / translation, numpy) -> (meta, binary):
    per kind points (S, N, 3) f32, state (S, N) u8 (vis = {kind: (state, ratios)}; else all 3), area (S, N) f32;
    then a reference surface (cube grid SAMP_REF_M, points (S, 6 m^2, 3)), then optional depth points / colors."""
    import sys
    sys.path.insert(0, str(HERE.parent / "code"))
    import numpy as np
    import torch
    import train as T
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    sc, ex, ro, tr = (tt(sq[k]) for k in ("scale", "exponents", "rotation", "translation"))
    S = len(sc)
    meta = {"sqs": S, "n": n, "m": T.cube_m(n), "ref_m": SAMP_REF_M, "kinds": {}}
    blob = b""
    with torch.no_grad():
        for kind in ("grid", "cube"):
            p = T.surface(sc, ex, ro, tr, n, kind)
            area = T.sample_areas(sc, ex, n, kind)
            st = vis[kind][0] if vis else torch.full(p.shape[:2], 3, dtype=torch.uint8, device=dev)
            meta["kinds"][kind] = {"N": int(p.shape[1]), "area_total": np.round(area.sum(1).cpu().numpy(), 4).tolist()}
            if vis:
                meta["kinds"][kind]["ratios"] = {k: np.round(x.cpu().numpy(), 4).tolist() for k, x in vis[kind][1].items()}
            blob += (p.cpu().numpy().astype("<f4").tobytes() + st.cpu().numpy().astype("u1").tobytes()
                     + area.cpu().numpy().astype("<f4").tobytes())
        ref = T.surface_cube(sc, ex, ro, tr, SAMP_REF_M)[0]
        for kind in ("grid", "cube"):  # coverage: distance from the fine reference surface to the nearest sample
            p = T.surface(sc, ex, ro, tr, n, kind)
            d = torch.cat([torch.cdist(ref[i:i + 64], p[i:i + 64]).min(2).values for i in range(0, S, 64)]) if S else ref[..., 0]
            meta["kinds"][kind]["gap_max"] = np.round(d.max(1).values.cpu().numpy(), 4).tolist() if S else []
            meta["kinds"][kind]["gap_mean"] = np.round(d.mean(1).cpu().numpy(), 4).tolist() if S else []
        blob += ref.cpu().numpy().astype("<f4").tobytes()
    if pts is not None:
        blob += pts[0].astype("<f4").tobytes() + pts[1].astype("u1").tobytes()
        meta["points"] = len(pts[0])
    return meta | (extra_meta or {}), blob


def sampling_frame(scene, fid, n, rule):
    key = (scene, fid, n, rule)
    if key not in _samp_cache:
        import sys
        sys.path.insert(0, str(HERE.parent / "code"))
        import numpy as np
        from model import CX, CY, FX, FY
        vis = {}
        for kind in ("grid", "cube"):
            v, a, t, p, state, area, ratios, po = frame_visibility(scene, fid, rule, kind, n)
            vis[kind] = (state, ratios)
        po = po.cpu().numpy()
        S = len(v["scale"])
        owned = np.bincount(po[po >= 0].ravel(), minlength=S)[:S] if S else np.zeros(0, int)
        import train as T
        classes = json.loads((T.FRAMES / "classes.json").read_text())
        d = v["depth"].astype(np.float32)
        vv, uu = np.indices(d.shape)
        ok = d > 0
        xyz = np.stack([(uu - CX) / FX * d, (vv - CY) / FY * d, d], -1)[ok][::2]
        rgb = v["rgb"][ok][::2]
        meta = {"cls": [classes[c] for c in v["cls"]], "instance": v["instance"].tolist(), "pixels": owned.tolist(),
                "scale": np.round(v["scale"], 3).tolist(), "exponents": np.round(v["exponents"], 3).tolist()}
        if len(_samp_cache) > 32:
            _samp_cache.clear()
        _samp_cache[key] = sampling_payload({k: v[k] for k in ("scale", "exponents", "rotation", "translation")}, n,
                                            vis, meta, (xyz, rgb))
    return _samp_cache[key]


@app.get("/api/samp/frame/{scene}/{fid}/{part}")
def samp_frame(scene: str, fid: int, part: str, n: int = 16, rule: str = "new"):
    from fastapi import Response
    meta, blob = sampling_frame(scene, fid, max(2, min(n, 64)), rule)
    return meta if part == "meta" else Response(blob, media_type="application/octet-stream")


@app.get("/api/samp/single/{part}")
def samp_single(part: str, a: float = 1.0, b: float = 0.6, c: float = 0.05, e1: float = 0.2, e2: float = 0.2, n: int = 16):
    """One SQ at the origin (no visibility; the viewer colors its samples by area)."""
    import numpy as np
    from fastapi import Response
    sq = {"scale": [[a, b, c]], "exponents": [[min(max(e1, 0.05), 2.0), min(max(e2, 0.05), 2.0)]],
          "rotation": [np.eye(3).tolist()], "translation": [[0.0, 0.0, 0.0]]}
    scale, eps = sq["scale"][0], sq["exponents"][0]
    meta, blob = sampling_payload(sq, max(2, min(n, 64)), extra_meta={"cls": ["synthetic SQ"], "instance": [0], "pixels": [0],
                                                                    "scale": [scale], "exponents": [eps]})
    return meta if part == "meta" else Response(blob, media_type="application/octet-stream")


app.mount("/", StaticFiles(directory=HERE, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8022)
