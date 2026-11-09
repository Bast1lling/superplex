"""Sanity-check viewer of exp 27: targets, labels, visibility weights and the slot model's outputs per eval step.

    uv run python experiments/27_slot_attention/viewer/server.py         -> http://localhost:8027

Runs: "data" (no model: the frame's labels / targets / sample weights for a chosen mask source) and every run in
results/large_files/runs/<run>/ (args.json; steps/<step>.npz saved at each evaluation, pred.npz at the end).
Per (run, step): RGB / GT label / rendered target SQs / rendered predicted SQs (colored by matched target) / error images, the labelled point cloud, kept and dropped target SQs,
all slots (SQ, existence, matched target, anchor), target cube samples with their visibility weights, free-space
samples, per-target table; masks (GT share vs. matched slot attention) on demand.
Needs results/large_files (predictions); the "data" run works without it.
"""

import base64
import colorsys
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "code"))
import common as C  # noqa: E402
import data as D  # noqa: E402
from train import group_box, group_inside, group_pair_iou, pair_iou_mc, weighted_chamfer  # noqa: E402

RUNS = EXP / "results" / "large_files" / "runs"
DEFAULTS = {"frame": "room_2/000500", "mask_source": "sdf", "distance": "scaled", "w_occ": 1.0, "lam": 0.1}
app = FastAPI()


@app.middleware("http")
async def free_gpu(request, call_next):
    """Release the allocator's cached GPU memory after each request (it held ~10 GB, slowing training runs)."""
    resp = await call_next(request)
    torch.cuda.empty_cache()
    return resp


def hue(i):
    return np.array(colorsys.hls_to_rgb((i * 0.618034) % 1, 0.55, 0.65))


def png(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def render_labels(sqs, color_idx, n_colors):
    """Hard rendering (`common.render`, stride 1, existence 1) of the SQs (dict of N) -> (H, W, 3) image: each pixel
    in the color of the SQ with the largest compositing weight (color_idx (N,): target index, -1 = gray), black
    where the coverage is < 0.5."""
    N = len(sqs["scale"])
    img = np.zeros((C.H, C.W, 3), np.uint8)
    if N == 0:
        return img
    r = C.render(sqs, torch.ones(N, device=C.DEV), stride=1, grad=False)
    best = r["w"].argmax(1).cpu().numpy()
    cover = r["cover"].cpu().numpy() >= 0.5
    pal = np.concatenate([np.array([hue(i) for i in range(max(n_colors, 1))]), [[0.55, 0.55, 0.55]]])
    ci = np.asarray(color_idx)[best]
    col = (pal[np.where(ci >= 0, ci, len(pal) - 1)] * 255).astype(np.uint8)
    v, u = r["v"].cpu().numpy(), r["u"].cpu().numpy()
    img[v[cover], u[cover]] = col[cover]
    return img


def superflex_by_instance(alls, inst, fr, M):
    """The frame's SuperFlex SQs rendered with all SQs of a GT instance in one color: instance-level mask sources
    (instance, inst1): the color of the instance's target (as the GT labels; gray = not a target); per-SQ sources:
    a color per instance id."""
    if fr["source"] in ("instance", "inst1"):
        t_of = {int(i): k for k, i in enumerate(fr["instance"].tolist())}
        return render_labels(alls, np.array([t_of.get(int(i), -1) for i in inst]), M)
    return render_labels(alls, inst, int(inst.max()) + 1 if len(inst) else 1)


def colorize(lab, n):
    pal = np.concatenate([np.array([hue(i) for i in range(max(n, 1))]), [[0.15, 0.15, 0.15]]])  # last: ignore
    return (pal[np.where(lab >= 0, lab, len(pal) - 1)] * 255).astype(np.uint8)


def heat(x, vmax):
    """0 .. vmax -> dark blue .. yellow (simple ramp), NaN -> black."""
    t = np.clip(np.nan_to_num(x, nan=0.0) / vmax, 0, 1)[..., None]
    lo, mid, hi = np.array([0.05, 0.05, 0.3]), np.array([0.85, 0.2, 0.35]), np.array([1.0, 0.95, 0.3])
    c = np.where(t < 0.5, lo + (mid - lo) * t * 2, mid + (hi - mid) * (t - 0.5) * 2)
    c = np.where(np.isnan(x)[..., None], 0.0, c)
    return (c * 255).astype(np.uint8)


def run_args(run):
    if run == "data":
        return dict(DEFAULTS)
    p = RUNS / run / "args.json"
    if not p.exists():
        raise HTTPException(404, f"no run {run} (results/large_files missing?)")
    return DEFAULTS | json.loads(p.read_text())


@lru_cache(maxsize=8)
def frame(frame_id, source, distance):
    C.DISTANCE = distance
    scene, fid = frame_id.split("/")
    return D.load_frame(scene, fid, source)


def pred_file(run, step):
    if run == "data":
        return None
    p = RUNS / run / ("pred.npz" if step == "final" else f"steps/{int(step):06d}.npz")
    if not p.exists():
        raise HTTPException(404, f"no saved step {step} for {run}")
    return np.load(p)


def sqs(d, prefix=""):
    return {k: torch.as_tensor(d[prefix + k], device=C.DEV).float() for k in C.KEYS}


def sq_json(s, i):
    return {"scale": s["scale"][i].tolist(), "shape": s["shape"][i].tolist(), "rot": s["rotate"][i].reshape(-1).tolist(),
            "trans": s["trans"][i].tolist()}


def point_group_dist(x, lab, gr):
    """|SDF| of points x (Q, 3) to the union of their target group's SQs (smallest over the group's SQs)."""
    out = []
    for a in range(0, len(x), 16384):
        sl = slice(a, a + 16384)
        d = C.sdf(x[sl, None], {k: v[lab[sl]] for k, v in gr["sq"].items()})[:, :, 0].abs()  # (q, G)
        out.append(torch.where(gr["valid"][lab[sl]], d, torch.inf).min(1).values)
    return torch.cat(out)


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


def finite(o):
    """NaN / inf -> None, recursively (JSON has no NaN: one run with a NaN metric broke the run list)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [finite(v) for v in o]
    return o


@app.get("/api/runs")
def runs():
    out = [{"name": "data", "steps": [], "final": False, "args": DEFAULTS, "metrics": {}}]
    if RUNS.exists():
        for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not (d / "args.json").exists():
                continue
            steps = sorted(int(p.stem) for p in (d / "steps").glob("*.npz")) if (d / "steps").exists() else []
            res = EXP / "results" / "runs" / f"{d.name}.json"
            m = json.loads(res.read_text()).get("final") or {} if res.exists() else {}
            out.append({"name": d.name, "steps": steps, "final": (d / "pred.npz").exists(),
                        "args": json.loads((d / "args.json").read_text()), "metrics": m})
    return finite(out)


@app.get("/api/frame")
@torch.no_grad()
def frame_view(run: str = "data", step: str = "final", source: str = ""):
    a = run_args(run)
    if run == "data" and source:
        a["mask_source"] = source
    fr = frame(a["frame"], a["mask_source"], a["distance"])
    C.DISTANCE = a["distance"]
    z = pred_file(run, step)
    gr = fr["groups"]
    M, lab = fr["M"], fr["lab"].cpu().numpy()
    scene, fid = a["frame"].split("/")
    v = np.load(D.FRAMES / scene / f"{fid}_view.npz")
    rgb = v["rgb"]

    # target fit: |SDF| of every labelled pixel to its own target
    own_t = point_group_dist(fr["pts"], fr["pts_lab"], gr)
    fit = np.full(lab.size, np.nan, np.float32)
    m = lab.reshape(-1) >= 0
    fit[m] = own_t.cpu().numpy()

    # dropped frustum SQs (not targets)
    allt = {k: torch.as_tensor(v[k], device=C.DEV).float() for k in ("scale", "exponents", "rotation", "translation")}
    alls = C.tsq(allt)
    kept = set(fr["flat_idx"].tolist())
    dropped = [sq_json(alls, i) | {"id": i} for i in range(len(allt["scale"])) if i not in kept]

    out = {"args": a, "M": M, "n_all": int(fr["n_all"]), "images": {"rgb": png(rgb), "labels": png(colorize(lab, M)),
           "target_fit": png(heat(fit.reshape(lab.shape), 0.1)),
           "target_render": png(render_labels(fr["flat"], fr["flat_owner"].cpu().numpy(), M)),
           "superflex_render": png(render_labels(alls, np.arange(len(allt["scale"])), len(allt["scale"]))),
           "superflex_inst": png(superflex_by_instance(alls, v["instance"].astype(np.int64), fr, M))},
           "dropped": dropped}

    # points (subsample)
    xyz_all = D.backproject(fr["depth"]).cpu().numpy()
    has = fr["depth"].reshape(-1).cpu().numpy() > 0
    sel = np.flatnonzero(has)
    sel = sel[np.random.RandomState(0).permutation(len(sel))[:50000]]
    patch = (sel // C.W // C.P) * C.GW + (sel % C.W) // C.P
    pts = {"xyz": xyz_all[sel].round(4).reshape(-1).tolist(), "rgb": rgb.reshape(-1, 3)[sel].reshape(-1).tolist(),
           "label": lab.reshape(-1)[sel].tolist(), "fit": np.nan_to_num(fit[sel], nan=-1).round(4).tolist()}

    # target cube samples with visibility weights; free-space samples
    ts = gr["pts"]
    w = D.sample_weights(ts, fr["depth"], a["w_occ"], a["lam"]) * gr["keep"]
    d = fr["depth"].reshape(-1)
    ok = torch.nonzero(d > C.ZNEAR + 0.06)[:, 0]
    g = torch.Generator(device=C.DEV).manual_seed(0)
    r = ok[torch.randint(len(ok), (4000,), device=C.DEV, generator=g)]
    zz = C.ZNEAR + torch.rand(4000, device=C.DEV, generator=g) * (d[r] - 0.05 - C.ZNEAR)
    free = C.rays(1, C.DEV)[0][r] * zz[:, None]
    out["samples"] = {"xyz": ts.reshape(-1, 3).cpu().numpy().round(4).reshape(-1).tolist(),
                      "w": w.reshape(-1).cpu().numpy().round(3).tolist(),
                      "t": np.repeat(np.arange(M), ts.shape[1]).tolist()}
    out["free"] = free.cpu().numpy().round(4).reshape(-1).tolist()

    tinfo = [{"id": i, "px": int((lab == i).sum()), "instance": int(fr["instance"][i]), "n_sq": int(gr["valid"][i].sum()),
              "fit_med": float(own_t[fr["pts_lab"] == i].median()), "slot": -1} for i in range(M)]
    flat, owner = fr["flat"], fr["flat_owner"].tolist()
    out["target_sqs"] = [sq_json(flat, j) | {"t": owner[j]} for j in range(len(owner))]
    if z is not None:
        ps = sqs(z)
        attn = z["attn"].astype(np.float32)  # (NP, K)
        K = attn.shape[1]
        sig = z["sig"]
        slot2t = np.full(K, -1)
        slot2t[sig] = np.arange(M)
        am = attn.argmax(1)
        # predicted masks: each patch in the color of its slot's matched target (gray: unmatched slot); m2f patches
        # whose best exist * mask is < 0.5 are black (no instance)
        amap = slot2t[am].reshape(C.GH, C.GW)
        img = colorize(np.where(amap >= 0, amap, -1), M)
        img[amap < 0] = (np.array([0.55, 0.55, 0.55]) * 255).astype(np.uint8)
        if a.get("arch") == "m2f":
            img[attn.max(1).reshape(C.GH, C.GW) < 0.5] = 0
        out["images"]["pred_masks"] = png(np.repeat(np.repeat(img, C.P, 0), C.P, 1))
        geo = set(a.get("losses", "").split(",")) & {"sdf", "pair_cd", "pair_iou", "obs_cd", "free", "overlap", "render", "part_cd"}
        shown = np.flatnonzero((z["exist"] > 0.5) | (slot2t >= 0))  # predicted (exist > 0.5) and all matched slots
        sig_t = torch.as_tensor(sig, device=C.DEV)
        parts = "p_scale" in z.files
        if parts:  # --parts: each slot's active parts (part exist > 0.5; its best part if none)
            Pk = {k: torch.as_tensor(z[f"p_{k}"], device=C.DEV).float() for k in C.KEYS}  # (K, G, ...)
            pex = torch.as_tensor(z["p_exist"], device=C.DEV).float()
            pact = pex > 0.5
            pact[torch.arange(K, device=C.DEV), pex.argmax(1)] |= ~pact.any(1)
            sh = torch.zeros(K, dtype=torch.bool, device=C.DEV)
            sh[torch.as_tensor(shown, device=C.DEV)] = True
            psel = pact & sh[:, None]
            ks, gs = torch.nonzero(psel, as_tuple=True)
            flatp = {k: v[ks, gs] for k, v in Pk.items()}
            if geo:
                out["images"]["pred_render"] = png(render_labels(flatp, slot2t[ks.cpu().numpy()], M))
            Pm, am_t = {k: v[sig_t] for k, v in Pk.items()}, pact[sig_t]
            lab_ = fr["pts_lab"]
            dd = C.sdf(fr["pts"][:, None], {k: v[lab_] for k, v in Pm.items()})[:, :, 0].abs()
            own_p = torch.where(am_t[lab_], dd, torch.inf).min(1).values
            out["parts"] = [sq_json(flatp, j) | {"exist": float(z["exist"][ks[j]] * pex[ks[j], gs[j]]), "t": int(slot2t[ks[j]]),
                                                  "slot": int(ks[j]), "part": int(gs[j])} for j in range(len(ks))]
        elif geo:  # rendered SQs only for runs that train geometry
            out["images"]["pred_render"] = png(render_labels({k: x[torch.as_tensor(shown, device=C.DEV)] for k, x in ps.items()},
                                                             slot2t[shown], M))
        ms = {k: x[sig_t] for k, x in ps.items()}
        if not parts:
            own_p = C.sdf_pairs(fr["pts"][:, None], {k: x[fr["pts_lab"]] for k, x in ms.items()})[:, 0].abs()
        perr = np.full(lab.size, np.nan, np.float32)
        perr[m] = own_p.cpu().numpy()
        if geo:
            out["images"]["pred_fit"] = png(heat(perr.reshape(lab.shape), 0.1))
        pts["slot"] = slot2t[am[patch]].tolist()
        pts["perr"] = np.nan_to_num(perr[sel], nan=-1).round(4).tolist()
        if parts:  # the union of the slot's active parts against the target union
            piou, piou_fr = (v.cpu().numpy() for v in group_pair_iou({"sq": Pm, "valid": am_t}, gr))
            G = am_t.shape[1]
            pg = C.surf({k: v.flatten(0, 1) for k, v in Pm.items()}, 16, "cube").reshape(M, -1, 3)
            pm = am_t[:, :, None].expand(-1, -1, pg.shape[1] // G).reshape(M, -1)
            pcd = weighted_chamfer(torch.where(pm[..., None], pg, 1e3), torch.where(gr["keep"][..., None], gr["pts"], 1e3),
                                   pm.float(), gr["keep"].float()).cpu().numpy()
        else:
            pg = C.surf(ms, 16, "grid")
            piou, piou_fr = (v.cpu().numpy() for v in pair_iou_mc(ms, gr))
            pcd = weighted_chamfer(pg, torch.where(gr["keep"][..., None], gr["pts"], 1e3), torch.ones(pg.shape[:2], device=C.DEV),
                                   gr["keep"].float()).cpu().numpy()
        share = fr["share"].cpu().numpy()
        valid = fr["pvalid"].cpu().numpy()
        maj = np.where(valid, share.argmax(1), -1)
        for i in range(M):
            A_, B_ = am == sig[i], maj == i
            tinfo[i] |= {"slot": int(sig[i]), "exist": float(z["exist"][sig[i]]), "pair_iou": None if np.isnan(piou[i]) else float(piou[i]), "pair_iou_fr": None if np.isnan(piou_fr[i]) else float(piou_fr[i]),
                         "pair_cd": float(pcd[i]), "mask_iou": float((A_ & B_).sum() / max((A_ | B_).sum(), 1)),
                         "pred_med": float(own_p[fr["pts_lab"] == i].median()), "patches": int(A_.sum()),
                         "n_parts": int(am_t[i].sum()) if parts else None}
        anc = z["anchor"] if "anchor" in z.files else None
        out["pred"] = [sq_json(ps, k) | {"exist": float(z["exist"][k]), "t": int(slot2t[k]), "patches": int((am == k).sum()),
                                         "anchor": anc[k].tolist() if anc is not None else None} for k in range(K)]
    out["targets"] = tinfo
    out["points"] = pts
    return out


@app.get("/api/mask")
@torch.no_grad()
def mask(run: str, step: str = "final", t: int = -1, slot: int = -1, source: str = ""):
    """GT patch share of target t and the attention map of a slot (t's matched slot unless given) -> (GH, GW) each."""
    a = run_args(run)
    if run == "data" and source:
        a["mask_source"] = source
    fr = frame(a["frame"], a["mask_source"], a["distance"])
    out = {"gh": C.GH, "gw": C.GW}
    if t >= 0:
        out["gt"] = fr["share"][:, t].cpu().numpy().round(3).tolist()
    z = pred_file(run, step)
    if z is not None:
        k = slot if slot >= 0 else (int(z["sig"][t]) if t >= 0 else -1)
        if k >= 0:
            out["slot"], out["attn"] = k, z["attn"][:, k].astype(np.float32).round(3).tolist()
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8027)
