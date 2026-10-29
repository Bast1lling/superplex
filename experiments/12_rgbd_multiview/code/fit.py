"""Fit one SQ per instance to a label grid (GT 3D, or carved from views) and evaluate it against the full 3D GT.

    python fit.py --exp 12 --name <run> [--set key=value ...]      (keys: CFG)

Loss per instance k (the presentation's instances loss): positives = the voxels of k (GT: solid, carved: solid
votes); negatives = uniform samples in k's box + MARGIN, weighted by their label: free 1, unknown w_unk, other
instances 0 (amodal overlap is free). L_k = mean softplus(d/tau) over positives + sum w softplus(-d/tau) / #free.

Evaluation (evaluate.py-style, per GT instance, 4 cm GT grid from the full point cloud, the space around the grid
counted as free): TP = GT solid voxels of k inside the union of k's SQs, FP = free voxels inside it,
IoU = TP / (|GT_k| + FP). Instances without SQ (not seen) count as IoU 0 in `viou_all`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from grid import FREE, UNKNOWN, carve_grid, gt_grid
from sq import SQs, optimize, sdf
from views import DEV, gt, load_views

EXPERIMENTS = Path(__file__).resolve().parents[2]
EXP_DIRS = {12: "12_rgbd_multiview", 13: "13_rgb_multiview", 14: "14_rgbd_singleview", 15: "15_rgb_singleview"}
CFG = dict(scene="room_0", source="views", depth="gt", frames="stride20", w_unk=0.1, res=0.04, steps=2000,
           tau=0.02, margin=0.2, min_vox=4, thick=0.1, outside_free=1, align=0, erode=0, min_votes=1, rel_band=0.0)


def parse_frames(spec):
    """stride20 -> every 20th frame; 500 or 500,1200 -> those frames."""
    if spec.startswith("stride"):
        return list(range(0, 2000, int(spec[6:])))
    return [int(f) for f in str(spec).split(",")]


def view_depths(views, cfg):
    if cfg["depth"] == "gt":
        return [v["depth"] for v in views]
    import depth  # monocular depth networks (experiments 13 / 15)
    name, oracle = cfg["depth"].removesuffix("_gtscale"), cfg["depth"].endswith("_gtscale")
    pred = [depth.predict(cfg["scene"], v, name) for v in views]
    if oracle:  # control: each view rescaled by its true median scale (uses GT depth, not a valid input)
        pred = [p * np.median(v["depth"][v["depth"] > 0] / p[v["depth"] > 0]) for v, p in zip(views, pred)]
    return depth.align_scales(views, pred) if cfg["align"] else pred


def fit(grid, cfg):
    n_inst = len(gt(cfg["scene"])["ids"])
    sets = {k: np.argwhere(grid.label_np == k) for k in range(n_inst)}
    sets = {k: s for k, s in sets.items() if len(s) >= cfg["min_vox"]}
    x = [grid.centers(s) for s in sets.values()]
    vox, count = torch.cat(x), torch.tensor([len(v) for v in x], device=DEV)
    start = torch.cumsum(count, 0) - count
    lo = torch.stack([v.min(0).values for v in x]) - cfg["margin"]
    hi = torch.stack([v.max(0).values for v in x]) + cfg["margin"]
    inst = torch.tensor(list(sets), device=DEV)
    model = SQs([v.cpu().numpy() for v in x], list(sets), pad=grid.res / 2)
    T, tau = len(sets), cfg["tau"]

    def loss(step):
        p = {k: v[:, None] for k, v in model.params().items()}
        pos = vox[start[:, None] + (torch.rand(T, 256, device=DEV) * count[:, None]).long()]
        neg = lo[:, None] + torch.rand(T, 256, 3, device=DEV) * (hi - lo)[:, None]
        lab = grid.lookup(neg)
        w = (lab == FREE) + cfg["w_unk"] * (lab == UNKNOWN)
        inside = F.softplus(sdf(pos, p) / tau).mean(1)
        outside = (w * F.softplus(-sdf(neg, p) / tau)).sum(1) / (lab == FREE).sum(1).clamp_min(1)
        return (inside + outside).mean()

    optimize(model, loss, cfg["steps"])
    return model.numpy()


@torch.no_grad()
def evaluate(sqs, ref, n_inst, pad=12, chunk=1 << 15):
    L = np.pad(ref.label_np, pad, constant_values=FREE)
    lo = torch.as_tensor(ref.lo_np - pad * ref.res, device=DEV)
    p = {k: torch.as_tensor(sqs[k], device=DEV) for k in ("trans", "scale", "eps", "rot")}
    inst = torch.as_tensor(sqs["instance"], device=DEV)
    onehot = F.one_hot(inst, n_inst).float()
    free = torch.as_tensor(np.argwhere(L == FREE), device=DEV)
    fp = torch.zeros(n_inst, device=DEV)
    for a in range(0, len(free), chunk):
        x = lo + free[a:a + chunk].float() * ref.res
        fp += (((sdf(x[:, None], p) < 0).float() @ onehot) > 0).sum(0)
    rows = []
    for k in range(n_inst):
        g = torch.as_tensor(np.argwhere(L == k), device=DEV)
        mine = (inst == k).nonzero()[:, 0]
        tp = 0
        if len(mine) and len(g):
            pk = {key: v[mine] for key, v in p.items()}
            tp = int(sum((sdf((lo + g[a:a + chunk].float() * ref.res)[:, None], pk).min(1).values < 0).sum().item()
                         for a in range(0, len(g), chunk)))
        n, f = len(g), float(fp[k])
        rows.append({"gt_vox": n, "fitted": bool(len(mine)), "tp": tp, "fp": f, "iou": tp / max(n + f, 1),
                     "precision": tp / max(tp + f, 1), "recall": tp / max(n, 1)})
    return rows


def summarize(rows, names):
    fit_rows = [r for r in rows if r["fitted"] and r["gt_vox"] > 0]
    big = [r for r, nm in zip(rows, names) if r["fitted"] and nm in ("wall", "floor", "ceiling", "rug")]
    small = [r for r, nm in zip(rows, names) if r["fitted"] and nm not in ("wall", "floor", "ceiling", "rug")]
    m = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else float("nan")
    return {"instances": len(rows), "fitted": len(fit_rows), "viou": m(fit_rows, "iou"),
            "precision": m(fit_rows, "precision"), "recall": m(fit_rows, "recall"),
            "viou_all": float(np.mean([r["iou"] if r["fitted"] else 0 for r in rows if r["gt_vox"] > 0])),
            "viou_structure": m(big, "iou"), "viou_objects": m(small, "iou")}


def render(sqs, scene, out, views=None):
    sys.path.insert(0, str(EXPERIMENTS / "08_volumetric_decomp" / "code"))
    from render import camera, palette, save_grid, splat, sqs_points, view_bounds
    g = gt(scene)
    keep = np.array([n != "ceiling" for n in g["names"]])
    pm = (g["inst"] >= 0) & keep[np.maximum(g["inst"], 0)]
    P, Y, N = g["pos"][pm][::6], g["inst"][pm][::6], g["nrm"][pm][::6]
    pal = palette(len(g["ids"]))
    sm = keep[sqs["instance"]]
    sub = {k: v[sm] for k, v in sqs.items()}
    sp, sn, sc = sqs_points(sub, pal[sub["instance"]])
    imgs, titles = [], []
    if views:
        for v in views[:1]:
            imgs.append((v["rgb"] * 255).astype(np.uint8)), titles.append(f"input frame {v['frame']}")
            lab = np.where(v["label"][..., None] >= 0, pal[np.maximum(v["label"], 0)], 0)
            imgs.append((lab * 255).astype(np.uint8)), titles.append("input labels")
    for elev, azim in ((55, -60), (89.9, -90)):
        R = camera(elev, azim)
        bnd = view_bounds(P, R)
        imgs.append(splat(P, pal[Y], N, R, bnd, 700, radius=2)), titles.append(f"GT points, elev {elev:.0f}")
        imgs.append(splat(sp, sc, sn, R, bnd, 700, radius=2)), titles.append(f"{len(sub['trans'])} SQs (no ceiling)")
    save_grid(imgs, titles, out / "render.jpg", cols=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    cfg = dict(CFG)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = type(CFG[k])(v) if k in CFG and not isinstance(CFG[k], str) else v
    t0 = time.time()
    torch.manual_seed(0)
    scene, n_inst = cfg["scene"], len(gt(cfg["scene"])["ids"])
    ref = gt_grid(scene, cfg["res"])
    views = None
    if cfg["source"] == "gt3d":
        grid = ref
    else:
        views = load_views(scene, parse_frames(cfg["frames"]))
        grid = carve_grid(views, view_depths(views, cfg), n_inst, cfg["res"], thick=cfg["thick"],
                          outside_free=bool(cfg["outside_free"]), erode=cfg["erode"], min_votes=cfg["min_votes"],
                          rel_band=cfg["rel_band"])
    sqs = fit(grid, cfg)
    rows = evaluate(sqs, ref, n_inst)
    summ = summarize(rows, gt(scene)["names"]) | {"seconds": round(time.time() - t0, 1),
                                                  "views": 0 if views is None else len(views)}
    out = EXPERIMENTS / EXP_DIRS[a.exp] / "results" / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "sqs.npz", **sqs)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "metrics.json").write_text(json.dumps({"summary": summ, "per_instance": rows,
                                                  "names": gt(scene)["names"]}, indent=1))
    if not a.no_render:
        render(sqs, scene, out, views)
    print(a.name, json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
