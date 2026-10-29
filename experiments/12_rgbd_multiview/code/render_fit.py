"""Depth-free fitting from multi-view instance masks: differentiable rendering of the SQ occupancies.

    python render_fit.py --exp 13 --name <run> [--set key=value ...]      (keys: CFG)

One SQ per instance. Every pixel ray o + t r (t = z-depth, N stratified samples in [T_NEAR, T_FAR]) is composited
like a surface rendering: occupancy of SQ m at a sample o_m = sigmoid(-d_m / tau), alpha = 1 - prod_m (1 - o_m),
weights w_i = alpha_i prod_{j<i} (1 - alpha_j). The pixel's instance distribution is p_k = sum_i w_i o_k / sum_m o_m
(+ "nothing" = remaining transmittance). Loss: cross-entropy with the pixel label (unlabelled pixels: only
-log(1 - p_nothing)), instance-balanced pixel sampling. tau is annealed from TAU0 to TAU1.
Init without depth: the center of each instance = least-squares intersection of the rays through its mask centroids
(views with >= 20 pixels), half extents = mean over views of sqrt(mask area) / f * distance / 2.
Optional `w_depth` > 0: + L1 between rendered (expected) depth and the given depth (RGBD variant, for comparison).
Optional `w_vol` > 0: + w_vol * mean SQ size (cube root of the volume of its box, m): silhouettes do not constrain
the never-seen sides (back of walls, underside of the floor), the smallest shape consistent with all masks is
preferred (as in shape-from-silhouette).
Optional `init` = <exp>:<run>: start from that run's SQs instead of the triangulation (refinement of a depth-based fit
with the sharp mask boundaries; instances without SQ there are dropped).
"""

import argparse
import json
import time

import numpy as np
import torch

from fit import EXP_DIRS, EXPERIMENTS, evaluate, parse_frames, render, summarize
from grid import gt_grid
from sq import SQs, optimize, sdf
from views import DEV, gt, intrinsics, load_views

CFG = dict(scene="room_0", frames="stride20", steps=3000, rays=4096, samples=128, t_near=0.1, t_far=8.0,
           tau0=0.2, tau1=0.01, lr=0.01, w_depth=0.0, w_vol=0.0, min_px=20, res=0.04, init="")


def rays_of(views):
    """Per view: camera center (3,), unit-depth ray directions (h*w, 3), labels (h*w,), depth (h*w,)."""
    fx, fy, cx, cy = intrinsics()
    out = []
    for v in views:
        h, w = v["label"].shape
        j, i = np.indices((h, w))
        d = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i, dtype=float)], -1).reshape(-1, 3)
        out.append((v["c2w"][:3, 3], d @ v["c2w"][:3, :3].T, v["label"].reshape(-1), v["depth"].reshape(-1)))
    return out


def triangulate(rays, n_inst, min_px):
    """Instance centers from the rays through the mask centroids, and a size estimate (no depth used)."""
    fx = intrinsics()[0]
    centers, halves, keep = [], [], []
    for k in range(n_inst):
        A, b, obs = np.zeros((3, 3)), np.zeros(3), []
        for o, d, lab, _ in rays:
            m = lab == k
            if m.sum() < min_px:
                continue
            r = d[m].mean(0)
            r = r / np.linalg.norm(r)
            P = np.eye(3) - np.outer(r, r)  # distance to the ray: |P (x - o)|
            A += P
            b += P @ o
            obs.append((o, r, m.sum()))
        if len(obs) < 2 or np.linalg.cond(A) > 1e6:
            continue
        c = np.linalg.solve(A, b)
        size = np.mean([np.sqrt(n) / fx * np.linalg.norm(c - o) / 2 for o, _, n in obs])
        centers.append(c), halves.append(np.full(3, size)), keep.append(k)
    return np.array(centers), np.array(halves), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = dict(CFG)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = type(CFG[k])(v) if not isinstance(CFG[k], str) else v
    t0 = time.time()
    torch.manual_seed(0)
    scene, n_inst = cfg["scene"], len(gt(cfg["scene"])["ids"])
    views = load_views(scene, parse_frames(cfg["frames"]))
    rays = rays_of(views)
    centers, halves, inst = triangulate(rays, n_inst, cfg["min_px"])
    # SQs initialised from the corners of the init boxes (axis-aligned, PCA of 8 corners)
    sets = [c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            for c, h in zip(centers, halves)]
    model = SQs(sets, inst)
    if cfg["init"]:
        exp, run = cfg["init"].split(":")
        z = np.load(EXPERIMENTS / EXP_DIRS[int(exp)] / "results" / "runs" / run / "sqs.npz")
        inst = [int(k) for k in z["instance"]]
        model = SQs([np.zeros((8, 3)) + z["trans"][i] + np.eye(3)[:1] * 0.01 * np.arange(8)[:, None] for i in range(len(inst))], inst)
        with torch.no_grad():
            t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)
            model.trans.copy_(t(z["trans"])), model.log_scale.copy_(t(z["scale"]).log())
            model.raw_eps.copy_(torch.logit((t(z["eps"]) - 0.1) / 1.8, eps=1e-4))
            model.axes.copy_(t(z["rot"])), model.quat.copy_(torch.tensor([1.0, 0, 0, 0], device=DEV).repeat(len(inst), 1))
    sq_inst = torch.as_tensor(inst, device=DEV)
    lut = torch.full((n_inst + 1,), -1, device=DEV)  # GT instance -> column of the fitted SQs (-1: none)
    lut[sq_inst] = torch.arange(len(inst), device=DEV)
    O = torch.tensor(np.stack([np.broadcast_to(o, d.shape) for o, d, _, _ in rays]).reshape(-1, 3), dtype=torch.float32, device=DEV)
    D = torch.tensor(np.concatenate([d for _, d, _, _ in rays]), dtype=torch.float32, device=DEV)
    LAB = torch.tensor(np.concatenate([lab for _, _, lab, _ in rays]), device=DEV)
    DEP = torch.tensor(np.concatenate([z for _, _, _, z in rays]), dtype=torch.float32, device=DEV)
    col = lut[LAB.clamp_min(-1)]  # target column per pixel, -1: unlabelled or instance without SQ
    # instance-balanced sampling: half the rays uniform over labelled pixels of a random instance, half uniform
    by_inst = [torch.nonzero(col == m)[:, 0] for m in range(len(inst))]
    N, R = cfg["samples"], cfg["rays"]
    t_lin = torch.linspace(cfg["t_near"], cfg["t_far"], N + 1, device=DEV)

    def loss(step):
        tau = cfg["tau0"] * (cfg["tau1"] / cfg["tau0"]) ** min(1.0, step / (0.7 * cfg["steps"]))
        m = torch.randint(len(inst), (R // 2,), device=DEV)
        pick = torch.stack([by_inst[i][torch.randint(len(by_inst[i]), (1,), device=DEV)] for i in m.tolist()])[:, 0]
        idx = torch.cat([pick, torch.randint(len(O), (R - R // 2,), device=DEV)])
        t = t_lin[:-1] + torch.rand(len(idx), N, device=DEV) * (t_lin[1] - t_lin[0])  # stratified
        x = O[idx, None] + t[..., None] * D[idx, None]  # (R, N, 3)
        occ = torch.sigmoid(-sdf(x[:, :, None], model.params()) / tau)  # (R, N, M)
        alpha = 1 - torch.exp(torch.log1p(-occ.clamp(max=1 - 1e-6)).sum(-1))
        trans = torch.exp(torch.cumsum(torch.log1p(-alpha.clamp(max=1 - 1e-6)), 1))
        w = alpha * torch.cat([torch.ones_like(trans[:, :1]), trans[:, :-1]], 1)  # (R, N)
        p = (w[..., None] * occ / occ.sum(-1, keepdim=True).clamp_min(1e-8)).sum(1)  # (R, M)
        nothing = trans[:, -1]
        c = col[idx]
        lab_ok = c >= 0
        ce = -torch.log(p[lab_ok, c[lab_ok]].clamp_min(1e-6)).mean()
        hit = -torch.log((1 - nothing[~lab_ok]).clamp_min(1e-6)).mean() if (~lab_ok).any() else 0
        out = ce + 0.1 * hit
        if cfg["w_depth"]:
            z = (w * t).sum(1) + nothing * cfg["t_far"]
            ok = DEP[idx] > 0
            out = out + cfg["w_depth"] * (z[ok] - DEP[idx][ok]).abs().mean()
        if cfg["w_vol"]:
            out = out + cfg["w_vol"] * model.log_scale.exp().prod(1).pow(1 / 3).mean()
        return out

    optimize(model, loss, cfg["steps"], lr=cfg["lr"])
    sqs = model.numpy()
    ref = gt_grid(scene, cfg["res"])
    rows = evaluate(sqs, ref, n_inst)
    summ = summarize(rows, gt(scene)["names"]) | {"seconds": round(time.time() - t0, 1), "views": len(views)}
    out = EXPERIMENTS / EXP_DIRS[a.exp] / "results" / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "sqs.npz", **sqs)
    (out / "config.json").write_text(json.dumps(cfg | {"method": "render"}, indent=1))
    (out / "metrics.json").write_text(json.dumps({"summary": summ, "per_instance": rows, "names": gt(scene)["names"]}, indent=1))
    render(sqs, scene, out, views)
    print(a.name, json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
