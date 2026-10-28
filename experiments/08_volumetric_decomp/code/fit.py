"""Volumetric per-instance SQ decomposition with greedy model selection.

    python fit.py --name room0_base [--set key=value ...]

Per instance k (GT instance labels), target = the instance's solid voxels T_k (volume.py), negatives = free voxels
and other instances' solid voxels in the instance's box + margin (unknown voxels are ignored). The union of the
instance's SQs is fitted with a logistic occupancy loss on the union SDF d = min_j d_j:
    L_k = [ sum_T softplus(d / tau) + sum_N w softplus(-d / tau) ] / |T_k|
(linear in the distance far from the boundary, so targets far outside / negatives deep inside always pull).

Structure (greedy, all instances in parallel): start with 1 SQ (PCA box of T_k). Each round proposes one change per
instance: *add* an SQ on the largest connected component of uncovered target voxels, or *split* the SQ that covers
the most negatives (2-means of its target voxels). After `steps_round` joint steps the change is kept only if the
instance's hard volumetric IoU rose by more than `delta`; otherwise the instance is restored. Finally SQs whose
removal costs less than `prune_delta` IoU are pruned (after merge rounds: the most box-like pair of SQs of an instance is
replaced by one box and refitted; kept under the same rule). delta is the price of one primitive in IoU (MDL-style);
`min_gain_vox` additionally requires a primitive to explain a minimum absolute volume (IoU gain x |T_k| voxels),
which stops tiny objects (a few voxels, where one voxel is a large IoU change) from being split.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from sklearn.cluster import KMeans

from volume import EXTERIOR, FREE, UNKNOWN, build_volume, load_scene, perturb
from sq import SQSet, sdf_slots, sdf_radial

EXP = Path(__file__).resolve().parents[1]
RUNS = EXP / "results" / "runs"

CFG = dict(
    scene="room_0", labels="instance", res=0.04, min_points=20,      # instances with fewer (2 cm) points are dropped
    K=8, margin=0.3, max_neg=200000,
    w_free=1.0, w_other=0.1, w_unknown=0.03, w_exterior=1.0, w_region=10.0,  # metrics: reeval.py uses 0.3 / 0.1
    samples_pos=512, samples_neg=1024, tau=0.02, tau_final=0.01,
    lr=0.01, lr_decay=True, steps_init=400, steps_round=250, rounds=8, steps_final=300,
    delta=0.03, prune_delta=0.03, max_fail=3, merge_rounds=2, prune_rounds=3, min_gain_vox=15.0, fill_min=0.1, min_comp=4,
    fill_planar=True, planar_thick=0.2, planar_min=0.5,
    noise=0.0, normal_deg=0.0, drop=0.0, holes=0.0,   # scan-like degradation of the input (evaluated on clean)
    eps_init=0.5, eps_max=1.2, steps_warm=100, seed=0, only="", verbose=False,
)


def parse_set(cfg, items):
    for it in items:
        k, v = it.split("=", 1)
        cfg[k] = type(cfg[k])(v) if k in cfg and not isinstance(cfg[k], bool) else (v.lower() == "true" if k in cfg else v)
    return cfg


# ---------------------------------------------------------------- data

def cached_volume(cfg, sc, tag):
    cache = Path("/tmp/claude-0/volcache") / f"{cfg['scene']}_{cfg['res']}_{cfg['min_points']}_{cfg['labels']}_{tag}.npz"
    if cache.exists():
        vol = dict(np.load(cache))
        vol["res"] = float(vol["res"])
        return vol
    vol = build_volume(sc, res=cfg["res"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **vol)
    return vol


def fill_planar(vox, res, thick, min_size):
    """Amodal completion of thin planar instances (floor, walls, windows, rugs, pictures): if the PCA box of the
    voxels is thinner than `thick` and larger than `min_size`, the 2D holes of its projection onto its plane (floor
    under furniture, unscanned window glass, wall behind a picture) are filled; the new cells get the instance's
    5-95% thickness range. Holes = cells inside the 2D convex hull of the projection (plain hole filling misses holes
that touch the border, e.g. floor under a sofa standing at the wall, or open window frames). Returns the extra voxel indices (may overlap other instances: amodal targets)."""
    if len(vox) < 20:
        return np.zeros((0, 3), int)
    x = vox.astype(float)
    mu = x.mean(0)
    _, vec = np.linalg.eigh((x - mu).T @ (x - mu))
    q = (x - mu) @ vec  # columns ascending: q[:, 0] is the thin axis
    ext = (q.max(0) - q.min(0)) * res
    if ext[0] > thick or ext[2] < min_size:
        return np.zeros((0, 3), int)
    uv = np.round(q[:, 1:] - q[:, 1:].min(0)).astype(int)
    grid = np.zeros(uv.max(0) + 1, bool)
    grid[uv[:, 0], uv[:, 1]] = True
    from scipy.spatial import Delaunay
    try:
        hull = Delaunay(np.unique(uv, axis=0))
    except Exception:  # degenerate (collinear) projection
        return np.zeros((0, 3), int)
    allc = np.argwhere(np.ones_like(grid))
    filled = (hull.find_simplex(allc) >= 0).reshape(grid.shape) & ~grid
    cells = np.argwhere(filled)
    if len(cells) == 0:
        return np.zeros((0, 3), int)
    w = np.arange(np.floor(np.quantile(q[:, 0], 0.05)), np.ceil(np.quantile(q[:, 0], 0.95)) + 1)
    c = cells + q[:, 1:].min(0)
    pts = np.concatenate([np.column_stack([np.full(len(c), wi), c]) for wi in w])
    new = np.round(pts @ vec.T + mu).astype(int)
    return np.unique(new, axis=0)


def prepare(cfg, sc, vol):
    """Flat target / negative voxel arrays per instance."""
    L, lo, res, dims = vol["label"], vol["lo"], vol["res"], vol["dims"]
    rng = np.random.default_rng(cfg["seed"])
    sel_ids = [int(v) for v in cfg["only"].split(",")] if cfg["only"] else None
    ks = [k for k in range(len(sc["instance_ids"])) if sel_ids is None or sc["instance_ids"][k] in sel_ids]
    I = len(ks)
    pos_x, pos_i, pos_v, neg_x, neg_i, neg_w = [], [], [], [], [], []
    region_lo, region_hi = [], []
    n_pos, n_neg = np.zeros(I, int), np.zeros(I, int)
    flatL = L.reshape(-1)
    for ii, k in enumerate(ks):
        vox = np.argwhere(L == k)
        if len(vox) == 0:  # fully overwritten: use the voxels of its points
            vox = np.unique(np.clip(np.round((sc["pos"][sc["y"] == k] - lo) / res).astype(int), 0, dims - 1), axis=0)
        if cfg["fill_planar"]:
            extra = fill_planar(vox, res, cfg["planar_thick"], cfg["planar_min"])
            extra = extra[((extra >= 0) & (extra < dims)).all(1)]
            if len(extra):
                vox = np.unique(np.concatenate([vox, extra]), axis=0)
        pos_x.append(lo + vox * res), pos_i.append(np.full(len(vox), ii)), pos_v.append(vox)
        n_pos[ii] = len(vox)
        m = int(np.ceil(cfg["margin"] / res))
        a, b = np.maximum(vox.min(0) - m, 0), np.minimum(vox.max(0) + m + 1, dims)
        sub = L[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
        region_lo.append((lo + a * res)[None]), region_hi.append((lo + (b - 1) * res)[None])
        own = np.zeros(sub.shape, bool)
        own[tuple((vox - a).T)] = True
        neg = np.argwhere((sub != k) & ~own & ((sub != UNKNOWN) | (cfg["w_unknown"] > 0))
                          & ((sub != EXTERIOR) | (cfg["w_exterior"] > 0))) + a
        lab = sub[tuple((neg - a).T)]
        w = np.select([lab == FREE, lab == UNKNOWN, lab == EXTERIOR],
                      [cfg["w_free"], cfg["w_unknown"], cfg["w_exterior"]], cfg["w_other"]).astype(np.float32)
        if len(neg) > cfg["max_neg"]:
            sel = rng.choice(len(neg), cfg["max_neg"], replace=False)
            w = w[sel] * len(neg) / cfg["max_neg"]
            neg = neg[sel]
        neg_x.append(lo + neg * res), neg_i.append(np.full(len(neg), ii)), neg_w.append(w)
        n_neg[ii] = len(neg)
    f = lambda a, dt=np.float32: torch.from_numpy(np.concatenate(a).astype(dt))
    return {"pos_x": f(pos_x), "pos_i": f(pos_i, np.int64), "pos_v": np.concatenate(pos_v),
            "neg_x": f(neg_x), "neg_i": f(neg_i, np.int64), "neg_w": f(neg_w),
            "region_lo": f(region_lo), "region_hi": f(region_hi),
            "n_pos": torch.from_numpy(n_pos), "n_neg": torch.from_numpy(n_neg),
            "off_pos": torch.from_numpy(np.concatenate([[0], np.cumsum(n_pos)[:-1]])),
            "off_neg": torch.from_numpy(np.concatenate([[0], np.cumsum(n_neg)[:-1]])), "I": I, "ks": ks}


def sample(D, n, which, g):
    I = D["I"]
    cnt, off = D[f"n_{which}"], D[f"off_{which}"]
    r = torch.rand(I, n, generator=g)
    idx = (off[:, None] + (r * cnt[:, None]).long().clamp(max=(cnt[:, None] - 1).clamp(min=0))).reshape(-1)
    w = (cnt.float() / n)[:, None].expand(I, n).reshape(-1)
    valid = (cnt > 0)[:, None].expand(I, n).reshape(-1)
    idx, w = idx[valid], w[valid]
    if which == "neg":
        w = w * D["neg_w"][idx]
    return D[f"{which}_x"][idx], D[f"{which}_i"][idx], w


# ---------------------------------------------------------------- loss / eval

def union_sdf(model, par, x, inst):
    d = sdf_slots(par, x, inst)                                  # (S, K)
    return torch.where(model.active[inst], d, torch.full_like(d, 1e3))


def loss_fn(cfg, model, D, g, tau):
    par = model.params()
    xp, ip, wp = sample(D, cfg["samples_pos"], "pos", g)
    xn, i_n, wn = sample(D, cfg["samples_neg"], "neg", g)
    dp = union_sdf(model, par, xp, ip).amin(1)
    dn = union_sdf(model, par, xn, i_n).amin(1)
    per = torch.zeros(D["I"]).index_add(0, ip, wp * torch.nn.functional.softplus(dp / tau))
    per = per.index_add(0, i_n, wn * torch.nn.functional.softplus(-dn / tau))
    per = per / D["n_pos"].clamp(min=1)
    # keep every SQ's axis-aligned extent inside its instance's region (no negatives are sampled beyond it)
    ext = (par["rot"].abs() * par["scale"][..., None, :]).sum(-1)              # (I, K, 3)
    out = torch.relu(par["trans"] + ext - D["region_hi"][:, None]) + torch.relu(D["region_lo"][:, None] - par["trans"] + ext)
    reg = (out.sum(-1) * model.active).sum(1)
    return per.mean() + cfg["w_region"] * reg.mean(), per.detach()


@torch.no_grad()
def evaluate(model, D, chunk=200000):
    """Hard volumetric stats per instance; also per-slot coverage for proposals / pruning."""
    par = model.params()
    I, K = model.I, model.K
    out = {}
    for which in ("pos", "neg"):
        x, inst = D[f"{which}_x"], D[f"{which}_i"]
        ds = [union_sdf(model, par, x[s:s + chunk], inst[s:s + chunk]) for s in range(0, len(x), chunk)]
        out[which] = torch.cat(ds) if ds else torch.zeros(0, K)
    dp, dn = out["pos"], out["neg"]
    wn = D["neg_w"]
    tp = torch.zeros(I).index_add(0, D["pos_i"], (dp.amin(1) < 0).float())
    fp = torch.zeros(I).index_add(0, D["neg_i"], wn * (dn.amin(1) < 0).float())
    fn = D["n_pos"].float() - tp
    iou = tp / (tp + fn + fp).clamp(min=1)
    # IoU without slot j (pruning)
    iou_wo = torch.zeros(I, K)
    for j in range(K):
        m = torch.ones(K, dtype=torch.bool)
        m[j] = False
        tpj = torch.zeros(I).index_add(0, D["pos_i"], (dp[:, m].amin(1) < 0).float())
        fpj = torch.zeros(I).index_add(0, D["neg_i"], wn * (dn[:, m].amin(1) < 0).float())
        iou_wo[:, j] = tpj / (D["n_pos"].float() + fpj).clamp(min=1)
    # negatives inside each slot (split proposals)
    slot_fp = torch.zeros(I * K).index_add(0, (D["neg_i"][:, None] * K + torch.arange(K)).reshape(-1),
                                           (wn[:, None] * (dn < 0).float()).reshape(-1)).view(I, K)
    return {"iou": iou, "tp": tp, "fp": fp, "fn": fn, "iou_wo": iou_wo, "slot_fp": slot_fp, "dp": dp}


# ---------------------------------------------------------------- proposals

def box_fit(x, res):
    mu = x.mean(0)
    c = x - mu
    if len(x) < 4:
        return mu, np.eye(3), np.full(3, res)
    _, vec = np.linalg.eigh(c.T @ c)
    R = vec[:, ::-1].copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    q = c @ R
    lo, hi = q.min(0), q.max(0)
    mu = mu + R @ ((lo + hi) / 2)
    return mu, R, (hi - lo) / 2 + res / 2


def compact_part(x, res, fill_min, rng, depth=0):
    """Largest part of x whose PCA box is filled to >= fill_min (recursive 2-means), so that ring- or L-shaped
    voxel sets (all walls of a room, an L-shaped wall) are not fitted by one room-sized box."""
    t, R, a = box_fit(x, res)
    fill = len(x) * res ** 3 / np.prod(2 * a)
    if fill >= fill_min or len(x) < 16 or depth >= 6:
        return x
    lab = KMeans(2, n_init=2, random_state=int(rng.integers(1 << 30))).fit_predict(x)
    parts = [compact_part(x[lab == c], res, fill_min, rng, depth + 1) for c in (0, 1)]
    return max(parts, key=len)


def largest_component(vox):
    a = vox.min(0)
    grid = np.zeros(vox.max(0) - a + 1, bool)
    grid[tuple((vox - a).T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if n == 0:
        return np.zeros(0, int)
    sizes = np.bincount(lab[tuple((vox - a).T)])
    sizes[0] = 0
    best = sizes.argmax()
    return np.nonzero(lab[tuple((vox - a).T)] == best)[0]


def propose(cfg, model, D, ev, i, rng, flip=False):
    """One structural proposal for instance i. Returns a description or None."""
    K, res = model.K, cfg["res"]
    free = np.nonzero(~model.active[i].numpy())[0]
    if len(free) == 0:
        return None
    sel = (D["pos_i"] == i).numpy()
    vox, x = D["pos_v"][sel], D["pos_x"].numpy()[sel]
    dp = ev["dp"][torch.from_numpy(sel)]
    act = model.active[i].numpy()
    uncovered = (dp.amin(1) >= 0).numpy()
    comp = largest_component(vox[uncovered]) if uncovered.sum() >= cfg["min_comp"] else np.zeros(0, int)
    fn_mass = len(comp)
    slot_fp = ev["slot_fp"][i].numpy() * act
    j = int(slot_fp.argmax())
    can_add = fn_mass >= cfg["min_comp"]
    prefer_add = fn_mass >= slot_fp[j]
    if flip:  # after a failed proposal try the other kind first
        prefer_add = not prefer_add
    if can_add and (prefer_add or slot_fp[j] <= 0):
        part = compact_part(x[uncovered][comp], res, cfg["fill_min"], rng)
        t, R, a = box_fit(part, res)
        return ("add", [(int(free[0]), t, R, a)])
    owned = (dp.argmin(1).numpy() == j) & (dp[:, j] < 0).numpy()
    if slot_fp[j] > 0 and owned.sum() >= 8:
        lab = KMeans(2, n_init=3, random_state=int(rng.integers(1 << 30))).fit_predict(x[owned])
        fits = [box_fit(x[owned][lab == c], res) for c in (0, 1)]
        return ("split", [(j, *fits[0]), (int(free[0]), *fits[1])])
    return None


def merge_proposal(cfg, model, D, ev, i):
    """Pair of active SQs of instance i whose owned voxels are best explained by one box: minimal
    vol(box(A u B)) / (vol(box(A)) + vol(box(B))). Returns (j_keep, j_drop, (t, R, a)) or None."""
    act = np.nonzero(model.active[i].numpy())[0]
    if len(act) < 2:
        return None
    sel = (D["pos_i"] == i).numpy()
    x = D["pos_x"].numpy()[sel]
    dp = ev["dp"][torch.from_numpy(sel)]
    own = dp.argmin(1).numpy()
    vol = lambda a: float(np.prod(2 * a))
    fits = {j: box_fit(x[own == j], cfg["res"]) for j in act if (own == j).sum() >= 4}
    best = None
    for ii, j in enumerate(act):
        for k in act[ii + 1:]:
            if j not in fits or k not in fits:
                continue
            u = box_fit(x[(own == j) | (own == k)], cfg["res"])
            r = vol(u[2]) / (vol(fits[j][2]) + vol(fits[k][2]))
            if best is None or r < best[0]:
                best = (r, j, k, u)
    return None if best is None else (best[1], best[2], best[3])


# ---------------------------------------------------------------- main loop

def optimize(cfg, model, D, steps, g, tau, lr, only_slots=None):
    """Adam on all SQs; with only_slots (I, K) bool, the other slots are frozen."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / steps)) if cfg["lr_decay"] else 1.0)
    for s in range(steps):
        loss, _ = loss_fn(cfg, model, D, g, tau)
        opt.zero_grad()
        loss.backward()
        if only_slots is not None:
            for p in model.parameters():
                p.grad[~only_slots] = 0
        opt.step()
        sched.step()
    return loss.item()


def snapshot_rows(model, rows):
    return {n: p.data[rows].clone() for n, p in model.named_parameters()} | {"active": model.active[rows].clone()}


def restore_rows(model, rows, snap):
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.data[rows] = snap[n]
        model.active[rows] = snap["active"]


def run(cfg, name):
    t0 = time.time()
    torch.manual_seed(cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    sc = load_scene(cfg["scene"], min_points=cfg["min_points"], labels=cfg["labels"])
    vol = cached_volume(cfg, sc, "clean2")
    D = prepare(cfg, sc, vol)
    D_eval = None
    degraded = any(cfg[k] > 0 for k in ("noise", "normal_deg", "drop", "holes"))
    if degraded:  # fit on the degraded input, report metrics on the clean volume
        D_eval = D
        sp = perturb(sc, cfg["noise"], cfg["normal_deg"], cfg["drop"], cfg["holes"], seed=cfg["seed"])
        tag = f"v2_n{cfg['noise']}_a{cfg['normal_deg']}_d{cfg['drop']}_h{cfg['holes']}"
        D = prepare(cfg, sp, cached_volume(cfg, sp, tag))
    I, K = D["I"], cfg["K"]
    print(f"{I} instances, {len(D['pos_x'])} target voxels, {len(D['neg_x'])} negatives, prep {time.time() - t0:.0f} s")
    model = SQSet(I, K, eps_range=(0.1, cfg["eps_max"]))
    for i in range(I):
        x = D["pos_x"][D["pos_i"] == i].numpy()
        t, R, a = box_fit(compact_part(x, cfg["res"], cfg["fill_min"], rng), cfg["res"])
        model.set_slot(i, 0, t, R, a, (cfg["eps_init"], cfg["eps_init"]))
    optimize(cfg, model, D, cfg["steps_init"], g, cfg["tau"], cfg["lr"])
    ev = evaluate(model, D)
    log = [{"round": 0, "iou": ev["iou"].mean().item(), "sqs": int(model.active.sum()), "t": time.time() - t0}]
    print(log[-1])
    fails = np.zeros(I, int)
    for r in range(1, cfg["rounds"] + 1):
        rows = torch.arange(I)
        snap = snapshot_rows(model, rows)
        iou_before = ev["iou"].clone()
        proposed = np.zeros(I, bool)
        kinds = {}
        new_slots = torch.zeros(I, K, dtype=torch.bool)
        for i in range(I):
            if fails[i] >= cfg["max_fail"]:
                continue
            p = propose(cfg, model, D, ev, i, rng, flip=bool(fails[i] % 2))
            if p is None:
                fails[i] = cfg["max_fail"]
                continue
            kinds[i] = p[0]
            for (j, t, R, a) in p[1]:
                model.set_slot(i, j, t, R, a, (cfg["eps_init"], cfg["eps_init"]))
                new_slots[i, j] = True
            proposed[i] = True
        if not proposed.any():
            break
        if cfg["steps_warm"]:
            optimize(cfg, model, D, cfg["steps_warm"], g, cfg["tau"], cfg["lr"], only_slots=new_slots)
        optimize(cfg, model, D, cfg["steps_round"], g, cfg["tau"], cfg["lr"])
        ev = evaluate(model, D)
        gain = (ev["iou"] - iou_before).numpy()
        npos = D["n_pos"].numpy()
        accept = proposed & (gain > cfg["delta"]) & (gain * npos > cfg["min_gain_vox"])
        reject = proposed & ~accept
        rej = torch.from_numpy(np.nonzero(reject)[0])
        restore_rows(model, rej, {k: v[rej] for k, v in snap.items()})
        if cfg["verbose"]:
            for i in np.nonzero(proposed)[0]:
                print(f"  inst {sc['instance_ids'][D['ks'][i]]} {sc['class_name'][D['ks'][i]]}: {kinds[i]} "
                      f"iou {iou_before[i]:.3f} -> {iou_before[i] + gain[i]:.3f} {'ACCEPT' if accept[i] else 'reject'}")
        fails[proposed & ~accept] += 1
        fails[accept] = 0
        ev = evaluate(model, D)
        n_add = sum(1 for i in np.nonzero(accept)[0] if kinds[i] == "add")
        log.append({"round": r, "proposed": int(proposed.sum()), "accepted": int(accept.sum()), "accepted_add": n_add,
                    "gain_median": float(np.median(gain[proposed])), "iou": ev["iou"].mean().item(),
                    "sqs": int(model.active.sum()), "t": time.time() - t0})
        print(log[-1])
    # merge: replace the most box-like pair of an instance by one box, refit, keep if the IoU fell by < prune_delta
    for r in range(cfg["merge_rounds"]):
        snap = snapshot_rows(model, torch.arange(I))
        iou_before = ev["iou"].clone()
        cand = torch.zeros(I, dtype=torch.bool)
        new_slots = torch.zeros(I, K, dtype=torch.bool)
        for i in range(I):
            p = merge_proposal(cfg, model, D, ev, i)
            if p is None:
                continue
            j, k, (t, R, a) = p
            model.set_slot(i, j, t, R, a, (cfg["eps_init"], cfg["eps_init"]))
            model.active[i, k] = False
            new_slots[i, j] = True
            cand[i] = True
        if not cand.any():
            break
        optimize(cfg, model, D, cfg["steps_warm"], g, cfg["tau"], cfg["lr"], only_slots=new_slots)
        optimize(cfg, model, D, cfg["steps_round"], g, cfg["tau"], cfg["lr"])
        ev = evaluate(model, D)
        drop = iou_before - ev["iou"]
        ok = cand & ((drop < cfg["prune_delta"]) | (drop * D["n_pos"] < cfg["min_gain_vox"]))
        rej = torch.nonzero(cand & ~ok)[:, 0]
        restore_rows(model, rej, {k: v[rej] for k, v in snap.items()})
        ev = evaluate(model, D)
        log.append({"round": f"merge{r}", "tried": int(cand.sum()), "merged": int(ok.sum()), "iou": ev["iou"].mean().item(),
                    "sqs": int(model.active.sum()), "t": time.time() - t0})
        print(log[-1])
    # prune with refit: drop the cheapest SQ of every instance with > 1 SQ, re-optimize, keep the removal if the
    # IoU dropped by less than prune_delta (the neighbours may grow into the gap: acts as a merge move)
    for r in range(cfg["prune_rounds"]):
        cand = model.active.sum(1) > 1
        if not cand.any():
            break
        snap = snapshot_rows(model, torch.arange(I))
        iou_before = ev["iou"].clone()
        cost = (ev["iou"][:, None] - ev["iou_wo"]).masked_fill(~model.active, 1e9)
        j = cost.argmin(1)
        rows = torch.nonzero(cand)[:, 0]
        model.active[rows, j[rows]] = False
        optimize(cfg, model, D, cfg["steps_round"], g, cfg["tau"], cfg["lr"])
        ev = evaluate(model, D)
        drop = iou_before - ev["iou"]
        ok = cand & ((drop < cfg["prune_delta"]) | (drop * D["n_pos"] < cfg["min_gain_vox"]))
        rej = torch.nonzero(cand & ~ok)[:, 0]
        restore_rows(model, rej, {k: v[rej] for k, v in snap.items()})
        ev = evaluate(model, D)
        log.append({"round": f"prune{r}", "tried": int(cand.sum()), "removed": int(ok.sum()), "iou": ev["iou"].mean().item(),
                    "sqs": int(model.active.sum()), "t": time.time() - t0})
        print(log[-1])
    optimize(cfg, model, D, cfg["steps_final"], g, cfg["tau_final"], cfg["lr"] * 0.3)
    ev = evaluate(model, D)
    log.append({"round": "final", "iou": ev["iou"].mean().item(), "sqs": int(model.active.sum()), "t": time.time() - t0})
    print(log[-1])
    if D_eval is not None:
        ev = evaluate(model, D_eval)
        D = D_eval
    return sc, vol, D, model, ev, log


def save(cfg, name, sc, vol, D, model, ev, log):
    from evaluate import metrics, render_all
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    par = {k: v.detach().numpy() for k, v in model.params().items()}
    act = model.active.numpy()
    ii, kk = np.nonzero(act)
    sqs = {k: v[ii, kk] for k, v in par.items()} | {"instance": ii}
    np.savez_compressed(out / "sqs.npz", **sqs, instance_ids=sc["instance_ids"])
    M = metrics(cfg, sc, vol, D, sqs, ev)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "metrics.json").write_text(json.dumps({"name": name, "summary": M["summary"], "log": log,
                                                  "per_instance": M["per_instance"]}, indent=1))
    render_all(sc, sqs, out)
    print(json.dumps(M["summary"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = parse_set(dict(CFG), args.set)
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", 4)))
    res = run(cfg, args.name)
    if not cfg["only"]:
        save(cfg, args.name, *res)


if __name__ == "__main__":
    main()
