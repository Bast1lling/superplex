"""Quality-driven decompositions: a part gets one SQ, and parts are split / merged by whether ONE SQ explains them
well (part IoU >= q), instead of 08's marginal-gain test on the whole instance (+delta IoU per extra SQ).

    python decomp.py --name <run> [--set method=topdown|bottomup|both|coacd|coacd_merge] [--set key=value ...]

topdown : start with one part per group; fit one SQ to every open part (batched); parts with IoU < q are split
          (2-means on voxel centers, then connected components) until IoU >= q or the part is small.
bottomup: start with cells of `cell` m (per group, split into connected components); repeatedly fit one SQ to the
          union of every adjacent pair of parts of the same group (batched) and merge the best disjoint pairs with
          IoU >= q.
both    : topdown, then bottomup-style merging of the leaves (fixes over-splitting by the arbitrary 2-means cuts).
rd      : top-down, a split is kept iff it lowers the absolute tolerant error by > lam voxels; rd_merge: + merging.
coacd   : parts = CoACD convex decomposition of each instance (convex.py, concavity threshold cx); coacd_merge: + merging.
Decisions use a discretization-tolerant IoU (tol voxels, partfit.iou).
Afterwards: one SQ per final part, joint refinement of every group's union (08's loss), prune SQs whose removal
costs < prune IoU, scoring with 08's strict weights and renders.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
import torch

import partfit as pf

EXP = Path(__file__).resolve().parents[1]
RUNS = EXP / "results" / "runs"
CFG = dict(scene="office_0", method="both", q=0.7, min_vox=8, max_depth=10, cell=0.2, cx=0.05, tol=0.5, lam=30.0, steps=300, refine_steps=400,
           prune=0.01, planar=True, eps_max=1.2, seed=0)


def parse_set(cfg, items):
    for it in items:
        k, v = it.split("=", 1)
        t = type(cfg[k])
        cfg[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
    return cfg


# ---------------------------------------------------------------- top-down

def split2(vox, x, min_vox, seed):
    """2-means split of a part, then connected components; components < min_vox go to the other cluster."""
    lab = KMeans(2, n_init=2, random_state=seed).fit_predict(x)
    out = []
    for c in (0, 1):
        idx = np.nonzero(lab == c)[0]
        if len(idx) == 0:
            continue
        for comp in pf.components(vox[idx]):
            out.append(idx[comp])
    big = [o for o in out if len(o) >= min_vox]
    small = [o for o in out if len(o) < min_vox]
    if not big:
        return [np.arange(len(vox))]
    if small:  # attach small components to the nearest big one
        cent = np.stack([x[b].mean(0) for b in big])
        for o in small:
            j = int(np.argmin(((cent - x[o].mean(0)) ** 2).sum(1)))
            big[j] = np.concatenate([big[j], o])
    return big


def topdown(S, parts, group, cfg, log):
    open_ = [(p, g, 0) for p, g in zip(parts, group)]
    leaves = []
    while open_:
        _, _, q, _ = pf.single_fit(S, [p for p, _, _ in open_], [g for _, g, _ in open_], steps=cfg["steps"],
                                   eps_max=cfg["eps_max"], n_eval=4096,
                                   tol=cfg["tol"] * S.res)
        nxt = []
        for (p, g, d), qi in zip(open_, q):
            if qi >= cfg["q"] or len(p) < 2 * cfg["min_vox"] or d >= cfg["max_depth"]:
                leaves.append((p, g))
                continue
            ch = split2(p, S.centers(p), cfg["min_vox"], cfg["seed"] + d)
            if len(ch) == 1:
                leaves.append((p, g))
            else:
                nxt += [(p[c], g, d + 1) for c in ch]
        log.append({"stage": "topdown", "fitted": len(open_), "leaves": len(leaves), "open": len(nxt),
                    "mean_q": float(q.mean())})
        print(log[-1], flush=True)
        open_ = nxt
    return [p for p, _ in leaves], [g for _, g in leaves]


def fit_err(S, parts, group, cfg):
    """Tolerant error (missed + weighted leaked voxels) of one SQ per part."""
    _, _, q, rec = pf.single_fit(S, parts, group, steps=cfg["steps"], eps_max=cfg["eps_max"], n_eval=4096,
                                 tol=cfg["tol"] * S.res)
    cnt = np.array([len(p) for p in parts])
    tp = rec * cnt
    fp = tp / np.maximum(q, 1e-6) - cnt
    return cnt - tp + fp


def topdown_rd(S, parts, group, cfg, log):
    """Rate-distortion top-down: a split is kept iff the children's tolerant error is lower than the parent's by
    more than lam voxels (absolute price of a primitive; 08's criterion without the relative delta or the K cap)."""
    nodes = list(zip(parts, group, fit_err(S, parts, group, cfg), [0] * len(parts)))
    leaves = []
    while nodes:
        kids, owner = [], []
        for n, (p, g, e, d) in enumerate(nodes):
            if len(p) < 2 * cfg["min_vox"] or d >= cfg["max_depth"]:
                leaves.append((p, g))
                continue
            ch = split2(p, S.centers(p), cfg["min_vox"], cfg["seed"] + d)
            if len(ch) == 1:
                leaves.append((p, g))
                continue
            kids += [(p[c], g) for c in ch]
            owner += [n] * len(ch)
        if not kids:
            break
        ek = fit_err(S, [k[0] for k in kids], [k[1] for k in kids], cfg)
        owner = np.array(owner)
        nxt = []
        for n in np.unique(owner):
            p, g, e, d = nodes[n]
            m = np.nonzero(owner == n)[0]
            if e - ek[m].sum() > cfg["lam"]:
                nxt += [(kids[i][0], g, ek[i], d + 1) for i in m]
            else:
                leaves.append((p, g))
        log.append({"stage": "topdown_rd", "split_tried": int(len(np.unique(owner))), "accepted": len(nxt),
                    "leaves": len(leaves)})
        print(log[-1], flush=True)
        nodes = nxt
    return [p for p, _ in leaves], [g for _, g in leaves]


# ---------------------------------------------------------------- bottom-up

def cells(S, parts, group, cell):
    c = max(1, int(round(cell / S.res)))
    out, grp = [], []
    for p, g in zip(parts, group):
        key = p // c
        _, inv = np.unique(key, axis=0, return_inverse=True)
        for k in range(inv.max() + 1):
            idx = np.nonzero(inv.ravel() == k)[0]
            for comp in pf.components(p[idx]):
                out.append(p[idx[comp]])
                grp.append(g)
    return out, grp


def adjacency(S, parts, group):
    """Pairs (a, b), a < b, of parts of the same group that touch (26-neighborhood)."""
    P = np.full(tuple(S.dims), -1, np.int64)
    for i, p in enumerate(parts):
        P[tuple(p.T)] = i
    pairs = set()
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]
    allv = np.concatenate(parts)
    ida = P[tuple(allv.T)]
    for o in offs:
        nb = allv + np.array(o)
        ok = ((nb >= 0) & (nb < S.dims)).all(1)
        b = np.full(len(allv), -1)
        b[ok] = P[tuple(nb[ok].T)]
        m = (b >= 0) & (b != ida) & (ida >= 0)
        a_, b_ = ida[m], b[m]
        same = np.asarray(group)[a_] == np.asarray(group)[b_]
        for x, y in zip(np.minimum(a_, b_)[same], np.maximum(a_, b_)[same]):
            pairs.add((int(x), int(y)))
    return P, pairs


def bottomup(S, parts, group, cfg, log, max_batch=6000):
    parts, group = list(parts), list(group)
    alive = [True] * len(parts)
    score = {}
    it = 0
    while True:
        P, pairs = adjacency_alive(S, parts, group, alive)
        todo = [pr for pr in pairs if pr not in score]
        Pt = torch.from_numpy(P).to(pf.DEV)
        for s in range(0, len(todo), max_batch):
            b = todo[s:s + max_batch]
            _, _, q, _ = pf.single_fit(S, [np.concatenate([parts[i], parts[j]]) for i, j in b], [group[i] for i, _ in b],
                                       steps=cfg["steps"], eps_max=cfg["eps_max"], P=Pt, own=np.array(b), n_eval=2048,
                                       tol=cfg["tol"] * S.res)
            score.update(zip(b, q.tolist()))
        cand = sorted(((score[pr], pr) for pr in pairs if score[pr] >= cfg["q"]), reverse=True)
        used, merged = set(), 0
        for sc_, (i, j) in cand:
            if i in used or j in used:
                continue
            used |= {i, j}
            parts[i] = np.concatenate([parts[i], parts[j]])
            parts[j] = np.zeros((0, 3), int)
            alive[j] = False
            merged += 1
        score = {pr: v for pr, v in score.items() if pr[0] not in used and pr[1] not in used}
        it += 1
        log.append({"stage": "bottomup", "iter": it, "parts": sum(alive), "pairs": len(pairs), "fitted": len(todo),
                    "merged": merged})
        print(log[-1], flush=True)
        if merged == 0:
            break
    keep = [i for i, a in enumerate(alive) if a]
    return [parts[i] for i in keep], [group[i] for i in keep]


def adjacency_alive(S, parts, group, alive):
    idx = [i for i, a in enumerate(alive) if a]
    P, pairs = adjacency(S, [parts[i] for i in idx], [group[i] for i in idx])
    m = np.array(idx)
    P = np.where(P >= 0, m[np.maximum(P, 0)], -1)
    return P, {(int(m[a]), int(m[b])) for a, b in pairs}


# ---------------------------------------------------------------- final fit, refinement, scoring

def finalize(S, parts, group, groups, cfg, log):
    """One SQ per part -> joint refinement of each group's union -> prune. Returns flat sqs (instance = group)."""
    model, _, q, _ = pf.single_fit(S, parts, group, steps=cfg["steps"], eps_max=cfg["eps_max"],
                                  tol=cfg["tol"] * S.res)
    sqs = pf.model_to_sqs(model, group)
    gm = pf.group_model(S, sqs, groups, cfg["eps_max"])
    gsets = pf.Sets(S, [S_targets(S, g) for g in groups], np.array(groups), is_part=False)
    pf.optimize(gm, gsets, cfg["refine_steps"], lr=0.005)
    tol = cfg["tol"] * S.res
    base, _, _ = pf.iou(gm, gsets, tol=tol)
    removed = 0
    for r in range(3):  # prune: drop the slot with the smallest IoU loss per group if the loss < prune
        K = gm.K
        loss = torch.full((gm.I, K), 1e9, device=pf.DEV)
        for j in range(K):
            m = gm.active[:, j].clone()
            if not m.any():
                continue
            gm.active[:, j] = False
            qj, _, _ = pf.iou(gm, gsets, n_eval=4096, tol=tol)
            gm.active[:, j] = m
            loss[:, j] = torch.where(m & (gm.active.sum(1) > 1), base - qj, loss[:, j])
        j = loss.argmin(1)
        ok = loss[torch.arange(gm.I), j] < cfg["prune"]
        if not ok.any():
            break
        gm.active[torch.nonzero(ok)[:, 0], j[ok]] = False
        removed += int(ok.sum())
        base, _, _ = pf.iou(gm, gsets, tol=tol)
    log.append({"stage": "final", "sqs": int(gm.active.sum()), "pruned": removed, "part_q_mean": float(q.mean()),
                "group_iou_mean": float(base.mean())})
    print(log[-1], flush=True)
    return pf.model_to_sqs(gm, groups)


def S_targets(S, g):
    return S.targets[g]


def score(scene, sqs, n_inst):
    """Per-instance strict vIoU (08's metric weights, no planar completion) of each instance's SQ union."""
    S = pf.Scene(scene, planar=False)
    groups = list(range(n_inst))
    gm = pf.group_model(S, sqs, groups, eps_max=1.9)
    sets = pf.Sets(S, S.targets, np.array(groups), is_part=False, margin=0.3)
    q, rec, fp = pf.iou(gm, sets, w=pf.STRICT, n_eval=32768)
    tp = (rec * sets.cnt).cpu().numpy()
    prec = tp / np.maximum(tp + fp.cpu().numpy(), 1e-6)
    return {"viou": q.cpu().numpy().tolist(), "recall": rec.cpu().numpy().tolist(), "precision": prec.tolist()}


def summarize(S, sqs, sc_):
    sys.path.insert(0, str(pf.EXP08))
    from plausible import plausibility
    I = S.n_inst
    K = np.bincount(sqs["instance"], minlength=I)
    per = {"name": S.sc["class_name"], "instance_id": S.sc["instance_ids"].tolist(), "voxels": [len(t) for t in S.targets],
           "K": K.tolist()} | sc_
    pl, over, under = plausibility(per)
    summ = {"instances": I, "sqs": int(K.sum()), "viou": float(np.mean(sc_["viou"])),
            "precision": float(np.mean(sc_["precision"])), "recall": float(np.mean(sc_["recall"])),
            "plausible": pl, "over": over, "under": under}
    return summ, per


def save(name, cfg, S, sqs, log, t0, exp=EXP):
    from evaluate import render_all
    out = exp / "results" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    sqs = {k: np.asarray(v) for k, v in sqs.items()}
    np.savez_compressed(out / "sqs.npz", **sqs, instance_ids=S.sc["instance_ids"])
    sc_ = score(S.cfg["scene"], sqs, S.n_inst)
    summ, per = summarize(S, sqs, sc_)
    summ["seconds"] = round(time.time() - t0, 1)
    (out / "config.json").write_text(json.dumps(cfg | S.cfg, indent=1))
    (out / "metrics.json").write_text(json.dumps({"name": name, "summary": summ, "log": log, "per_instance": per}, indent=1))
    render_all(S.sc, sqs, out)
    print(name, json.dumps(summ), flush=True)
    return summ, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = parse_set(dict(CFG), a.set)
    torch.manual_seed(cfg["seed"])
    t0 = time.time()
    S = pf.Scene(cfg["scene"], planar=cfg["planar"])
    groups = list(range(S.n_inst))
    parts, group, log = list(S.targets), groups, []
    if cfg["method"] in ("topdown", "both"):
        parts, group = topdown(S, parts, group, cfg, log)
    if cfg["method"] in ("rd", "rd_merge"):
        parts, group = topdown_rd(S, parts, group, cfg, log)
    if cfg["method"] == "bottomup":
        parts, group = cells(S, parts, group, cfg["cell"])
    if cfg["method"] in ("coacd", "coacd_merge"):
        import convex
        parts, group = convex.decompose(parts, group, cfg["cx"], cfg["min_vox"])
        log.append({"stage": "coacd", "parts": len(parts)})
        print(log[-1], flush=True)
    if cfg["method"] in ("bottomup", "both", "coacd_merge", "rd_merge"):
        parts, group = bottomup(S, parts, group, cfg, log)
    sqs = finalize(S, parts, group, groups, cfg, log)
    save(a.name, cfg, S, sqs, log, t0)


if __name__ == "__main__":
    main()
