"""Fixed slots + existence + parsimony (no structure search), with the volumetric loss of fit.py.

Mirrors what a feed-forward model (fixed number of queries, existence head) would be trained with. K SQs per
instance, initialized by k-means of its target voxels (PCA box per cluster); init=multiscale: slot 0 is the box
of the whole instance, the other K-1 slots k-means parts. Existence e_j = sigmoid(logit_j).
Loss per sample = expectation over independent Bernoulli existences of the occupancy loss of the first existing SQ
in order of increasing SDF (Paschalidou et al. 2019), with the "no SQ exists" outcome scored as d = +fallback:
    E = sum_i e_(i) prod_{k<i} (1 - e_(k)) l(d_(i)) + prod_k (1 - e_k) l(fallback)
plus parsimony lambda * sum_j e_j per instance (in units of the per-instance loss). Active at evaluation: e > 0.5.

    python fit_exist.py --name room0_exist_l01 --set lam=0.1
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

from fit import CFG, RUNS, box_fit, evaluate, parse_set, prepare, sample, save
from sq import SQSet, sdf_slots
from volume import build_volume, load_scene

XCFG = dict(CFG) | dict(init="kmeans", K=8, lam=0.1, steps=3000, exist_init=0.9, lr_exist=0.05, fallback=0.3, anneal=True)


def expected(d, e, inst, loss_fn, fallback_val):
    """d (S, K) SDFs, e (I, K) existence -> (S,) expected loss."""
    order = d.argsort(1)
    ds = d.gather(1, order)
    es = e[inst].gather(1, order)
    none_before = torch.cumprod(torch.cat([torch.ones_like(es[:, :1]), 1 - es], 1), 1)
    return (es * none_before[:, :-1] * loss_fn(ds)).sum(1) + none_before[:, -1] * fallback_val


def run(cfg, name):
    t0 = time.time()
    g = torch.Generator().manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    sc = load_scene(cfg["scene"], min_points=cfg["min_points"])
    from pathlib import Path
    cache = Path("/tmp/claude-0/volcache") / f"{cfg['scene']}_{cfg['res']}_{cfg['min_points']}_v2.npz"
    if cache.exists():
        vol = dict(np.load(cache))
        vol["res"] = float(vol["res"])
    else:
        vol = build_volume(sc, res=cfg["res"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **vol)
    D = prepare(cfg, sc, vol)
    I, K = D["I"], cfg["K"]
    model = SQSet(I, K, eps_range=(0.1, cfg["eps_max"]))
    ms = cfg["init"] == "multiscale"  # slot 0 = box of the whole instance, the others k-means parts
    for i in range(I):
        x = D["pos_x"][D["pos_i"] == i].numpy()
        k = min(K - ms, len(x))
        lab = KMeans(k, n_init=1, random_state=cfg["seed"]).fit_predict(x) if k > 1 else np.zeros(len(x), int)
        for j in range(K):
            xs = x if (ms and j == 0) else x[lab == ((j - ms) % k)]
            t, R, a = box_fit(xs, cfg["res"])
            model.set_slot(i, j, t, R, a, (cfg["eps_init"], cfg["eps_init"]))
    logit = torch.nn.Parameter(torch.full((I, K), math.log(cfg["exist_init"] / (1 - cfg["exist_init"]))))
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": cfg["lr"]}, {"params": [logit], "lr": cfg["lr_exist"]}])
    tau = cfg["tau"]
    sp = torch.nn.functional.softplus
    fb_pos = sp(torch.tensor(cfg["fallback"] / tau)).item()
    log = []
    for step in range(cfg["steps"] + 1):
        if cfg["anneal"]:
            tau = cfg["tau"] + (cfg["tau_final"] - cfg["tau"]) * step / cfg["steps"]
        par = model.params()
        e = torch.sigmoid(logit)
        xp, ip, wp = sample(D, cfg["samples_pos"], "pos", g)
        xn, i_n, wn = sample(D, cfg["samples_neg"], "neg", g)
        lp = expected(sdf_slots(par, xp, ip), e, ip, lambda d: sp(d / tau), fb_pos)
        ln = expected(sdf_slots(par, xn, i_n), e, i_n, lambda d: sp(-d / tau), 0.0)
        per = torch.zeros(I).index_add(0, ip, wp * lp).index_add(0, i_n, wn * ln) / D["n_pos"].clamp(min=1)
        ext = (par["rot"].abs() * par["scale"][..., None, :]).sum(-1)
        out = torch.relu(par["trans"] + ext - D["region_hi"][:, None]) + torch.relu(D["region_lo"][:, None] - par["trans"] + ext)
        reg = (out.sum(-1) * e).sum(1)
        loss = per.mean() + cfg["w_region"] * reg.mean() + cfg["lam"] * e.sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                model.active.copy_(torch.sigmoid(logit) > 0.5)
                ev = evaluate(model, D)
            log.append({"step": step, "loss": loss.item(), "iou": ev["iou"].mean().item(), "sqs": int(model.active.sum()),
                        "t": time.time() - t0})
            print(log[-1], flush=True)
    model.active.copy_(torch.sigmoid(logit) > 0.5)
    # instances whose SQs all died keep their most existent one
    dead = ~model.active.any(1)
    model.active[dead, logit[dead].argmax(1)] = True
    ev = evaluate(model, D)
    return sc, vol, D, model, ev, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = parse_set(dict(XCFG), args.set)
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", 4)))
    save(cfg, args.name, *run(cfg, args.name))


if __name__ == "__main__":
    main()
