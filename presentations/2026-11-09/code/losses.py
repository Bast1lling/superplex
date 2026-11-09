"""Matching and losses (exp 27's train.py, the terms the presentation uses).

Matching: Hungarian on the masks (cross-entropy + Dice of each slot's attention against each target's patch shares)
-> one slot per target (sig); unmatched slots are "empty". Losses (each weight 1):
  mask     cross-entropy of the attention (log-softmax over the slots) against the target shares of the patches
  dice     Dice loss of the matched slots' masks
  exist    BCE of the slot existence (matched = 1), targets smoothed to 0.01 / 0.99
  render   (no depth) the matched slots' SQs rendered (soft ray casting): -log of the pixel's own slot's
           compositing weight, i.e. the rendered SQs should reproduce the instance mask
  r_depth  (with depth) relative L1 of the rendered depth vs. the observed depth (covered pixels)
  obs_cd   visibility-aware Chamfer between the matched SQ and the instance's observed points (no target SQs)
  pair_cd  visibility-weighted Chamfer between the matched SQ and its target (the union of the target's SQs)
  part_cd  per matched slot, Hungarian matching of its 16 parts to the target's SuperFlex SQs (Chamfer cost; minus
           log existence, so existing parts are preferred), visibility-weighted Chamfer of each matched pair
  part_exist BCE of the part existences (matched parts = 1), smoothed like exist
Sample weights (Chamfer terms): 1 in the frustum, exp(-distance / 10 cm) outside it (data.sample_weights).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import data as D
import sq
from data import DEV

SMOOTH, N_OBS = 0.01, 512


def smooth(t):
    return t * (1 - 2 * SMOOTH) + SMOOTH


@torch.no_grad()
def match(pred, fr):
    """-> slot index per target (M,)."""
    v = fr["pvalid"]
    t = fr["share"][v]
    la = pred["logits"][v].log_softmax(1)
    a = la.exp()
    ce = -(la.T @ t) / t.sum(0).clamp_min(1e-6)
    dice = 1 - (2 * a.T @ t) / (a.sum(0)[:, None] + t.sum(0)[None])
    qi, ti = linear_sum_assignment((ce + dice).cpu().numpy())
    sig = torch.empty(fr["M"], dtype=torch.long, device=DEV)
    sig[torch.as_tensor(ti, device=DEV)] = torch.as_tensor(qi, device=DEV)
    return sig


@torch.no_grad()
def part_match(P, gr, exist_logit=None):
    """Per target, Hungarian matching of the slot's parts P (dict (M, G, ...)) to the target's valid SQs on a
    symmetric Chamfer cost -> index tensors (target, part, target SQ) of the pairs."""
    M, G = P["scale"].shape[:2]
    Gt = gr["valid"].shape[1]
    ps = sq.surf({k: v.flatten(0, 1) for k, v in P.items()}, 6).reshape(M, G, -1, 3)
    ts = sq.surf({k: v.flatten(0, 1) for k, v in gr["sq"].items()}, 6).reshape(M, Gt, -1, 3)
    n = ps.shape[2]
    d = torch.cdist(ps.reshape(M, G * n, 3), ts.reshape(M, Gt * n, 3)).reshape(M, G, n, Gt, n)
    cost = d.min(4).values.mean(2) + d.min(2).values.mean(-1)
    if exist_logit is not None:
        cost = cost - F.logsigmoid(exist_logit)[:, :, None]
    cost, valid = cost.cpu().numpy(), gr["valid"].cpu().numpy()
    mi, pi, ti = [], [], []
    for m in range(M):
        cols = np.flatnonzero(valid[m])
        r, c = linear_sum_assignment(cost[m][:, cols])
        mi += [m] * len(r)
        pi += r.tolist()
        ti += cols[c].tolist()
    return tuple(torch.as_tensor(x, dtype=torch.long, device=DEV) for x in (mi, pi, ti))


def weighted_chamfer(pp, tp, wp, wt):
    d = torch.cdist(pp, tp)
    return (d.min(2).values * wp).sum(1) / wp.sum(1).clamp_min(1e-6) + (d.min(1).values * wt).sum(1) / wt.sum(1).clamp_min(1e-6)


def losses(pred, fr, sig, use):
    K, M = pred["logits"].shape[1], fr["M"]
    out = {}
    v = fr["pvalid"]
    t = fr["share"][v]
    la = pred["logits"][v].log_softmax(1)
    T = torch.zeros_like(la)
    T[:, sig] = t
    out["mask"] = -(T * la).sum(1).mean()
    a = la.exp()[:, sig]
    out["dice"] = (1 - (2 * (a * t).sum(0) + 1) / (a.sum(0) + t.sum(0) + 1)).mean()
    tgt = torch.zeros(K, device=DEV)
    tgt[sig] = 1
    out["exist"] = F.binary_cross_entropy_with_logits(pred["exist_logit"], smooth(tgt))
    ms = {k: pred[k][sig] for k in sq.KEYS}  # the matched slots' SQs, in target order
    gr, depth = fr["groups"], fr["depth"]
    if "render" in use or "r_depth" in use:
        r = sq.render(ms, stride=2)
        if "render" in use:
            lab = fr["lab"][r["v"], r["u"]]
            ok = lab >= 0
            out["render"] = -torch.log(r["w"][ok, lab[ok]] + 1e-4).mean()
        if "r_depth" in use:
            obs = depth[r["v"], r["u"]]
            ok = (obs > 0) & (r["cover"] > 0.5)
            out["r_depth"] = ((r["depth"] - obs).abs() / obs.clamp_min(1e-3))[ok].mean() if ok.any() else r["depth"].sum() * 0
    if "obs_cd" in use:
        i = (torch.rand(M, N_OBS, device=DEV) * fr["pt_count"][:, None]).long()
        pts = fr["pts"][fr["pt_order"][fr["pt_start"][:, None] + i]]  # (M, N_OBS, 3) observed points per target
        pp = sq.surf(ms)
        with torch.no_grad():
            vis = (D.sample_weights(pp, depth, 0.0, 1e-9) > 0.5).float()  # SQ samples that should be observed
        d = torch.cdist(pts, pp)
        out["obs_cd"] = (d.min(2).values.mean(1) + (d.min(1).values * vis).sum(1) / vis.sum(1).clamp_min(1)).mean()
    if "pair_cd" in use:
        pp = sq.surf(ms)
        tp = torch.where(gr["keep"][..., None], gr["pts"], 1e3)
        wt = D.sample_weights(gr["pts"], depth) * gr["keep"]
        out["pair_cd"] = weighted_chamfer(pp, tp, D.sample_weights(pp.detach(), depth), wt).mean()
    if "part_cd" in use:
        Pm = {k: pred["parts"][k][sig] for k in sq.KEYS}
        el = pred["parts"]["exist_logit"][sig]
        mi, pi, ti = part_match({k: v.detach() for k, v in Pm.items()}, gr, el.detach())
        pp = sq.surf({k: v[mi, pi] for k, v in Pm.items()})
        tp = gr["pts"].reshape(M, gr["valid"].shape[1], -1, 3)[mi, ti]
        out["part_cd"] = weighted_chamfer(pp, tp, D.sample_weights(pp.detach(), depth), D.sample_weights(tp, depth)).mean()
        tgt = torch.zeros_like(el)
        tgt[mi, pi] = 1
        out["part_exist"] = F.binary_cross_entropy_with_logits(el, smooth(tgt))
    return out
