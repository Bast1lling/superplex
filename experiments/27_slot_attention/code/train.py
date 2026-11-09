"""Overfit the slot model (`model.py`) on one Replica frame; all loss weights 1, Adam at a constant --lr (5e-3).

    uv run python train.py --name base [--frame room_2/000500] [--steps 3000] [ablation flags]
    uv run python dashboard.py                                         -> http://127.0.0.1:7827

Per step (PLAN.md):
--arch m2f: Mask2Former-style queries instead of Slot Attention (`model.M2FDecoder`): sigmoid masks per query; mask
loss = BCE + Dice of the matched queries only, matching cost BCE + Dice, patch -> argmax of exist * mask.
1. Hungarian matching (no grad) of the M targets to the K slots on cost CE[k, m] + Dice[k, m] between the slot masks
   A = softmax_K(logits) (N patches x K) and the targets' patch shares t (N x M); CE normalized per target.
2. Losses (`--losses`, comma list; each weight 1):
   mask     CE over the slots per labelled patch, target T[n, sigma(m)] = t[n, m] (unmatched slots 0)
   dice     Dice of each matched pair
   exist    BCE of the existence logits, matched = 1, unmatched = 0
   (SDF = --distance: scaled (default, exp 20's (g - 1) / |grad g|, ~Euclidean) or radial (SuperFlex's))
   sdf      SuperFlex's surface term: |LeakyReLU(tau tanh(SDF / tau))| (tau = 0.1 m) of every labelled observed point
            against its slot's SQ; --assign gt: the slot matched to the point's target only; pred: all slots
            weighted by the point's patch's slot probabilities (SuperFlex's assignment)
   pair_cd  visibility-weighted Chamfer between each matched slot's SQ and its target (cube samples, 294 per SQ; a
            target = the union of its group's SQs: one SQ, or all SQs of the GT instance with --mask_source instance,
            its samples inside another SQ of the group left out)
   pair_iou 1 - visibility-weighted soft IoU of the pair, sigmoid(-SDF / 1 cm) against the target union's hard
            occupancy; points: 1024 uniform in the oriented box (+ 3 cm) of the slot's SQ and of each target SQ,
            importance-weighted by the box mixture density (`box_samples`; before 2026-11-08 evening: 1024 points
            uniform in the pair's axis-aligned box, nearly no samples inside thin slabs)
   obs_cd   visibility-aware Chamfer against the observed points, no target SQs: 512 of the slot's labelled points
            (resampled per step) -> its SQ's cube samples (294), plus SQ samples -> nearest of those points for the
            samples in the frustum and not behind the observed depth (+ 3 cm; the SQ's back side and occluded parts
            are free; samples outside the frustum have weight 0)
   free     soft union occupancy 1 - prod(1 - sigmoid(-SDF / 1 cm)) of the matched SQs at points sampled on the pixel
            rays in front of the observed depth (>= 5 cm in front)
   overlap  SuperFlex's overlap term over the matched SQs: relu(sum_i sigmoid(-(SDF_i + 1 cm) / 5 mm) - 1), at
            observed points and points uniform in their box
   part_cd  (--parts G) per matched slot, Hungarian matching of its G parts to its target group's SQs (Chamfer cost,
            24 cube samples), visibility-weighted Chamfer of each matched (part, target SQ) pair (294 samples each)
   part_exist BCE of the part existences of the matched slots (matched parts = 1)
   render   (ablation for free) differentiable rendering of the matched SQs (stride 2): relative depth L1 (rays with
            coverage > 0.5) + instance term -log w(pixel's slot)
   Sample weights (pair terms): in the frustum 1 if not behind the observed surface, else --w_occ; outside the
   frustum exp(-d / --lam), d = metric distance to the frustum (`data.sample_weights`).
Geometry terms use the matched slots only; their existence is not part of the geometry.
Metrics: pair_iou_fr = volumetric IoU of each matched pair inside the camera frustum (main); pair_iou_full = without
the frustum restriction; both estimated by importance sampling in the SQs' own boxes (`pair_iou_mc`).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

EXP = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRACKIO_DIR", str(EXP / "results" / "large_files" / "trackio"))
import trackio  # noqa: E402

import common as C  # noqa: E402
import data as D  # noqa: E402
from common import DEV, NP, P, GH, GW  # noqa: E402
from model import Net  # noqa: E402

PROJECT = "slot_attention"
OUT = EXP / "results" / "large_files" / "runs"
DEFAULT_LOSSES = "mask,dice,exist,sdf,pair_cd,pair_iou,free,overlap"
TAU_SDF, TAU_OCC = 0.1, 0.01
N_FREE, N_OVL, N_IOU, N_SURF, N_OBS = 8192, 4096, 1024, 16, 512
A = None  # parsed args


def sq_of(pred, idx):
    return {k: pred[k][idx] for k in C.KEYS}


# ------------------------------------------------------------------------------------------------ matching

def probs(pred):
    """(N, K) mask probabilities: softmax over the slots (Slot Attention) or a sigmoid per query (--arch m2f)."""
    return pred["logits"].sigmoid() if pred.get("sigmoid") else pred["logits"].softmax(1)


def assign_scores(pred):
    """(N, K) scores whose argmax over K is the patch's slot (m2f: exist * mask, Mask2Former's inference)."""
    if pred.get("sigmoid"):
        return pred["logits"].sigmoid() * pred["exist_logit"].sigmoid()[None]
    return pred["logits"]


@torch.no_grad()
def match(pred, fr):
    """-> slot index per target (M,) (Hungarian on CE (BCE for m2f) + Dice of the masks)."""
    v = fr["pvalid"]
    logits = pred["logits"]
    t = fr["share"][v]  # (n, M)
    if pred.get("sigmoid"):
        lg = logits[v]
        a = lg.sigmoid()
        ce = -(F.logsigmoid(lg).T @ t + F.logsigmoid(-lg).T @ (1 - t)) / len(lg)  # (K, M) mean BCE
    else:
        la = logits[v].log_softmax(1)  # (n, K)
        a = la.exp()
        ce = -(la.T @ t) / t.sum(0).clamp_min(1e-6)  # (K, M)
    dice = 1 - (2 * a.T @ t) / (a.sum(0)[:, None] + t.sum(0)[None])
    qi, ti = linear_sum_assignment((ce + dice).cpu().numpy())
    sig = torch.empty(fr["M"], dtype=torch.long, device=DEV)
    sig[torch.as_tensor(ti, device=DEV)] = torch.as_tensor(qi, device=DEV)
    return sig


# --------------------------------------------------------------------------------------------------- losses

def weighted_chamfer(pp, tp, wp, wt):
    d = torch.cdist(pp, tp)
    a = (d.min(2).values * wp).sum(1) / wp.sum(1).clamp_min(1e-6)
    b = (d.min(1).values * wt).sum(1) / wt.sum(1).clamp_min(1e-6)
    return a + b


def group_inside(x, gr):
    """x (M, N, 3) -> (M, N): inside the union of target group m's SQs."""
    return ((C.sdf(x, gr["sq"]) < 0) & gr["valid"][:, :, None]).any(1)


def group_box(pp, gr):
    """Box around the prediction samples pp (M, n, 3) and group m's union samples -> lo, hi (M, 3), +10 %."""
    k = gr["keep"][..., None]
    lo = torch.minimum(pp.min(1).values, torch.where(k, gr["pts"], torch.inf).min(1).values)
    hi = torch.maximum(pp.max(1).values, torch.where(k, gr["pts"], -torch.inf).max(1).values)
    pad = (hi - lo) * 0.1
    return lo - pad, hi + pad


def smooth(t):
    """--exist_smooth e: existence targets e / 1 - e instead of 0 / 1 (bounded logits; see README, collapse)."""
    return t * (1 - 2 * A.exist_smooth) + A.exist_smooth


def losses(pred, fr, sig):
    use = set(A.losses.split(","))
    K, M = pred["logits"].shape[1], fr["M"]
    out = {}
    v = fr["pvalid"]
    t = fr["share"][v]
    sigm = pred.get("sigmoid")
    if sigm:  # Mask2Former: BCE + Dice of the matched queries only
        lg = pred["logits"][v][:, sig]
        if "mask" in use:
            out["mask"] = F.binary_cross_entropy_with_logits(lg, t)
    else:
        la = pred["logits"][v].log_softmax(1)
        if "mask" in use:
            T = torch.zeros_like(la)
            T[:, sig] = t
            out["mask"] = -(T * la).sum(1).mean()
    if "dice" in use:
        a = lg.sigmoid() if sigm else la.exp()[:, sig]
        out["dice"] = (1 - (2 * (a * t).sum(0) + 1) / (a.sum(0) + t.sum(0) + 1)).mean()
    if "exist" in use:
        tgt = torch.zeros(K, device=DEV)
        tgt[sig] = 1
        out["exist"] = F.binary_cross_entropy_with_logits(pred["exist_logit"], smooth(tgt))
    ms = sq_of(pred, sig)  # matched SQs in target order
    gr = fr["groups"]
    wt_t = torch.ones(M, device=DEV)  # per-target weight of the geometry terms
    if A.geo_owned:  # only targets whose slot owns >= 1 patch (argmax)
        owned = torch.zeros(K, dtype=torch.bool, device=DEV)
        owned[assign_scores(pred)[v].argmax(1)] = True
        wt_t = owned[sig].float()
    gsel = torch.nonzero(wt_t > 0)[:, 0]
    mss = {k: w[gsel] for k, w in ms.items()}  # for the set terms (free, overlap)  # target m = the union of its group's SQs (one SQ unless --mask_source instance)
    if "sdf" in use:
        x = fr["pts"]
        if A.assign == "gt":
            s = C.sdf_pairs(x[:, None], {k: w[fr["pts_lab"]] for k, w in ms.items()})[:, 0]
            wp_ = wt_t[fr["pts_lab"]]
            out["sdf"] = (torch.abs(F.leaky_relu(TAU_SDF * torch.tanh(s / TAU_SDF))) * wp_).sum() / wp_.sum().clamp_min(1)
        else:  # SuperFlex: all slots weighted by the point's patch's assignment
            pix = torch.nonzero(fr["lab"].reshape(-1) >= 0)[:, 0]
            patch = (pix // D.W // P) * GW + (pix % D.W) // P
            pr = probs(pred)
            assign = (pr / pr.sum(1, keepdim=True).clamp_min(1e-6))[patch]  # (Q, K)
            s = torch.cat([C.sdf(x[None, a:a + 8192], {k: w[None] for k, w in sq_of(pred, slice(None)).items()})[0]
                           for a in range(0, len(x), 8192)], 1)  # (K, Q)
            r = torch.abs(F.leaky_relu(TAU_SDF * torch.tanh(s / TAU_SDF)))
            out["sdf"] = (assign.T * r).sum(0).mean()
    if "pair_cd" in use or "pair_iou" in use:
        pp = C.surf(ms, N_SURF, "cube")
        tp = torch.where(gr["keep"][..., None], gr["pts"], 1e3)  # union surface samples (others far away)
        wp = D.sample_weights(pp.detach(), fr["depth"], A.w_occ, A.lam)
        wt = D.sample_weights(gr["pts"], fr["depth"], A.w_occ, A.lam) * gr["keep"]
        if "pair_cd" in use:
            out["pair_cd"] = (weighted_chamfer(pp, tp, wp, wt) * wt_t).sum() / wt_t.sum().clamp_min(1)
        if "pair_iou" in use:  # box importance sampling (as the metric), boxes + 3 TAU_OCC for the soft edge
            x, wq = box_samples({k: v.detach() for k, v in ms.items()}, gr, N_IOU, margin=3 * TAU_OCC)
            p = torch.sigmoid(-C.sdf_pairs(x, ms) / TAU_OCC)
            with torch.no_grad():
                g = group_inside(x, gr).float()
                w = D.sample_weights(x, fr["depth"], A.w_occ, A.lam) * wq
            iou = (w * p * g).sum(1) / (w * (p + g - p * g)).sum(1).clamp_min(1e-12)
            out["pair_iou"] = ((1 - iou) * wt_t).sum() / wt_t.sum().clamp_min(1)
    if "part_cd" in use or "part_exist" in use:  # --parts: each matched slot's parts vs. its target group's SQs
        Pm = {k: pred["parts"][k][sig] for k in C.KEYS}  # (M, G, ...)
        mi, pi, ti = part_match({k: v.detach() for k, v in Pm.items()}, gr,
                                pred["parts"]["exist_logit"][sig].detach() if A.part_match_exist else None)
        if "part_cd" in use:
            pp = C.surf({k: v[mi, pi] for k, v in Pm.items()}, N_SURF, "cube")  # (n, S, 3)
            Gt = gr["valid"].shape[1]
            tp = gr["pts"].reshape(M, Gt, -1, 3)[mi, ti]  # the matched member's own samples
            wp = D.sample_weights(pp.detach(), fr["depth"], A.w_occ, A.lam)
            wt = D.sample_weights(tp, fr["depth"], A.w_occ, A.lam)
            out["part_cd"] = (weighted_chamfer(pp, tp, wp, wt) * wt_t[mi]).sum() / wt_t[mi].sum().clamp_min(1)
        if "part_exist" in use:
            tgt = torch.zeros_like(pred["parts"]["exist_logit"][sig])
            tgt[mi, pi] = 1
            out["part_exist"] = F.binary_cross_entropy_with_logits(pred["parts"]["exist_logit"][sig], smooth(tgt))
    if "obs_cd" in use:  # visibility-aware Chamfer against the slot's observed points (no target SQs)
        r = (torch.rand(M, N_OBS, device=DEV) * fr["pt_count"][:, None]).long()
        P_ = fr["pts"][fr["pt_order"][fr["pt_start"][:, None] + r]]  # (M, N_OBS, 3) points of each target
        pp = C.surf(ms, N_SURF, "cube")  # (M, 294, 3)
        with torch.no_grad():
            vis = (D.sample_weights(pp, fr["depth"], 0.0, 1e-9) > 0.5).float()  # in frustum, not behind the surface
        d = torch.cdist(P_, pp)  # (M, N_OBS, 294)
        to_sq = d.min(2).values.mean(1)
        to_pts = (d.min(1).values * vis).sum(1) / vis.sum(1).clamp_min(1)
        out["obs_cd"] = ((to_sq + to_pts) * wt_t).sum() / wt_t.sum().clamp_min(1)
    if "free" in use:
        d = fr["depth"].reshape(-1)
        ok = torch.nonzero(d > C.ZNEAR + 0.06)[:, 0]
        r = ok[torch.randint(len(ok), (N_FREE,), device=DEV)]
        dirs = C.rays(1, DEV)[0][r]
        z = C.ZNEAR + torch.rand(N_FREE, device=DEV) * (d[r] - 0.05 - C.ZNEAR)
        x = dirs * z[:, None]
        p = torch.sigmoid(-C.sdf(x[None], {k: w[None] for k, w in mss.items()})[0] / TAU_OCC)  # (M, n)
        out["free"] = (1 - torch.exp(torch.log1p(-p.clamp(max=0.9999)).sum(0))).mean()
    if "overlap" in use:
        pts = fr["pts"][torch.randint(len(fr["pts"]), (N_OVL,), device=DEV)]
        lo, hi = fr["pts"].min(0).values, fr["pts"].max(0).values
        x = torch.cat([pts, lo + torch.rand(N_OVL, 3, device=DEV) * (hi - lo)])
        ind = torch.sigmoid(-(C.sdf(x[None], {k: w[None] for k, w in mss.items()})[0] + 0.01) / 0.005)
        out["overlap"] = torch.relu(ind.sum(0) - 1).mean()
    if "render" in use:
        r = C.render(ms, torch.ones(M, device=DEV), stride=2)
        obs = fr["depth"][r["v"], r["u"]]
        lab = fr["lab"][r["v"], r["u"]]
        okd = (obs > 0) & (r["cover"] > 0.5)
        out["r_depth"] = ((r["depth"] - obs).abs() / obs.clamp_min(1e-3))[okd].mean() if okd.any() else r["depth"].sum() * 0
        okl = lab >= 0
        out["r_inst"] = -torch.log(r["w"][okl, lab[okl]] + 1e-4).mean()
    return out


# -------------------------------------------------------------------------------------------------- metrics

def single(ms):
    """M single SQs as M groups of one."""
    return {"sq": {k: v[:, None] for k, v in ms.items()}, "valid": torch.ones(len(ms["scale"]), 1, dtype=torch.bool, device=DEV)}


@torch.no_grad()
def group_box_samples(ga, gb, n, margin=0.0):
    """n points uniform in the oriented box (half-axes + margin) of each valid SQ of groups ga[m] and gb[m] ->
    x (M, B n, 3) and the importance weight 1 / q(x) / (number of valid samples) per point (M, B n), q = the mixture
    density of the valid boxes (0 weight for samples of invalid / padded members)."""
    sq = {k: torch.cat([ga["sq"][k], gb["sq"][k]], 1) for k in ("scale", "rotate", "trans")}
    M = len(sq["scale"])
    sc, R, t = sq["scale"] + margin, sq["rotate"], sq["trans"]  # (M, B, ...)
    bv = torch.cat([ga["valid"], gb["valid"]], 1)
    B = sc.shape[1]
    u = (torch.rand(M, B, n, 3, device=DEV) * 2 - 1) * sc[:, :, None]
    x = (torch.einsum("mbij,mbnj->mbni", R, u) + t[:, :, None]).reshape(M, B * n, 3)
    xv = bv[:, :, None].expand(M, B, n).reshape(M, B * n)
    loc = torch.einsum("mbji,mbnj->mbni", R, x[:, None] - t[:, :, None])  # (M, B, B n, 3)
    inbox = (loc.abs() <= sc[:, :, None]).all(-1) & bv[:, :, None]
    q = (inbox.float() / (8 * sc.prod(-1))[:, :, None]).sum(1) / bv.sum(1, keepdim=True).clamp_min(1)
    return x, xv.float() / q.clamp_min(1e-12) / xv.sum(1, keepdim=True).clamp_min(1)


def box_samples(ms, gr, n, margin=0.0):
    return group_box_samples(single(ms), gr, n, margin)


@torch.no_grad()
def group_pair_iou(ga, gb, n=4096):
    """Volumetric IoU of the unions of groups ga[m] and gb[m], full and inside the camera frustum -> (M,), (M,) (NaN
    where both are empty there). Importance sampling in the SQs' own oriented boxes (`group_box_samples`; a plain SQ
    lies inside its box): |S| = mean of 1_S(x) / q(x). Dense on thin slabs, unlike uniform points in a box."""
    x, w = group_box_samples(ga, gb, n)
    ia, ib = group_inside(x, ga), group_inside(x, gb)
    f = D.frustum_distance(x) == 0
    out = []
    for m in (torch.ones_like(f), f):
        I, Va, Vb = (w * (ia & ib & m)).sum(1), (w * (ia & m)).sum(1), (w * (ib & m)).sum(1)
        U = Va + Vb - I
        out.append(torch.where(U > 0, I / U.clamp_min(1e-12), torch.nan))
    return out


def pair_iou_mc(ms, gr, n=4096):
    """IoU of each matched slot SQ (ms, M) with its target group's union (gr): `group_pair_iou`."""
    return group_pair_iou(single(ms), gr, n)


@torch.no_grad()
def part_match(P, gr, exist_logit=None):
    """Hungarian matching per target m between the slot's G parts P (dict (M, G, ...)) and the target group's valid
    SQs, on a symmetric Chamfer cost (24 cube samples each) -> index tensors (m, part, member) of the pairs."""
    M, G = P["scale"].shape[:2]
    Gt = gr["valid"].shape[1]
    ps = C.surf({k: v.flatten(0, 1) for k, v in P.items()}, 6, "cube").reshape(M, G, -1, 3)
    ts = C.surf({k: v.flatten(0, 1) for k, v in gr["sq"].items()}, 6, "cube").reshape(M, Gt, -1, 3)
    n = ps.shape[2]
    d = torch.cdist(ps.reshape(M, G * n, 3), ts.reshape(M, Gt * n, 3)).reshape(M, G, n, Gt, n)
    cost = d.min(4).values.mean(2) + d.min(2).values.mean(-1)  # (M, G, Gt)
    if exist_logit is not None:  # --part_match_exist: prefer parts that already exist (fewer assignment swaps)
        cost = cost - F.logsigmoid(exist_logit)[:, :, None]
    cost = cost.cpu().numpy()
    valid = gr["valid"].cpu().numpy()
    mi, pi, ti = [], [], []
    for m in range(M):
        cols = np.flatnonzero(valid[m])
        r, c = linear_sum_assignment(cost[m][:, cols])
        mi += [m] * len(r)
        pi += r.tolist()
        ti += cols[c].tolist()
    return tuple(torch.as_tensor(x, dtype=torch.long, device=DEV) for x in (mi, pi, ti))


def frustum_iou(x, ia, ib):
    """IoU of the occupancies ia, ib (M, n) over the points x (M, n, 3) inside the camera frustum; NaN for pairs with
    no occupied point there."""
    f = D.frustum_distance(x) == 0
    u = ((ia | ib) & f).sum(1)
    return torch.where(u > 0, ((ia & ib) & f).sum(1) / u.clamp_min(1), torch.nan)


@torch.no_grad()
def metrics(pred, fr):
    sig = match(pred, fr)
    v = fr["pvalid"]
    am = assign_scores(pred)[v].argmax(1)
    maj = fr["share"][v].argmax(1)
    ious = []
    for m in range(fr["M"]):
        a, b = am == sig[m], maj == m
        if b.any():
            ious.append(float((a & b).sum() / (a | b).sum()))
    ex = torch.sigmoid(pred["exist_logit"])
    ms = sq_of(pred, sig)
    gr = fr["groups"]
    s_own = C.sdf_pairs(fr["pts"][:, None], {k: w[fr["pts_lab"]] for k, w in ms.items()})[:, 0].abs()
    act = torch.nonzero(ex > 0.5)[:, 0]
    if len(act):
        s_any = torch.cat([C.sdf(fr["pts"][None, a:a + 8192], {k: w[None] for k, w in sq_of(pred, act).items()})[0].abs().min(0).values
                           for a in range(0, len(fr["pts"]), 8192)])
    else:
        s_any = torch.full((1,), float("nan"))
    extra = {}
    if "parts" in pred:  # geometry = the union of each slot's active parts (part exist > 0.5; the best part if none)
        Pm = {k: pred["parts"][k][sig] for k in C.KEYS}
        pex = pred["parts"]["exist_logit"][sig].sigmoid()
        pact = pex > 0.5
        pact[torch.arange(len(pex)), pex.argmax(1)] |= ~pact.any(1)
        ga = {"sq": Pm, "valid": pact}
        pi, pi_fr = group_pair_iou(ga, gr)
        mi, pj, tj = part_match(Pm, gr)
        _, part_fr = group_pair_iou(single({k: v[mi, pj] for k, v in Pm.items()}), single({k: v[mi, tj] for k, v in gr["sq"].items()}))
        G = pex.shape[1]
        pg = C.surf({k: v.flatten(0, 1) for k, v in Pm.items()}, N_SURF, "cube").reshape(len(pex), -1, 3)
        pmask = pact[:, :, None].expand(-1, -1, pg.shape[1] // G).reshape(len(pex), -1)
        pcd = weighted_chamfer(torch.where(pmask[..., None], pg, 1e3), torch.where(gr["keep"][..., None], gr["pts"], 1e3),
                               pmask.float(), gr["keep"].float())
        lab = fr["pts_lab"]
        dd = C.sdf(fr["pts"][:, None], {k: v[lab] for k, v in Pm.items()})[:, :, 0].abs()
        s_own = torch.where(pact[lab], dd, torch.inf).min(1).values
        allex = pred["parts"]["exist_logit"].sigmoid() > 0.5
        sel = (ex > 0.5)[:, None] & allex
        if sel.any():
            fl = {k: pred["parts"][k][sel] for k in C.KEYS}
            s_any = torch.cat([C.sdf(fr["pts"][None, a:a + 8192], {k: w[None] for k, w in fl.items()})[0].abs().min(0).values
                               for a in range(0, len(fr["pts"]), 8192)])
        else:
            s_any = torch.full((1,), float("nan"))
        extra = {"part_iou_fr": float(part_fr.nanmean()), "parts_active": float(pact.sum(1).float().mean()),
                 "part_targets": float(gr["valid"].sum(1).float().mean())}
    else:
        pg = C.surf(ms, 16, "grid")
        pi, pi_fr = pair_iou_mc(ms, gr)
        pcd = weighted_chamfer(pg, torch.where(gr["keep"][..., None], gr["pts"], 1e3), torch.ones(pg.shape[:2], device=DEV),
                               gr["keep"].float())
    return extra | {"mask_miou": float(np.mean(ious)), "ari": float(adjusted_rand_score(maj.cpu().numpy(), am.cpu().numpy())),
            "n_active": int(len(act)), "targets": fr["M"], "matched_active": float((ex[sig] > 0.5).float().mean()),
            "unmatched_active": int(len(act) - (ex[sig] > 0.5).sum()), "pair_iou_full": float(pi.nanmean()),
            "pair_iou_fr": float(pi_fr.nanmean()),
            "pair_cd": float(pcd.mean()),
            "sdf_own": float(s_own.mean()), "sdf_any": float(s_any.mean()),
            "used_slots": int(torch.unique(am).numel())}


MAIN = ("mask_miou", "pair_iou_fr", "sdf_any", "matched_active", "unmatched_active")  # dashboard: main/, rest other/


def save(pred, fr, sig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"anchor": pred["anchor"].detach().cpu().numpy()} if "anchor" in pred else {}
    # attn: the mask probabilities (m2f: times existence, so its argmax is the patch's query as in the metrics)
    attn = probs(pred) * (pred["exist_logit"].sigmoid()[None] if pred.get("sigmoid") else 1)
    if "parts" in pred:
        extra |= {f"p_{k}": pred["parts"][k].detach().cpu().numpy() for k in C.KEYS}
        extra["p_exist"] = pred["parts"]["exist_logit"].sigmoid().detach().cpu().numpy()
    np.savez_compressed(path, **extra, **{k: pred[k].detach().cpu().numpy() for k in C.KEYS},
                        exist=torch.sigmoid(pred["exist_logit"]).detach().cpu().numpy(),
                        attn=attn.detach().cpu().numpy().astype(np.float16), sig=sig.cpu().numpy(),
                        lab=fr["lab"].cpu().numpy())


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", default="overfit 1 frame")
    ap.add_argument("--frame", default="room_2/000500")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-3, help="input projection + Slot Attention / M2F decoder")
    ap.add_argument("--lr_heads", type=float, default=None, help="SQ + existence heads (and slot self-attention); default --lr")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--sa_layers", type=int, default=0)
    ap.add_argument("--anchor", default="centroid", choices=["centroid", "none"])
    ap.add_argument("--losses", default=DEFAULT_LOSSES)
    ap.add_argument("--assign", default="gt", choices=["gt", "pred"])
    ap.add_argument("--mask_source", default="sdf", choices=["sdf", "sdf_inst", "proj", "instance", "inst1", "instance_proj"])
    ap.add_argument("--w_occ", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="slot", choices=["slot", "m2f"], help="Slot Attention or Mask2Former-style queries")
    ap.add_argument("--parts", type=int, default=0, help="G part SQs per slot (set decoder; 0: one SQ per slot)")
    ap.add_argument("--part_targets", default="all", choices=["all", "visible"], help="target SQs of an instance")
    # defaults since 2026-11-08 (exp 27's collapse study, results/README.md): before, 0 / 1e-8 / off
    ap.add_argument("--exist_smooth", type=float, default=0.01, help="label smoothing of the existence BCEs")
    ap.add_argument("--adam_eps", type=float, default=1e-6)
    ap.add_argument("--part_match_exist", action=argparse.BooleanOptionalAction, default=True,
                    help="part matching cost + -log sigmoid(part exist) (fewer assignment swaps)")
    ap.add_argument("--gt_masks", action="store_true", help="diagnostic: slots = GT-mask-pooled tokens, no Slot Attention")
    ap.add_argument("--exist_detach", action="store_true", help="existence head on detached slot features")
    ap.add_argument("--geo_owned", action="store_true", help="geometry losses only for targets whose slot owns a patch")
    ap.add_argument("--distance", default="scaled", choices=["scaled", "radial"], help="SQ distance everywhere (common.sdf)")
    A = ap.parse_args()
    torch.manual_seed(A.seed), np.random.seed(A.seed)
    C.DISTANCE = A.distance
    scene, fid = A.frame.split("/")
    fr = D.load_frame(scene, fid, A.mask_source, A.part_targets)
    print(f"{fr['name']}: {fr['M']} targets (of {fr['n_all']} in the frustum), {len(fr['pts'])} labelled points", flush=True)
    model = Net(A.d, A.K, A.T, A.sa_layers, A.anchor, A.exist_detach, A.arch, A.parts).to(DEV)
    share = fr["share"] if A.gt_masks else None
    fwd = lambda: model(fr["dino"], fr["xyz"], share)
    heads = [p for n, p in model.named_parameters() if n.split(".")[0] in ("scale", "shape", "rot", "trans", "exist", "out_norm", "layers", "part_dec")]
    hid = {id(p) for p in heads}
    rest = [p for p in model.parameters() if id(p) not in hid]
    opt = torch.optim.Adam([{"params": rest, "lr": A.lr}, {"params": heads, "lr": A.lr_heads or A.lr}], eps=A.adam_eps)
    trackio.init(project=PROJECT, name=A.name, group=A.group, config=vars(A) | {"targets": fr["M"]})
    (OUT / A.name).mkdir(parents=True, exist_ok=True)
    (OUT / A.name / "args.json").write_text(json.dumps(vars(A)))
    t0 = time.time()
    hist = []
    skipped = 0
    best = (-1.0, 0, None)
    for step in range(1, A.steps + 1):
        pred = fwd()
        sig = match(pred, fr)
        parts = losses(pred, fr, sig)
        loss = sum(parts.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):  # skip the update (a NaN would poison Adam's state), count it
            skipped += 1
            opt.zero_grad(set_to_none=True)
            print(f"step {step}: non-finite gradient, skipped ({skipped} so far)", flush=True)
            continue
        opt.step()
        if not torch.isfinite(loss):
            print(f"non-finite loss at step {step}: {parts}", flush=True)
            break
        log = {"main/train_loss": float(loss.detach()), "other/grad_norm": float(gn),
               **{f"loss/{k}": float(v) for k, v in parts.items()}}
        if step % A.eval_every == 0 or step == A.steps:
            model.eval()
            with torch.no_grad():
                pe = fwd()
                m = metrics(pe, fr)
                save(pe, fr, match(pe, fr), OUT / A.name / "steps" / f"{step:06d}.npz")  # for the viewer
            model.train()
            log |= {f"{'main' if k in MAIN else 'other'}/{k}": v for k, v in m.items()}
            hist.append({"step": step, **m})
            score = m.get("pair_iou_fr")
            if score is not None and score == score and score > best[0]:  # keep the best eval state (late collapses)
                best = (score, step, {k: v.detach().clone() for k, v in model.state_dict().items()})
            print(f"step {step} loss {float(loss):.3f} " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                                  for k, v in m.items()) + f" {time.time() - t0:.0f}s", flush=True)
        trackio.log(log, step=step)
    trackio.finish()
    model.eval()
    with torch.no_grad():
        pred = fwd()
        sig = match(pred, fr)
    save(pred, fr, sig, OUT / A.name / "pred.npz")
    torch.save(model.state_dict(), OUT / A.name / "model.pt")
    if best[2] is not None:
        torch.save(best[2], OUT / A.name / "best.pt")
    res = {"name": A.name, "group": A.group, "args": vars(A), "final": hist[-1] if hist else None, "history": hist,
           "seconds": time.time() - t0, "skipped_steps": skipped,
           "best": {"step": best[1], "pair_iou_fr": best[0]}}
    (EXP / "results" / "runs").mkdir(parents=True, exist_ok=True)
    (EXP / "results" / "runs" / f"{A.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
