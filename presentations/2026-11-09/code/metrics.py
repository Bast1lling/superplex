"""Metrics per frame (exp 27's), per target m with its matched slot sig[m]:

- mask IoU: the slot's patches (argmax over the slots) vs. the target's (majority label), mean over targets (mIoU);
  ARI of the two patch partitions.
- frustum IoU (main metric): volumetric IoU, inside the camera frustum, of the prediction (the slot's SQ, or the
  union of its active parts: part existence > 0.5, the most existing one if none) and the target (the union of its
  SQs). Estimated by importance sampling (`box_samples`): n points uniform in the oriented box of every SQ of both
  sides; each point weighted 1 / q(x) with q the mixture density of these boxes. A plain SQ lies inside its box, so
  this is unbiased, and unlike uniform points in a common box it samples thin slabs (walls, table tops) densely.
  IoU = sum w 1[in pred & in target & in frustum] / sum w 1[(in pred | in target) & in frustum].
- part IoU: the same per (part, target SQ) pair of the part matching, mean over pairs.
"""

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

import data as D
import sq
from data import DEV
from losses import match, part_match


def inside(x, g):
    """x (M, n, 3) -> (M, n): inside the union of group m's valid SQs."""
    return ((sq.sdf(x, g["sq"]) < 0) & g["valid"][:, :, None]).any(1)


def single(s):
    return {"sq": {k: v[:, None] for k, v in s.items()}, "valid": torch.ones(len(s["scale"]), 1, dtype=torch.bool, device=DEV)}


@torch.no_grad()
def box_samples(ga, gb, n):
    """n points uniform in the box of every valid SQ of groups ga[m], gb[m] -> x (M, B n, 3), weights (M, B n)."""
    sc = torch.cat([ga["sq"]["scale"], gb["sq"]["scale"]], 1)
    R = torch.cat([ga["sq"]["rotate"], gb["sq"]["rotate"]], 1)
    t = torch.cat([ga["sq"]["trans"], gb["sq"]["trans"]], 1)
    bv = torch.cat([ga["valid"], gb["valid"]], 1)
    M, B = bv.shape
    u = (torch.rand(M, B, n, 3, device=DEV) * 2 - 1) * sc[:, :, None]
    x = (torch.einsum("mbij,mbnj->mbni", R, u) + t[:, :, None]).reshape(M, B * n, 3)
    xv = bv[:, :, None].expand(M, B, n).reshape(M, B * n)
    loc = torch.einsum("mbji,mbnj->mbni", R, x[:, None] - t[:, :, None])  # every sample in every box's frame
    inbox = (loc.abs() <= sc[:, :, None]).all(-1) & bv[:, :, None]
    q = (inbox.float() / (8 * sc.prod(-1))[:, :, None]).sum(1) / bv.sum(1, keepdim=True).clamp_min(1)  # mixture density
    return x, xv.float() / q.clamp_min(1e-12) / xv.sum(1, keepdim=True).clamp_min(1)


@torch.no_grad()
def frustum_iou(ga, gb, n=4096, chunk=4):
    """IoU per group pair m, `chunk` pairs at a time (memory ~ chunk * boxes^2 * n)."""
    M = len(ga["valid"])
    if M > chunk:
        sub = lambda g, a: {"sq": {k: v[a:a + chunk] for k, v in g["sq"].items()}, "valid": g["valid"][a:a + chunk]}
        return torch.cat([frustum_iou(sub(ga, a), sub(gb, a), n, chunk) for a in range(0, M, chunk)])
    x, w = box_samples(ga, gb, n)
    ia, ib = inside(x, ga), inside(x, gb)
    f = D.frustum_distance(x) == 0
    I, U = (w * (ia & ib & f)).sum(1), (w * ((ia | ib) & f)).sum(1)
    return torch.where(U > 0, I / U.clamp_min(1e-12), torch.nan)


def active_parts(pred, sig):
    pex = pred["parts"]["exist_logit"][sig].sigmoid()
    act = pex > 0.5
    act[torch.arange(len(pex)), pex.argmax(1)] |= ~act.any(1)
    return act


@torch.no_grad()
def metrics(pred, fr, per_target=False):
    sig = match(pred, fr)
    v = fr["pvalid"]
    am, maj = pred["logits"][v].argmax(1), fr["share"][v].argmax(1)
    miou = []
    for m in range(fr["M"]):
        a, b = am == sig[m], maj == m
        miou.append(float((a & b).sum() / (a | b).sum()) if b.any() else np.nan)
    gr = fr["groups"]
    out = {}
    if "parts" in pred:
        Pm = {k: pred["parts"][k][sig] for k in sq.KEYS}
        iou = frustum_iou({"sq": Pm, "valid": active_parts(pred, sig)}, gr)
        mi, pj, tj = part_match(Pm, gr)
        piou = frustum_iou(single({k: v[mi, pj] for k, v in Pm.items()}), single({k: v[mi, tj] for k, v in gr["sq"].items()}))
        out["part_iou_fr"] = float(piou.nanmean())
    else:
        iou = frustum_iou(single({k: pred[k][sig] for k in sq.KEYS}), gr)
    out |= {"mask_miou": float(np.nanmean(miou)), "ari": float(adjusted_rand_score(maj.cpu().numpy(), am.cpu().numpy())),
            "pair_iou_fr": float(iou.nanmean()), "targets": fr["M"],
            "active_slots": int((pred["exist_logit"] > 0).sum())}
    if per_target:
        out["per_target"] = {"mask_iou": miou, "iou_fr": iou.cpu().numpy().tolist(), "slot": sig.cpu().numpy().tolist()}
    return out
