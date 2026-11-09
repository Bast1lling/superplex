"""Amodal targets from the frame's SuperFlex SQs (all of an instance's SQs that reach into the frustum, visible or not).

- groups_all: exp 27's target groups with part_targets "all" (same target order as the visible groups), plus
  `vis` (M, G): member has >= 1 visible cube sample (in the frustum, not behind the observed depth).
- gt_box: per target, the gravity-aligned box of the union surface of its SQs: z extent = min / max height,
  yaw + horizontal extents = the minimum-area rectangle (searched over 0..90 deg in 1 deg steps). Camera frame:
  centre c (M, 3), rotation R (M, 3, 3) (box axes as columns), half extents s (M, 3).
- box_loss: mean corner distance of the slot boxes (scale / rotate / trans of the slot heads) to the target boxes,
  min over the 4 equivalent descriptions of the target (yaw + k 90 deg, horizontal extents swapped for odd k).
- part_losses_all: exp 27's part_cd + part_exist against groups_all; pairs matched to a hidden member weighted
  --w_hidden (visible ones 1).
"""

import math

import torch
import torch.nn.functional as F

import common as C
import data as D
import train as T27  # exp 27's (weighted_chamfer, part_match, smooth, A)
from common import DEV

SIGNS = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32, device=DEV)


@torch.no_grad()
def visible_members(gr, depth):
    M, G = gr["valid"].shape
    S = C.surf({k: v.flatten(0, 1) for k, v in gr["sq"].items()}, D.N_SURF, "cube")
    nvis = (D.sample_weights(S, depth, 0.0, 1e-9) > 0.5).sum(1).reshape(M, G)
    return (nvis > 0) & gr["valid"]


@torch.no_grad()
def gt_box(gr, R_g):
    """gr: target groups (pts (M, n, 3) camera frame, keep (M, n)); R_g (3, 3) camera -> gravity."""
    p = gr["pts"] @ R_g.T  # gravity frame
    k = gr["keep"]
    inf = torch.tensor(float("inf"), device=DEV)
    zmin = torch.where(k, p[..., 2], inf).min(1).values
    zmax = torch.where(k, p[..., 2], -inf).max(1).values
    th = torch.deg2rad(torch.arange(90, device=DEV, dtype=torch.float32))  # (A,)
    c, s = torch.cos(th), torch.sin(th)
    u = p[..., 0, None] * c + p[..., 1, None] * s  # (M, n, A): coordinates along the rotated axes
    v = -p[..., 0, None] * s + p[..., 1, None] * c
    kk = k[..., None]
    umin, umax = torch.where(kk, u, inf).min(1).values, torch.where(kk, u, -inf).max(1).values  # (M, A)
    vmin, vmax = torch.where(kk, v, inf).min(1).values, torch.where(kk, v, -inf).max(1).values
    a = ((umax - umin) * (vmax - vmin)).argmin(1)  # (M,)
    i = torch.arange(len(a), device=DEV)
    cu, cv = (umin[i, a] + umax[i, a]) / 2, (vmin[i, a] + vmax[i, a]) / 2
    ca, sa = c[a], s[a]
    centre_g = torch.stack([cu * ca - cv * sa, cu * sa + cv * ca, (zmin + zmax) / 2], 1)
    half = torch.stack([(umax[i, a] - umin[i, a]) / 2, (vmax[i, a] - vmin[i, a]) / 2, (zmax - zmin) / 2], 1)
    z, o = torch.zeros_like(ca), torch.ones_like(ca)
    Rb = torch.stack([ca, -sa, z, sa, ca, z, z, z, o], -1).reshape(-1, 3, 3)  # box axes in the gravity frame
    return {"c": centre_g @ R_g, "R": R_g.T @ Rb, "s": half.clamp_min(0.005)}


def corners(c, R, s):
    """(M, 3), (M, 3, 3), (M, 3) -> (M, 8, 3)."""
    return c[:, None] + torch.einsum("mij,mkj->mki", R, SIGNS[None] * s[:, None])


def box_loss(c, R, s, box):
    """mean corner distance, min over the target's 4 equivalent descriptions -> (M,)."""
    pc = corners(c, R, s)
    out = []
    for k in range(4):
        a = k * math.pi / 2
        Rz = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], device=DEV)
        sk = box["s"][:, [1, 0, 2]] if k % 2 else box["s"]
        out.append((pc - corners(box["c"], box["R"] @ Rz, sk)).norm(dim=-1).mean(1))
    return torch.stack(out).min(0).values


def part_losses_all(pred, fr, sig, w_hidden):
    A = T27.A
    gr = fr["groups_all"]
    M = fr["M"]
    Pm = {k: pred["parts"][k][sig] for k in C.KEYS}
    el = pred["parts"]["exist_logit"][sig]
    mi, pi, ti = T27.part_match({k: v.detach() for k, v in Pm.items()}, gr, el.detach() if A.part_match_exist else None)
    w = torch.where(fr["vis_all"][mi, ti], 1.0, w_hidden)
    pp = C.surf({k: v[mi, pi] for k, v in Pm.items()}, D.N_SURF, "cube")
    Gt = gr["valid"].shape[1]
    tp = gr["pts"].reshape(M, Gt, -1, 3)[mi, ti]
    wp = D.sample_weights(pp.detach(), fr["depth"], A.w_occ, A.lam)
    wt = D.sample_weights(tp, fr["depth"], A.w_occ, A.lam)
    out = {"part_cd": (T27.weighted_chamfer(pp, tp, wp, wt) * w).sum() / w.sum().clamp_min(1e-6)}
    tgt = torch.zeros_like(el)
    tgt[mi, pi] = 1
    we = torch.ones_like(el)
    we[mi, pi] = w
    out["part_exist"] = (F.binary_cross_entropy_with_logits(el, T27.smooth(tgt), reduction="none") * we).sum() / we.sum()
    return out


@torch.no_grad()
def box_metrics(pred, fr):
    """corner error (m) and yaw error (deg, modulo 90) of the matched slots' boxes."""
    sig = T27.match(pred, fr)
    b = fr["box"]
    err = box_loss(pred["trans"][sig], pred["rotate"][sig], pred["scale"][sig], b)
    fa = pred["rotate"][sig][:, :, 0]  # box x axis (camera frame)
    ga = b["R"][:, :, 0]
    cosang = (fa * ga).sum(1).clamp(-1, 1)
    ang = torch.rad2deg(torch.arccos(cosang)) % 90
    ang = torch.minimum(ang, 90 - ang)
    return {"box_corner_err": float(err.mean()), "box_yaw_err": float(ang.median())}
