"""Geometry of plain SQs, rendering and matching, shared by data / model / train (written from scratch for exp 24).

Camera: the /4 view of the NICE-SLAM Replica renderings as cached by exp 22 (168 x 300 pixels, rows 4..676 of the
full image, fx = fy = 150), 42 x 75 patches of 4 x 4 pixels (16 x 16 at full resolution, one DINOv3 token each).
Camera frame: x right, y down, z forward.

SQs are dicts {scale (.., 3) half-axes in m, shape (.., 2) exponents (eps1 along the z axis / latitude, eps2 in the
xy plane), rotate (.., 3, 3) columns = SQ axes, trans (.., 3)}.
"""

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

H, W = 168, 300
FX = FY = 150.0
CX, CY = (599.5 - 2) / 4, (339.5 - 2) / 4 - 1
GH, GW, P = 42, 75, 4
NP = GH * GW
DEV = "cuda"
SAMPLING = "grid"  # --sampling: surface samples of the losses / matching cost ("grid" or "cube"); metrics always grid
KEYS = ("scale", "shape", "rotate", "trans")


def sq(scale, shape, rotate, trans):
    return {"scale": scale, "shape": shape, "rotate": rotate, "trans": trans}


def tsq(t, idx=None):
    """Target dict (scale, exponents, rotation, translation) -> SQ dict, optionally indexed."""
    d = sq(t["scale"], t["exponents"], t["rotation"], t["translation"])
    return d if idx is None else {k: v[idx] for k, v in d.items()}


def transform(s, T):
    """SQs s (dict of (..., )) moved by the rigid transform T (4, 4) (or broadcastable (..., 4, 4))."""
    R, t = T[..., :3, :3], T[..., :3, 3]
    return s | {"rotate": R @ s["rotate"], "trans": (R @ s["trans"][..., None])[..., 0] + t}


def quat2mat(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def fourier(x, n, scale):
    f = (2.0 ** torch.arange(n, device=x.device)) * math.pi / scale
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


# ----------------------------------------------------------------------------------------------- surface samples

def spow(x, p):
    return torch.sign(x) * x.abs().clamp_min(1e-6) ** p


@lru_cache(maxsize=8)
def _angles(n, device):
    eta = (torch.arange(n, device=device) + 0.5) / n * torch.pi - torch.pi / 2
    omega = (torch.arange(n, device=device) + 0.5) / n * 2 * torch.pi - torch.pi
    e, o = torch.meshgrid(eta, omega, indexing="ij")
    return e.reshape(-1), o.reshape(-1)


def surface_grid(scale, shape, rot, trans, n):
    """Regular (eta, omega) grid, n x n samples (SuperFlex's parametric sampling) -> (S, n * n, 3)."""
    e, o = _angles(n, scale.device)
    e1, e2 = shape[:, :1], shape[:, 1:]
    ce = spow(e.cos(), e1)
    p = torch.stack([scale[:, :1] * ce * spow(o.cos(), e2), scale[:, 1:2] * ce * spow(o.sin(), e2),
                     scale[:, 2:] * spow(e.sin(), e1)], -1)
    return p @ rot.transpose(1, 2) + trans[:, None]


@lru_cache(maxsize=8)
def _cube_dirs(m, device):
    g = (torch.arange(m, device=device) + 0.5) / m * 2 - 1
    a, b = torch.meshgrid(g, g, indexing="ij")
    one = torch.ones_like(a)
    return torch.stack([torch.stack(f, -1) for f in ((one, a, b), (-one, b, a), (b, one, a), (a, -one, b),
                                                     (a, b, one), (b, a, -one))]).reshape(-1, 3)


def surface_cube(scale, shape, rot, trans, n):
    """A cube grid (6 m^2 ~ n^2 samples) projected radially onto the SQ: x = s * u * F(u)^(-eps1 / 2), F the
    inside-outside function. Near-uniform on the faces of boxy SQs (the grid leaves their pole caps empty)."""
    m = max(2, round(n / 6 ** 0.5))
    u = _cube_dirs(m, scale.device)  # (N, 3)
    e1, e2 = shape[:, :1], shape[:, 1:]
    ax = u.abs().clamp_min(1e-9)
    G = (ax[:, 0] ** (2 / e2) + ax[:, 1] ** (2 / e2)).clamp_min(1e-30)
    Fv = G ** (e2 / e1) + ax[:, 2] ** (2 / e1)
    loc = u * Fv[..., None] ** (-e1[..., None] / 2) * scale[:, None]
    return loc @ rot.transpose(1, 2) + trans[:, None]


def surface(scale, shape, rot, trans, n, kind=None):
    return (surface_cube if (kind or SAMPLING) == "cube" else surface_grid)(scale, shape, rot, trans, n)


def surf(s, n, kind=None):
    return surface(s["scale"], s["shape"], s["rotate"], s["trans"], n, kind)


# ------------------------------------------------------------------------------------------------------- SDF

def sdf(x, s):
    """SuperFlex's radial SDF of plain SQs, in log space (no overflow for small exponents): x (B, M, 3), s dict of
    (B, N, ...) -> (B, N, M). Its sign is exact, its magnitude the radial distance."""
    X = torch.einsum("bnji,bnmj->bnmi", s["rotate"], x[:, None] - s["trans"][:, :, None])
    l = (X.abs().clamp_min(1e-6) / s["scale"][:, :, None]).log()
    e1, e2 = s["shape"][..., :1], s["shape"][..., 1:]
    lxy = torch.logaddexp(2 / e2 * l[..., 0], 2 / e2 * l[..., 1]) * (e2 / e1)
    lF = torch.logaddexp(lxy, 2 / e1 * l[..., 2])
    return X.norm(dim=-1) * (1 - torch.exp(-e1 / 2 * lF))


def sdf_pairs(x, s):
    """x (S, M, 3), s dict of S SQs -> (S, M): SDF of SQ i at its own points."""
    return sdf(x, {k: v[:, None] for k, v in s.items()})[:, 0]


@torch.no_grad()
def inside(x, s, chunk=64):
    out = [sdf(x, {k: v[:, a:a + chunk] for k, v in s.items()}) < 0 for a in range(0, s["scale"].shape[1], chunk)]
    return torch.cat(out, 1) if out else torch.zeros(x.shape[0], 0, x.shape[1], dtype=torch.bool, device=x.device)


@torch.no_grad()
def inside_pairs(x, s):
    return sdf_pairs(x, s) < 0


@torch.no_grad()
def pair_iou(a, b, M=2048):
    """Volumetric IoU of SQ pairs a[i], b[i]: M points uniform in the box around both."""
    both = torch.cat([surf(a, 8, "grid"), surf(b, 8, "grid")], 1)
    lo, hi = both.min(1).values, both.max(1).values
    x = lo[:, None] + torch.rand(len(lo), M, 3, device=lo.device) * (hi - lo)[:, None]
    ia, ib = inside_pairs(x, a), inside_pairs(x, b)
    return (ia & ib).sum(1) / (ia | ib).sum(1).clamp_min(1)


@torch.no_grad()
def pair_iou_matrix(a, b, M=1024):
    """IoU of every SQ of a with every SQ of b -> (A, B); pairs whose sample boxes do not overlap are 0."""
    A, Bn = len(a["scale"]), len(b["scale"])
    if A == 0 or Bn == 0:
        return torch.zeros(A, Bn, device=DEV)
    box = lambda q: (lambda p: (p.min(1).values, p.max(1).values))(surf(q, 8, "grid"))
    (la, ha), (lb, hb) = box(a), box(b)
    pa, pb = (ha - la) * 0.1 + 0.01, (hb - lb) * 0.1 + 0.01
    hit = (((la - pa)[:, None] <= (hb + pb)[None]) & ((lb - pb)[None] <= (ha + pa)[:, None])).all(-1)
    ai, bi = torch.nonzero(hit, as_tuple=True)
    out = torch.zeros(A, Bn, device=la.device)
    for s in range(0, len(ai), 4096):
        i, j = ai[s:s + 4096], bi[s:s + 4096]
        lo, hi = torch.minimum(la[i], lb[j]), torch.maximum(ha[i], hb[j])
        x = lo[:, None] + torch.rand(len(i), M, 3, device=lo.device) * (hi - lo)[:, None]
        ina = inside_pairs(x, {k: v[i] for k, v in a.items()})
        inb = inside_pairs(x, {k: v[j] for k, v in b.items()})
        out[i, j] = (ina & inb).sum(1) / (ina | inb).sum(1).clamp_min(1)
    return out


def chamfer(a, b):
    d = torch.cdist(a, b)
    return d.min(2).values.mean(1) + d.min(1).values.mean(1)


# --------------------------------------------------------------------------------------------------- matching

C_CD, C_CLS, C_EXIST = 10.0, 2.0, 1.0


@torch.no_grad()
def match_cost(pp, tp, prob, exist):
    """Hungarian matching from surface samples pp (Q, n, 3), tp (T, n, 3), prob (Q, T), exist (Q,)."""
    Q, n = pp.shape[:2]
    T = len(tp)
    tpf = tp.reshape(-1, 3)
    cd = []
    for a in range(0, Q, 512):
        d = torch.cdist(pp[a:a + 512].reshape(-1, 3), tpf).reshape(-1, n, T, n)
        cd.append(d.min(3).values.mean(1) + d.min(1).values.mean(2))
    cost = C_CD * torch.cat(cd) + C_CLS * (1 - prob) + C_EXIST * (1 - exist)[:, None]
    return linear_sum_assignment(cost.cpu().numpy())


# --------------------------------------------------------------------------------------------------- rendering
# Differentiable ray casting of plain SQs (the algorithm of exp 22's render.py, reimplemented): per (ray, SQ) pair
# a box cull, K SDF samples along the ray (argmin + entry by bisection, no grad), then alpha = exist *
# sigmoid(-SDF(argmin) / tau) (envelope theorem) and the entry depth with one Newton step (implicit gradient);
# front-to-back compositing per ray.

R_K, R_BISECT, TAU_PX, ZNEAR, MIN_EXIST, EPS = 24, 6, 0.5, 0.1, 0.02, 1e-3


@lru_cache(maxsize=8)
def rays(stride, device):
    v = torch.arange(stride // 2, H, stride, device=device)
    u = torch.arange(stride // 2, W, stride, device=device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    d = torch.stack([(uu - CX) / FX, (vv - CY) / FY, torch.ones_like(uu, dtype=torch.float32)], -1)
    return d.reshape(-1, 3).float(), vv.reshape(-1), uu.reshape(-1)


def _sdf_rows(s, si, x, chunk=200_000):
    out = [sdf_pairs(x[a:a + chunk], {k: v[si[a:a + chunk]] for k, v in s.items()}) for a in range(0, len(si), chunk)]
    return torch.cat(out) if out else x.new_zeros(0, x.shape[1])


@torch.no_grad()
def _cull(s, d, stride):
    R, c = s["rotate"], s["trans"]
    margin = 4 * TAU_PX * stride * c[:, 2].clamp_min(ZNEAR) / FX
    h = s["scale"] + margin[:, None]
    o = -(R.transpose(1, 2) @ c[:, :, None])[:, :, 0]
    dl = torch.einsum("nji,rj->rni", R, d)
    dl = torch.where(dl.abs() < 1e-9, torch.full_like(dl, 1e-9), dl)
    ta, tb = (-h - o)[None] / dl, (h - o)[None] / dl
    t0 = torch.minimum(ta, tb).amax(-1).clamp_min(ZNEAR)
    t1 = torch.maximum(ta, tb).amin(-1)
    ri, si = torch.nonzero(t1 > t0, as_tuple=True)
    return ri, si, t0[ri, si], t1[ri, si]


@torch.no_grad()
def _search(s, d, ri, si, t0, t1, chunk=400_000):
    if len(ri) > chunk:
        parts = [_search(s, d, ri[a:a + chunk], si[a:a + chunk], t0[a:a + chunk], t1[a:a + chunk]) for a in range(0, len(ri), chunk)]
        return tuple(torch.cat(x) for x in zip(*parts))
    k = (torch.arange(R_K, device=d.device) + 0.5) / R_K
    ts = t0[:, None] + (t1 - t0)[:, None] * k
    v = _sdf_rows(s, si, d[ri][:, None] * ts[..., None])
    step = (t1 - t0) / R_K
    tm = ts.gather(1, v.argmin(1)[:, None])[:, 0]
    ts2 = (tm[:, None] + step[:, None] * torch.linspace(-1, 1, 9, device=d.device)).clamp(t0[:, None], t1[:, None])
    tm = ts2.gather(1, _sdf_rows(s, si, d[ri][:, None] * ts2[..., None]).argmin(1)[:, None])[:, 0]
    neg = v < 0
    hit = neg.any(1)
    first = neg.float().argmax(1)
    hi = ts.gather(1, first[:, None])[:, 0]
    lo = torch.where(first > 0, ts.gather(1, (first - 1).clamp_min(0)[:, None])[:, 0], t0)
    for _ in range(R_BISECT):
        mid = (lo + hi) / 2
        ins = _sdf_rows(s, si, (d[ri] * mid[:, None])[:, None])[:, 0] < 0
        hi, lo = torch.where(ins, mid, hi), torch.where(ins, lo, mid)
    te = torch.where(hit & (first == 0), t0, hi)
    return tm, hit, torch.where(hit, te, t1)


def render(s, exist, stride=2, grad=True):
    """N SQs in the camera frame with existences (N,) -> {"w": (R, N) compositing weights, "z": (R, N), "depth",
    "cover": (R,), "v", "u": pixel of each ray}."""
    d, vv, uu = rays(stride, exist.device)
    Rn, N = len(d), len(exist)
    ri, si, t0, t1 = _cull(s, d, stride)
    tm, hit, te = _search(s, d, ri, si, t0, t1)
    with (torch.enable_grad() if grad else torch.no_grad()):
        dm = _sdf_rows(s, si, (d[ri] * tm[:, None])[:, None])[:, 0]
        hp = torch.nonzero(hit)[:, 0]
        z = tm.clone()
        if len(hp):
            se = _sdf_rows(s, si[hp], (d[ri[hp]] * te[hp, None])[:, None])[:, 0]
            with torch.no_grad():
                h = 5e-3
                sp = _sdf_rows(s, si[hp], (d[ri[hp]] * (te[hp] + h)[:, None])[:, None])[:, 0]
                sm = _sdf_rows(s, si[hp], (d[ri[hp]] * (te[hp] - h)[:, None])[:, None])[:, 0]
                g = ((sp - sm) / (2 * h)).clamp(max=-0.05)
            z = z.index_put((hp,), te[hp] - se / g)
        tau = TAU_PX * stride * z.detach().clamp_min(ZNEAR) / FX
        alpha = (exist[si] * torch.sigmoid(-dm / tau)).clamp(max=0.99)
        A = torch.zeros(Rn, N, device=d.device, dtype=alpha.dtype).index_put((ri, si), alpha)
        Z = torch.zeros(Rn, N, device=d.device, dtype=z.dtype).index_put((ri, si), z)
        with torch.no_grad():
            order = torch.where(A > 0, Z.detach(), torch.full_like(Z, torch.inf)).argsort(1)
        As = A.gather(1, order)
        tr = torch.cumprod(torch.cat([torch.ones_like(As[:, :1]), 1 - As[:, :-1]], 1), 1)
        w = torch.zeros_like(A).scatter(1, order, As * tr)
        cover = w.sum(1)
        depth = (w * Z).sum(1) / cover.clamp_min(EPS)
    return {"w": w, "z": Z, "depth": depth, "cover": cover, "v": vv, "u": uu}


# --------------------------------------------------------------------------------------------- patch helpers

def group_patches(x, B):
    """(B, H, W, C) -> (B, GH, GW, P * P, C)."""
    return x.reshape(B, GH, P, GW, P, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, GH, GW, P * P, -1)


@lru_cache(maxsize=4)
def patch_rays(device):
    """(NP, 3) ray direction (z = 1) through every patch center."""
    gy, gx = torch.meshgrid(torch.arange(GH, device=device), torch.arange(GW, device=device), indexing="ij")
    return torch.stack([((gx + 0.5) * P - 0.5 - CX) / FX, ((gy + 0.5) * P - 0.5 - CY) / FY,
                        torch.ones_like(gx, dtype=torch.float32)], -1).reshape(-1, 3).float()


def patch_centroids(depth):
    """depth (B, H, W) -> centroid (B, NP, 3) of each patch's back-projected pixels (3 m along the ray if none),
    has (B, NP) bool, median-free mean depth (B, NP)."""
    B = depth.shape[0]
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    xyz = group_patches(torch.stack([(u - CX) / FX * depth, (v - CY) / FY * depth, depth], -1), B)
    valid = group_patches((depth > 0).float()[..., None], B)
    n = valid.sum(3)
    c = ((xyz * valid).sum(3) / n.clamp_min(1)).reshape(B, NP, 3)
    has = (n[..., 0] > 0).reshape(B, NP)
    c = torch.where(has[..., None], c, patch_rays(depth.device)[None] * 3.0)
    return c, has


def peaks(exist, V):
    """exist (B, V * NP) -> non-3x3-maxima set to -1, per view (CenterNet NMS)."""
    e = exist.reshape(-1, 1, GH, GW)
    keep = e == F.max_pool2d(e, 3, 1, 1)
    return torch.where(keep, e, -1.0).reshape(exist.shape)


def instance_share(label, inst):
    """label (H, W) per pixel instance, inst (T,) -> (T, NP) share of each instance's pixels per patch."""
    m = (label[None] == inst[:, None, None]).float()
    return F.avg_pool2d(m[:, None], P)[:, 0].flatten(1)
