"""Plain superquadrics: rotation, surface samples, signed distance, differentiable rendering (exp 27's common.py).

An SQ is a dict {scale (.., 3) half-axes in m, shape (.., 2) exponents (eps1, eps2), rotate (.., 3, 3) columns = SQ
axes, trans (.., 3)}. Camera frame: x right, y down, z forward, metres.
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from data import CX, CY, FX, FY, H, W, ZNEAR

KEYS = ("scale", "shape", "rotate", "trans")


def quat2mat(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def fourier(x, n=10, lo=0.04, hi=20.0):
    """x (..., D) -> (..., 2 D n): sin / cos at n wavelengths log-spaced in [lo, hi]."""
    wl = torch.logspace(math.log10(lo), math.log10(hi), n, device=x.device)
    a = x[..., None] * (2 * math.pi / wl)
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


# ----------------------------------------------------------------------------------------------- surface samples

@lru_cache(maxsize=8)
def _cube_dirs(m, device):
    g = (torch.arange(m, device=device) + 0.5) / m * 2 - 1
    a, b = torch.meshgrid(g, g, indexing="ij")
    one = torch.ones_like(a)
    return torch.stack([torch.stack(f, -1) for f in ((one, a, b), (-one, b, a), (b, one, a), (a, -one, b),
                                                     (a, b, one), (b, a, -one))]).reshape(-1, 3)


def surf(s, n=16):
    """Surface samples (S, ~n^2, 3): a cube grid projected radially onto each SQ (near-uniform on boxy SQs)."""
    m = max(2, round(n / 6 ** 0.5))
    u = _cube_dirs(m, s["scale"].device)
    e1, e2 = s["shape"][:, :1], s["shape"][:, 1:]
    ax = u.abs().clamp_min(1e-9)
    G = (ax[:, 0] ** (2 / e2) + ax[:, 1] ** (2 / e2)).clamp_min(1e-30)
    Fv = G ** (e2 / e1) + ax[:, 2] ** (2 / e1)
    loc = u * Fv[..., None] ** (-e1[..., None] / 2) * s["scale"][:, None]
    return loc @ s["rotate"].transpose(1, 2) + s["trans"][:, None]


# ------------------------------------------------------------------------------------------------------- SDF

def sdf(x, s):
    """Metric signed distance of points x (B, M, 3) to SQs s (dict of (B, N, ...)) -> (B, N, M). Outside: the
    first-order distance (g - 1) / |grad g| of g = F^(eps1 / 2); inside: the smallest of it and the exact distances
    along the 3 local axes to the surface (exp 27; median error vs. brute force < 1 mm). In log space."""
    X = torch.einsum("bnji,bnmj->bnmi", s["rotate"], x[:, None] - s["trans"][:, :, None])
    ls = s["scale"][:, :, None].log()
    l = X.abs().clamp_min(1e-6).log() - ls
    e1, e2 = s["shape"][..., :1], s["shape"][..., 1:]
    lG = torch.logaddexp(2 / e2 * l[..., 0], 2 / e2 * l[..., 1])
    lF = torch.logaddexp(e2 / e1 * lG, 2 / e1 * l[..., 2])
    a = (e1 / 2 - 1) * lF
    lgx = a + (e2 / e1 - 1) * lG + (2 / e2 - 1) * l[..., 0] - ls[..., 0]
    lgy = a + (e2 / e1 - 1) * lG + (2 / e2 - 1) * l[..., 1] - ls[..., 1]
    lgz = a + (2 / e1 - 1) * l[..., 2] - ls[..., 2]
    lgn = 0.5 * torch.logsumexp(torch.stack([2 * lgx, 2 * lgy, 2 * lgz], -1), -1)
    d = (torch.exp(e1 / 2 * lF) - 1) * torch.exp(-lgn)
    ax, ay, az = l.clamp(max=0.0).exp().unbind(-1)
    sc = s["scale"][:, :, None]
    lo = 1e-12  # bases >= 1e-12 and exponents <= 0 before exp: no NaN gradients (exp 27)
    Gp = torch.exp(((e2 / e1) * lG).clamp(max=0.0))
    cz = torch.exp(((2 / e1) * l[..., 2]).clamp(max=0.0))
    rz = (1 - Gp).clamp_min(lo) ** (e1 / 2)
    rxy = (1 - cz).clamp_min(lo) ** (e1 / e2)
    rx = (rxy - torch.exp(((2 / e2) * l[..., 1]).clamp(max=0.0))).clamp_min(lo) ** (e2 / 2)
    ry = (rxy - torch.exp(((2 / e2) * l[..., 0]).clamp(max=0.0))).clamp_min(lo) ** (e2 / 2)
    dax = torch.stack([(rx - ax) * sc[..., 0], (ry - ay) * sc[..., 1], (rz - az) * sc[..., 2]], -1).clamp_min(0)
    return torch.where(d < 0, -torch.minimum(-d, dax.amin(-1)), d)


def sdf_pairs(x, s):
    """x (S, M, 3), s dict of S SQs -> (S, M): SDF of SQ i at its own points."""
    return sdf(x, {k: v[:, None] for k, v in s.items()})[:, 0]


# --------------------------------------------------------------------------------------------------- rendering
# Differentiable ray casting (exp 27): per (ray, SQ) pair a box cull, 24 SDF samples along the ray (argmin + entry
# by bisection, no gradient), then alpha = sigmoid(-SDF(argmin) / tau) and the entry depth with one Newton step;
# front-to-back compositing per ray.

R_K, R_BISECT, TAU_PX, EPS = 24, 6, 0.5, 1e-3


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
    h = s["scale"] + (4 * TAU_PX * stride * c[:, 2].clamp_min(ZNEAR) / FX)[:, None]
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
    ts = t0[:, None] + (t1 - t0)[:, None] * ((torch.arange(R_K, device=d.device) + 0.5) / R_K)
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


def render(s, stride=2):
    """N SQs (camera frame) -> {"w": (R, N) compositing weights per ray, "depth": (R,), "cover": (R,), "v", "u"}."""
    d, vv, uu = rays(stride, s["scale"].device)
    Rn, N = len(d), len(s["scale"])
    ri, si, t0, t1 = _cull(s, d, stride)
    tm, hit, te = _search(s, d, ri, si, t0, t1)
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
    alpha = torch.sigmoid(-dm / tau).clamp(max=0.99)
    A = torch.zeros(Rn, N, device=d.device, dtype=alpha.dtype).index_put((ri, si), alpha)
    Z = torch.zeros(Rn, N, device=d.device, dtype=z.dtype).index_put((ri, si), z)
    with torch.no_grad():
        order = torch.where(A > 0, Z.detach(), torch.full_like(Z, torch.inf)).argsort(1)
    As = A.gather(1, order)
    tr = torch.cumprod(torch.cat([torch.ones_like(As[:, :1]), 1 - As[:, :-1]], 1), 1)
    w = torch.zeros_like(A).scatter(1, order, As * tr)
    cover = w.sum(1)
    return {"w": w, "depth": (w * Z).sum(1) / cover.clamp_min(EPS), "cover": cover, "v": vv, "u": uu}
