"""Differentiable rendering of plain SQs from the camera (the frame's pose) and the projection losses on it.

Rays: one per STRIDE x STRIDE pixels (84 x 150 at stride 2), direction ((u - CX) / FX, (v - CY) / FY, 1), so the ray
parameter t is the camera depth z. Per (ray, SQ) pair:
1. cull (no grad): slab test against the SQ's box (half-extents = scale + a silhouette margin; a plain SQ with
   exponents <= 2 lies inside its box) -> interval [t0, t1], t0 >= ZNEAR;
2. search (no grad): K samples of the SQ's SDF (SuperFlex's radial SDF, `sdf_batch`; its sign is exact) in
   [t0, t1] -> t_min (argmin of the SDF along the ray, refined once on a finer grid) and, if a sample is inside,
   the entry depth t_e (first sign change, bisection);
3. differentiable values:
   - d = SDF at the point t_min (envelope theorem: the gradient of min_t SDF(t) w.r.t. the SQ is the partial at
     the argmin, so t_min itself needs no gradient);
   - z = t_e with one Newton correction t_e - SDF(t_e) / dSDF/dt (dSDF/dt by finite differences, detached): the
     gradient of the implicit surface depth (as IDR / DVR); z = t_min (detached) for rays that miss;
   - alpha = existence * sigmoid(-d / tau), tau = TAU_PX pixels' footprint at z (soft silhouette, ~1 pixel wide).
Compositing per ray: SQs sorted by z (front to back), w_i = alpha_i prod_{j before i} (1 - alpha_j), background
1 - sum_i w_i. The depth order itself is not differentiable (the depth loss moves occluders through z).

Losses (per frame, means over valid rays; `render_losses`), on the final decoder's SQs:
- inst: the instance of every query = its Hungarian-matched target's instance; -log sum_{q of the pixel's GT
  instance} w_q, on pixels whose instance has a matched target (pixel-level instance segmentation; any SQ of the
  instance may cover the pixel, so the target's decomposition into SQs does not matter);
- cls: -log sum_q w_q p_q(class of the pixel's instance) (no matching needed);
- depth: |sum_q w_q z_q / sum_q w_q - observed depth| / observed depth, on pixels with depth and coverage > 0.5.
"""

import torch
import torch.nn.functional as F

import train as T
from model import CX, CY, FX, FY

STRIDE = 2
K = 24  # search samples per (ray, SQ) pair
BISECT = 6
TAU_PX = 0.5  # silhouette softness in pixels (of the rendered resolution)
ZNEAR = 0.1
MIN_EXIST = 0.02  # queries below this existence are not rendered (their alpha is ~0 anyway), except matched ones
CHUNK = 200_000  # pairs per SDF evaluation
SEARCH_CHUNK = 400_000  # (ray, SQ) pairs per search pass: ~0.5 GB of samples
EPS = 1e-3


def rays(H, W, device, stride=None):
    """-> directions (R, 3), pixel rows (R,), pixel cols (R,) of the rendered grid."""
    s = stride or STRIDE
    v = torch.arange(s // 2, H, s, device=device)
    u = torch.arange(s // 2, W, s, device=device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    d = torch.stack([(uu - CX) / FX, (vv - CY) / FY, torch.ones_like(uu, dtype=torch.float32)], -1)
    return d.reshape(-1, 3).float(), vv.reshape(-1), uu.reshape(-1), (len(v), len(u))


def _sdf(sq, si, x):
    """SDF of SQ si[p] at points x[p] (P, M, 3) -> (P, M), in chunks."""
    out = []
    for a in range(0, len(si), CHUNK):
        sl = slice(a, a + CHUNK)
        out.append(T.sdf_pairs(x[sl], {k: v[si[sl]] for k, v in sq.items()}))
    return torch.cat(out) if out else x.new_zeros(0, x.shape[1])


@torch.no_grad()
def cull(sq, d, stride):
    """Slab test of every ray against every SQ's box (+ margin) -> ray idx, SQ idx, t0, t1 of the overlapping pairs."""
    R = sq["rotate"]  # (N, 3, 3), columns = SQ axes in the camera frame
    c = sq["trans"]
    zc = c[:, 2].clamp_min(ZNEAR)
    margin = 4 * TAU_PX * stride * zc / FX  # a few silhouette widths, at the SQ's center depth
    h = sq["scale"] + margin[:, None]  # (N, 3)
    o = -(R.transpose(1, 2) @ c[:, :, None])[:, :, 0]  # camera origin in each SQ's frame (N, 3)
    dl = torch.einsum("nji,rj->rni", R, d)  # ray directions in each SQ's frame (R, N, 3)
    dl = torch.where(dl.abs() < 1e-9, torch.full_like(dl, 1e-9), dl)
    ta, tb = (-h - o)[None] / dl, (h - o)[None] / dl
    t0 = torch.minimum(ta, tb).amax(-1).clamp_min(ZNEAR)
    t1 = torch.maximum(ta, tb).amin(-1)
    ri, si = torch.nonzero(t1 > t0, as_tuple=True)
    return ri, si, t0[ri, si], t1[ri, si]


@torch.no_grad()
def search(sq, d, ri, si, t0, t1):
    """-> t_min (P,), hit (P,) bool, t_entry (P,) (t1 where no hit); in chunks of SEARCH_CHUNK pairs (memory)."""
    n = SEARCH_CHUNK
    if len(ri) > n:
        parts = [_search(sq, d, ri[a:a + n], si[a:a + n], t0[a:a + n], t1[a:a + n]) for a in range(0, len(ri), n)]
        return tuple(torch.cat(x) for x in zip(*parts))
    return _search(sq, d, ri, si, t0, t1)


@torch.no_grad()
def _search(sq, d, ri, si, t0, t1):
    k = (torch.arange(K, device=d.device) + 0.5) / K
    ts = t0[:, None] + (t1 - t0)[:, None] * k  # (P, K)
    s = _sdf(sq, si, d[ri][:, None] * ts[..., None])
    m = s.argmin(1)
    step = (t1 - t0) / K
    tm = ts.gather(1, m[:, None])[:, 0]
    k2 = torch.linspace(-1, 1, 9, device=d.device)
    ts2 = (tm[:, None] + step[:, None] * k2).clamp(t0[:, None], t1[:, None])  # finer grid around the argmin
    s2 = _sdf(sq, si, d[ri][:, None] * ts2[..., None])
    tm = ts2.gather(1, s2.argmin(1)[:, None])[:, 0]
    neg = s < 0
    hit = neg.any(1)
    first = neg.float().argmax(1)
    hi = ts.gather(1, first[:, None])[:, 0]
    lo = torch.where(first > 0, ts.gather(1, (first - 1).clamp_min(0)[:, None])[:, 0], t0)
    hit_lo = hit & (first == 0)  # inside at the first sample (box entered inside the SQ, e.g. at the near plane)
    for _ in range(BISECT):
        mid = (lo + hi) / 2
        inside = _sdf(sq, si, (d[ri] * mid[:, None])[:, None])[:, 0] < 0
        hi, lo = torch.where(inside, mid, hi), torch.where(inside, lo, mid)
    te = torch.where(hit_lo, t0, hi)
    return tm, hit, torch.where(hit, te, t1)


@torch.no_grad()
def entry_depths(sq, H, W, stride=1):
    """Hard ray casting: (R, N) depth at which each ray (`rays`) enters each SQ, inf where it misses -> E, rows, cols."""
    d, vv, uu, _ = rays(H, W, sq["scale"].device, stride)
    E = torch.full((len(d), len(sq["scale"])), torch.inf, device=d.device)
    if len(sq["scale"]):
        ri, si, t0, t1 = cull(sq, d, stride)
        _, hit, te = search(sq, d, ri, si, t0, t1)
        E[ri[hit], si[hit]] = te[hit]
    return E, vv, uu


def render(sq, exist, H, W, stride=None, grad=True):
    """Render N SQs (dict scale / shape / rotate / trans in the camera frame, (N, ...)) with existences (N,) ->
    {"w": (R, N) per-ray weights, "bg": (R,), "z": (R, N) (0 where no pair), "depth": (R,) expected depth over the
    covered part, "cover": (R,), "v", "u": pixel of each ray, "grid": (rows, cols)}."""
    stride = stride or STRIDE
    d, vv, uu, grid = rays(H, W, exist.device, stride)
    Rn, N = len(d), len(exist)
    ri, si, t0, t1 = cull(sq, d, stride)
    tm, hit, te = search(sq, d, ri, si, t0, t1)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        dm = _sdf(sq, si, (d[ri] * tm[:, None])[:, None])[:, 0]
        hp = torch.nonzero(hit)[:, 0]
        z = tm.clone()
        if len(hp):
            xe = d[ri[hp]] * te[hp, None]
            se = _sdf(sq, si[hp], xe[:, None])[:, 0]
            with torch.no_grad():
                h = 5e-3
                sp = _sdf(sq, si[hp], (d[ri[hp]] * (te[hp] + h)[:, None])[:, None])[:, 0]
                sm = _sdf(sq, si[hp], (d[ri[hp]] * (te[hp] - h)[:, None])[:, None])[:, 0]
                g = ((sp - sm) / (2 * h)).clamp(max=-0.05)  # entering: the SDF decreases along the ray
            z = z.index_put((hp,), te[hp] - se / g)
        tau = TAU_PX * stride * z.detach().clamp_min(ZNEAR) / FX
        alpha = (exist[si] * torch.sigmoid(-dm / tau)).clamp(max=0.99)
        A = torch.zeros(Rn, N, device=d.device, dtype=alpha.dtype).index_put((ri, si), alpha)
        Z = torch.zeros(Rn, N, device=d.device, dtype=z.dtype).index_put((ri, si), z)
        with torch.no_grad():
            order = torch.where(A > 0, Z.detach(), torch.full_like(Z, torch.inf)).argsort(1)
        As = A.gather(1, order)
        trans = torch.cumprod(torch.cat([torch.ones_like(As[:, :1]), 1 - As[:, :-1]], 1), 1)
        w = torch.zeros_like(A).scatter(1, order, As * trans)
        cover = w.sum(1)
        depth = (w * Z).sum(1) / cover.clamp_min(EPS)
    return {"w": w, "bg": 1 - cover, "z": Z, "depth": depth, "cover": cover, "v": vv, "u": uu, "grid": grid}


def frame_queries(pred, b, matched):
    """Queries of frame b to render: existence > MIN_EXIST or matched (index tensor)."""
    keep = pred["exist"][b] > MIN_EXIST
    keep[matched] = True
    return torch.nonzero(keep)[:, 0]


def pixel_targets(batch, b, t, out):
    """GT per ray: instance label, class of the instance (-1: instance without a target SQ / unlabelled), depth."""
    lab = batch["label"][b][out["v"], out["u"]]
    dep = batch["depth"][b][out["v"], out["u"]]
    cls = torch.full_like(lab, -1)
    if len(t["instance"]):
        n = int(max(lab.max(), t["instance"].max())) + 2
        lut = torch.full((n,), -1, dtype=torch.long, device=lab.device)
        lut[t["instance"]] = t["cls"]
        cls = torch.where(lab >= 0, lut[lab.clamp_min(0)], -1)
    return lab, cls, dep


def render_losses(pred, batch, fb, qs, tg, cat):
    """Projection losses of the matched batch (fb, qs, tg: frame, query, index into the concatenated targets cat)
    -> {"r_inst", "r_cls", "r_depth"} (means over frames)."""
    H, W = batch["label"].shape[1:]
    tot = {"r_inst": 0.0, "r_cls": 0.0, "r_depth": 0.0}
    B = len(batch["targets"])
    for b, t in enumerate(batch["targets"]):
        m = fb == b
        qm, im = qs[m], cat["instance"][tg[m]]
        q = frame_queries(pred, b, qm)
        sq = {k: pred[k][b, q] for k in ("scale", "shape", "rotate", "trans")}
        out = render(sq, pred["exist"][b, q], H, W)
        lab, cls, dep = pixel_targets(batch, b, t, out)
        q_inst = torch.full((len(q),), -2, dtype=torch.long, device=q.device)
        pos = torch.searchsorted(q, qm)  # q is sorted (nonzero), qm a subset
        q_inst[pos] = im
        w = out["w"]
        if len(qm):
            ok = (lab >= 0) & torch.isin(lab, im)
            if ok.any():
                p = (w[ok] * (q_inst[None] == lab[ok, None])).sum(1)
                tot["r_inst"] = tot["r_inst"] + (-(p + EPS).log()).mean()
        ok = cls >= 0
        if ok.any():
            pc = (w[ok] * pred["cls"][b, q].softmax(-1)[:, cls[ok]].T).sum(1)
            tot["r_cls"] = tot["r_cls"] + (-(pc + EPS).log()).mean()
        ok = (dep > 0) & (out["cover"] > 0.5)
        if ok.any():
            tot["r_depth"] = tot["r_depth"] + ((out["depth"][ok] - dep[ok]).abs() / dep[ok]).mean()
    return {k: v / B for k, v in tot.items()}


@torch.no_grad()
def render_metrics(pred, batch, b, t, qi, ti):
    """Eval: render the predicted SQs (all queries, alpha = existence * silhouette) -> pixel accuracy of the front-most
    query's instance (via the matching qi -> ti) and of the rendered class (argmax of sum_q w_q p_q), coverage of
    the labelled pixels, depth abs rel on covered pixels with depth."""
    H, W = batch["label"].shape[1:]
    qi = torch.as_tensor(qi, device=T.DEV, dtype=torch.long)
    q = frame_queries(pred, b, qi)
    out = render({k: pred[k][b, q] for k in ("scale", "shape", "rotate", "trans")}, pred["exist"][b, q], H, W, grad=False)
    lab, cls, dep = pixel_targets(batch, b, t, out)
    r = {}
    q_inst = torch.full((len(q),), -2, dtype=torch.long, device=q.device)
    if len(qi):
        q_inst[torch.searchsorted(q, qi)] = t["instance"][torch.as_tensor(ti, device=T.DEV, dtype=torch.long)]
    front = out["w"].argmax(1)
    covered = out["cover"] > 0.5
    ok = lab >= 0
    if ok.any():
        r["r_cover"] = float(covered[ok].float().mean())
        r["r_inst_acc"] = float(((q_inst[front] == lab) & covered)[ok].float().mean())
    okc = cls >= 0
    if okc.any():
        pc = out["w"] @ pred["cls"][b, q].softmax(-1)
        r["r_cls_acc"] = float(((pc.argmax(1) == cls) & covered)[okc].float().mean())
    okd = (dep > 0) & covered
    if okd.any():
        r["r_depth_absrel"] = float(((out["depth"][okd] - dep[okd]).abs() / dep[okd]).mean())
    return r
