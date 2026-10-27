"""Signed distance (approximations) for superquadrics, evaluated in the SQ's local frame.

Inside-outside function (a = half-extents, e1/e2 = shape exponents, convex for e1, e2 <= 2):
    F(q) = (|qx/a1|^(2/e2) + |qy/a2|^(2/e2))^(e2/e1) + |qz/a3|^(2/e1)
Gauge function G = F^(e1/2) is positively homogeneous of degree 1 (G(s q) = s G(q)), G = 1 on the surface.

Methods (all take local points q (M, N, 3), scale a (M, 1, 3), e1, e2 (M, 1) and return (M, N)):
    radial         |q| (1 - 1/G): distance to the surface point on the ray from the center (overestimates)
    gauge_lb       r_in (G - 1): Lipschitz lower bound for convex SQs (|grad G| <= 1/r_in, r_in = inradius;
                   r_in = min(a) only for e1, e2 <= 1)
    taubin         (G - 1) / |grad G|: first-order distance, exact on flat faces
    newton_radial  closest point by Levenberg-Marquardt, started at the radial projection
    newton_grid    same, started at the best of a k x k candidate grid (or the radial projection)

The closest-point search parametrizes the surface by radial projection of a direction,
    y(phi, psi) = D / G(D),  D = a * (cos phi cos psi, cos phi sin psi, sin phi),  phi, psi in [0, pi/2],
on the first octant (the query is folded with |q|, valid because SQs are symmetric about the
coordinate planes). Unlike the classic (eta, omega) angle parametrization, this stays well
conditioned for box-like shapes, where tiny angle ranges otherwise cover whole faces.
Gradients of the Newton methods use the envelope theorem: the optimization runs without autograd,
then the distance is re-evaluated once, differentiably, at the (detached) optimal parameters.
"""

import math

import torch

TINY = 1e-30
HALF_PI = math.pi / 2


def _logs(q, a, e1, e2):
    lq = torch.log(q.abs().clamp_min(TINY))  # log |q_i|
    la = lq - torch.log(a)
    A = (2 / e2) * la[..., 0]
    B = (2 / e2) * la[..., 1]
    C = (2 / e1) * la[..., 2]
    logS = torch.logaddexp(A, B)
    logF = torch.logaddexp((e2 / e1) * logS, C)
    return lq, A, B, C, logS, logF


def gauge(q, a, e1, e2):
    return torch.exp(0.5 * e1 * _logs(q, a, e1, e2)[-1])


def gauge_grad(q, a, e1, e2):
    """G and its gradient, computed in log space (each component is one exp of a sum of logs)."""
    lq, A, B, C, logS, logF = _logs(q, a, e1, e2)
    base = (0.5 * e1 - 1) * logF
    gx = torch.exp(base + (e2 / e1 - 1) * logS + A - lq[..., 0])
    gy = torch.exp(base + (e2 / e1 - 1) * logS + B - lq[..., 1])
    gz = torch.exp(base + C - lq[..., 2])
    grad = torch.stack([gx, gy, gz], -1) * torch.sign(q)
    return torch.exp(0.5 * e1 * logF), grad


# ---------- closed-form approximations ----------

def radial(q, a, e1, e2):
    logG = 0.5 * e1 * _logs(q, a, e1, e2)[-1]
    n = q.norm(dim=-1)
    return n - torch.exp(torch.log(n.clamp_min(TINY)) - logG)  # |q| - |q| / G without 0 * inf at the center


def inradius(a, e1, e2, k=32):
    """Radius of the largest centered ball inside the SQ, 1 / max G(u) over unit directions u (k x k octant grid).

    Equals min(a) for e1, e2 <= 1, but is smaller for pointy shapes (octahedron: 1 / |1/a|).
    The grid slightly underestimates max G (error quadratic in the grid spacing, measured <0.05% for
    k = 32), so the result is shrunk by 0.2% to stay a strict lower bound.
    """
    t = grid_theta(k, a.device)
    u = torch.stack([t[:, 0].cos() * t[:, 1].cos(), t[:, 0].cos() * t[:, 1].sin(), t[:, 0].sin()], -1)
    return 0.998 / gauge(u, a, e1, e2).amax(-1, keepdim=True)  # (M, 1)


def gauge_lb(q, a, e1, e2):
    return inradius(a, e1, e2) * (gauge(q, a, e1, e2) - 1)


def taubin(q, a, e1, e2):
    G, g = gauge_grad(q, a, e1, e2)
    return (G - 1) / g.norm(dim=-1).clamp_min(TINY)


# ---------- closest point (Levenberg-Marquardt) ----------

def surface(theta, a, e1, e2):
    """Surface point y (..., 3) and Jacobian dy/dtheta (..., 3, 2) for directions theta = (phi, psi)."""
    phi, psi = theta[..., 0], theta[..., 1]
    cf, sf = torch.cos(phi).clamp_min(0), torch.sin(phi)
    cp, sp = torch.cos(psi).clamp_min(0), torch.sin(psi)
    D = a * torch.stack([cf * cp, cf * sp, sf], -1)
    G, g = gauge_grad(D, a, e1, e2)
    G = G[..., None]
    y = D / G
    dphi = a * torch.stack([-sf * cp, -sf * sp, cf], -1)
    dpsi = a * torch.stack([-cf * sp, cf * cp, torch.zeros_like(cf)], -1)
    # dy/dD v = (v - y (grad G . v)) / G   (y is homogeneous of degree 0 in D)
    jphi = (dphi - y * (g * dphi).sum(-1, keepdim=True)) / G
    jpsi = (dpsi - y * (g * dpsi).sum(-1, keepdim=True)) / G
    return y, torch.stack([jphi, jpsi], -1)


def radial_theta(p, a):
    u = p / a
    return torch.stack([torch.atan2(u[..., 2], torch.hypot(u[..., 0], u[..., 1])), torch.atan2(u[..., 1], u[..., 0])], -1)


def grid_theta(k, device):
    t = torch.linspace(0, HALF_PI, k, device=device)
    return torch.stack(torch.meshgrid(t, t, indexing="ij"), -1).reshape(-1, 2)  # (k*k, 2)


@torch.no_grad()
def lm(p, theta, a, e1, e2, iters):
    """Minimize |y(theta) - p|^2 over theta in [0, pi/2]^2; per-point damping, monotone (only improving steps)."""
    y, J = surface(theta, a, e1, e2)
    r = y - p
    cost = (r * r).sum(-1)
    lam = torch.full_like(cost, 1e-3)
    for _ in range(iters):
        j0, j1 = J[..., 0], J[..., 1]
        h00, h01, h11 = (j0 * j0).sum(-1), (j0 * j1).sum(-1), (j1 * j1).sum(-1)
        g0, g1 = (j0 * r).sum(-1), (j1 * r).sum(-1)
        a00, a11 = h00 * (1 + lam) + 1e-12, h11 * (1 + lam) + 1e-12
        det = a00 * a11 - h01 * h01
        step = torch.stack([-(a11 * g0 - h01 * g1) / det, -(a00 * g1 - h01 * g0) / det], -1)
        th2 = (theta + step).clamp(0, HALF_PI)
        y2, J2 = surface(th2, a, e1, e2)
        r2 = y2 - p
        c2 = (r2 * r2).sum(-1)
        ok = c2 < cost
        theta = torch.where(ok[..., None], th2, theta)
        r = torch.where(ok[..., None], r2, r)
        J = torch.where(ok[..., None, None], J2, J)
        cost = torch.where(ok, c2, cost)
        lam = torch.where(ok, lam * 0.3, lam * 10).clamp(1e-9, 1e9)
    return theta, cost


@torch.no_grad()
def best_candidate(p, a, e1, e2, k):
    """Best of the radial projection and a k x k direction grid (candidates shared per SQ)."""
    theta = radial_theta(p, a)
    cost = ((surface(theta, a, e1, e2)[0] - p) ** 2).sum(-1)
    cand_t = grid_theta(k, p.device)  # (K, 2)
    cand = surface(cand_t, a, e1, e2)[0]  # (M, K, 3); a (M, 1, 3) and e (M, 1) broadcast over K
    d2 = torch.cdist(p, cand) ** 2  # (M, N, K)
    c2, idx = d2.min(-1)
    better = c2 < cost
    return torch.where(better[..., None], cand_t[idx], theta)


def _signed(q, a, e1, e2, theta):
    """Differentiable distance at fixed theta (envelope theorem), signed by the inside-outside test."""
    p = q.abs()
    d = (surface(theta.detach(), a, e1, e2)[0] - p).norm(dim=-1)
    return torch.where(gauge(q, a, e1, e2) < 1, -d, d)


def newton_radial(q, a, e1, e2, iters=10):
    p = q.detach().abs()
    theta, _ = lm(p, radial_theta(p, a.detach()), a.detach(), e1.detach(), e2.detach(), iters)
    return _signed(q, a, e1, e2, theta)


def newton_grid(q, a, e1, e2, k=8, iters=10):
    p, ad, e1d, e2d = q.detach().abs(), a.detach(), e1.detach(), e2.detach()
    theta, _ = lm(p, best_candidate(p, ad, e1d, e2d, k), ad, e1d, e2d, iters)
    return _signed(q, a, e1, e2, theta)


@torch.no_grad()
def reference(q, a, e1, e2, k=96, iters=60):
    """Ground truth: LM from the best of a dense k x k grid and from the radial projection, keep the closer."""
    p = q.abs()
    t1, c1 = lm(p, best_candidate(p, a, e1, e2, k), a, e1, e2, iters)
    t2, c2 = lm(p, radial_theta(p, a), a, e1, e2, iters)
    return _signed(q, a, e1, e2, torch.where((c1 <= c2)[..., None], t1, t2))


METHODS = {
    "radial": radial,
    "gauge_lb": gauge_lb,
    "taubin": taubin,
    "newton_radial": newton_radial,
    "newton_grid": newton_grid,
}


def to_local(x, rot, trans):
    """World points x (N, 3) -> local frames of M SQs (M, N, 3); rot is local-to-world (M, 3, 3)."""
    return torch.einsum("mnj,mji->mni", x[None] - trans[:, None], rot)


def evaluate(method, x, scale, eps, rot, trans, chunk_pairs=1 << 22):
    """SDF of N world points to each of M SQs -> (M, N), chunked over points."""
    m = len(scale)
    n_chunk = max(1, chunk_pairs // m)
    a, e1, e2 = scale[:, None], eps[:, 0, None], eps[:, 1, None]
    return torch.cat([method(to_local(x[i:i + n_chunk], rot, trans), a, e1, e2)
                      for i in range(0, len(x), n_chunk)], 1)
