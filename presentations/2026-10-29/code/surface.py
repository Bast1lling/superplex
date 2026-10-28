"""surface: a fixed number of SQs fitted to the point cloud's surface, without labels.

    python surface.py [--res 0.02] [scene ...]

forward:  every point lies on the union of the SQs,       mean_p |min_j d_j(p)|
backward: every SQ surface lies on points,                 mean_s |s - nearest point(s)| over samples s of all surfaces
"""

import math

import torch
from scipy.spatial import cKDTree

from data import DEV, cli, load_points
from sq import K, kmeans_init, optimize, save, sdf

STEPS, N_POINTS, N_SURFACE = 2000, 16384, 128


def surface_samples(p, n):
    """n points per SQ at random parametric angles: (M, n, 3)."""
    M = len(p["trans"])
    eta = (torch.rand(M, n, device=DEV) - 0.5) * math.pi
    omega = (torch.rand(M, n, device=DEV) * 2 - 1) * math.pi
    spow = lambda v, e: v.sign() * v.abs().clamp_min(1e-6) ** e
    a, e1, e2 = p["scale"][:, None], p["eps"][:, None, 0], p["eps"][:, None, 1]
    local = torch.stack([a[..., 0] * spow(eta.cos(), e1) * spow(omega.cos(), e2),
                         a[..., 1] * spow(eta.cos(), e1) * spow(omega.sin(), e2),
                         a[..., 2] * spow(eta.sin(), e1)], -1)
    return p["trans"][:, None] + local @ p["rot"].transpose(1, 2)


def main(scene):
    pts = load_points(scene)
    pos = torch.from_numpy(pts["pos"]).to(DEV)
    tree = cKDTree(pts["pos"])
    model = kmeans_init(pts["pos"], K[scene])

    def loss():
        p = model.params()
        x = pos[torch.randint(len(pos), (N_POINTS,), device=DEV)]
        forward = sdf(x[:, None], p).min(1).values.abs().mean()
        s = surface_samples(p, N_SURFACE).reshape(-1, 3)
        nearest = pos[torch.from_numpy(tree.query(s.detach().cpu().numpy(), workers=-1)[1]).to(DEV)]
        backward = (s - nearest).norm(dim=-1).mean()
        return forward + backward

    save(optimize(model, loss, STEPS), scene, "surface")


if __name__ == "__main__":
    for scene in cli():
        main(scene)
