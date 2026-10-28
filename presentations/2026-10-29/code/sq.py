"""Superquadrics: parameters, radial signed distance, box initialisation, optimisation loop and saving."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from data import DEV, result_path

K = {"room_0": 128, "room_1": 64, "office_0": 64}  # fixed budget of surface / volume (08 / 10's counts, rounded)


def quat_to_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def sdf(x, p):
    """Radial signed distance |q| (1 - 1/G(q)) of points x (..., 3) to superquadrics p (dict of broadcastable
    tensors). q = x in the SQ frame, G = inside-outside function with G = 1 on the surface (experiment 02)."""
    q = ((x - p["trans"])[..., None, :] @ p["rot"])[..., 0, :]
    la = torch.log(q.abs().clamp_min(1e-30)) - torch.log(p["scale"])
    e1, e2 = p["eps"][..., 0], p["eps"][..., 1]
    log_f = torch.logaddexp(e2 / e1 * torch.logaddexp(2 / e2 * la[..., 0], 2 / e2 * la[..., 1]), 2 / e1 * la[..., 2])
    n = q.norm(dim=-1)
    return n - torch.exp(torch.log(n.clamp_min(1e-30)) - 0.5 * e1 * log_f)


class SQs(torch.nn.Module):
    """M superquadrics: center, half extents (exp), shape exponents eps in [0.1, 1.9] (sigmoid, init 1), rotation
    = initial box axes x learned quaternion. `instance` = owning GT instance (-1: no labels)."""

    def __init__(self, trans, axes, log_scale, raw_eps=None, quat=None, instance=None):
        super().__init__()
        M = len(trans)
        self.trans = torch.nn.Parameter(trans)
        self.log_scale = torch.nn.Parameter(log_scale)
        self.raw_eps = torch.nn.Parameter(torch.zeros(M, 2, device=DEV) if raw_eps is None else raw_eps)
        self.quat = torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=DEV).repeat(M, 1) if quat is None else quat)
        self.register_buffer("axes", axes)
        self.instance = torch.full((M,), -1, device=DEV) if instance is None else instance

    @classmethod
    def from_boxes(cls, sets, instance=None, pad=0.0):
        """One SQ per point set, initialised as its PCA box (half extents + pad)."""
        boxes = [box(np.asarray(s, np.float64)) for s in sets]
        t, R, h = (torch.tensor(np.stack(b), dtype=torch.float32, device=DEV) for b in zip(*boxes))
        return cls(t, R, (h + pad).clamp_min(0.01).log(), instance=instance)

    @classmethod
    def select(cls, rows):
        """New SQs from (model, index) pairs."""
        take = lambda name: torch.stack([getattr(m, name)[i].detach() for m, i in rows])
        return cls(take("trans"), take("axes"), take("log_scale"), take("raw_eps"), take("quat"), take("instance"))

    def params(self):
        return {"trans": self.trans, "scale": self.log_scale.exp(), "eps": 0.1 + 1.8 * torch.sigmoid(self.raw_eps),
                "rot": self.axes @ quat_to_matrix(self.quat)}


def box(x):
    """PCA box of points: center, axes (columns, right-handed), half extents."""
    mu = x.mean(0)
    _, axes = np.linalg.eigh(np.cov((x - mu).T))
    axes[:, 2] *= np.sign(np.linalg.det(axes))
    q = (x - mu) @ axes
    lo, hi = q.min(0), q.max(0)
    return mu + axes @ ((lo + hi) / 2), axes, (hi - lo) / 2


def kmeans_init(pos, k, seed=0):
    """k-means of the points into k clusters, one box per cluster."""
    lab = KMeans(k, n_init=1, random_state=seed).fit_predict(pos)
    return SQs.from_boxes([pos[lab == c] for c in range(k) if (lab == c).sum() >= 4])


def optimize(model, loss_fn, steps, lr=0.01):
    """Adam with cosine decay of the learning rate (to 5 %)."""
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.05 * lr)
    for step in range(steps):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}", flush=True)
    return model


def save(model, scene, method):
    p = {k: v.detach().cpu().numpy() for k, v in model.params().items()}
    out = result_path(scene, method)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **p, instance=model.instance.cpu().numpy())
    print(f"{scene} {method}: {len(p['trans'])} SQs -> {out}")
