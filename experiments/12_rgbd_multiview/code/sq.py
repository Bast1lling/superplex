"""Superquadrics (as in presentations/2026-10-29 on the presentation branch): parameters, radial signed distance,
PCA-box initialisation."""

import numpy as np
import torch
import torch.nn.functional as F

from views import DEV


def quat_to_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def sdf(x, p):
    """Radial signed distance of points x (..., 3) to superquadrics p (dict of broadcastable tensors)."""
    q = ((x - p["trans"])[..., None, :] @ p["rot"])[..., 0, :]
    la = torch.log(q.abs().clamp_min(1e-30)) - torch.log(p["scale"])
    e1, e2 = p["eps"][..., 0], p["eps"][..., 1]
    log_f = torch.logaddexp(e2 / e1 * torch.logaddexp(2 / e2 * la[..., 0], 2 / e2 * la[..., 1]), 2 / e1 * la[..., 2])
    n = q.norm(dim=-1)
    return n - torch.exp(torch.log(n.clamp_min(1e-30)) - 0.5 * e1 * log_f)


def box(x):
    """PCA box of points: center, axes (columns, right-handed), half extents."""
    mu = x.mean(0)
    _, axes = np.linalg.eigh(np.cov((x - mu).T))
    axes[:, 2] *= np.sign(np.linalg.det(axes))
    q = (x - mu) @ axes
    lo, hi = q.min(0), q.max(0)
    return mu + axes @ ((lo + hi) / 2), axes, (hi - lo) / 2


class SQs(torch.nn.Module):
    """One SQ per point set: center, half extents (exp), eps in [0.1, 1.9] (sigmoid, init 1), rotation = PCA axes x
    learned quaternion. `instance` = GT instance of each SQ."""

    def __init__(self, sets, instance, pad=0.0):
        super().__init__()
        t, R, h = (torch.tensor(np.stack(b), dtype=torch.float32, device=DEV) for b in zip(*[box(s) for s in sets]))
        M = len(sets)
        self.trans = torch.nn.Parameter(t)
        self.log_scale = torch.nn.Parameter((h + pad).clamp_min(0.01).log())
        self.raw_eps = torch.nn.Parameter(torch.zeros(M, 2, device=DEV))
        self.quat = torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=DEV).repeat(M, 1))
        self.register_buffer("axes", R)
        self.instance = torch.as_tensor(instance, device=DEV)

    def params(self):
        return {"trans": self.trans, "scale": self.log_scale.exp(), "eps": 0.1 + 1.8 * torch.sigmoid(self.raw_eps),
                "rot": self.axes @ quat_to_matrix(self.quat)}

    def numpy(self):
        out = {k: v.detach().cpu().numpy() for k, v in self.params().items()}
        return out | {"instance": self.instance.cpu().numpy()}


def optimize(model, loss_fn, steps, lr=0.01, log_every=500):
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.05 * lr)
    for step in range(steps):
        loss = loss_fn(step)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}", flush=True)
    return model
