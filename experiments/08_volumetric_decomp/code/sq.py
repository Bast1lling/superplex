"""Superquadric parameters (batched as instances x slots) and differentiable SDF approximations."""

import math

import torch

TINY = 1e-30


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 4) (w, x, y, z), numerically stable branch selection."""
    shp = R.shape[:-2]
    m = R.reshape(-1, 3, 3)
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cands = torch.stack([
        torch.stack([1 + t, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], 1),
        torch.stack([m[:, 2, 1] - m[:, 1, 2], 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0]], 1),
        torch.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], m[:, 1, 2] + m[:, 2, 1]], 1),
        torch.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], 1),
    ], 1)
    diag = torch.stack([1 + t, 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                        1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], 1)
    q = cands[torch.arange(len(m)), diag.argmax(1)]
    return torch.nn.functional.normalize(q, dim=1).reshape(*shp, 4)


def log_gauge(q, a, e1, e2):
    """log G for local points q (..., 3), G = F^(e1/2) (=1 on the surface, degree-1 homogeneous)."""
    la = torch.log(q.abs().clamp_min(TINY)) - torch.log(a)
    A = (2 / e2) * la[..., 0]
    B = (2 / e2) * la[..., 1]
    C = (2 / e1) * la[..., 2]
    logF = torch.logaddexp((e2 / e1) * torch.logaddexp(A, B), C)
    return 0.5 * e1 * logF


def sdf_radial(q, a, e1, e2):
    """|q| (1 - 1/G): signed distance along the ray from the center (exact for spheres, overestimates |d|)."""
    n = q.norm(dim=-1)
    return n - torch.exp(torch.log(n.clamp_min(TINY)) - log_gauge(q, a, e1, e2))


def sdf_scaled(q, a, e1, e2):
    """min(a) (G - 1) clipped: Lipschitz-ish lower bound inside, cheap. Used only for comparisons."""
    return a.min(-1).values * (torch.exp(log_gauge(q, a, e1, e2)) - 1)


class SQSet(torch.nn.Module):
    """I instances x K slots. `active` (I, K) marks the slots in use."""

    def __init__(self, I: int, K: int, eps_range=(0.1, 1.9), scale_range=(0.01, 10.0)):
        super().__init__()
        self.I, self.K = I, K
        self.eps_min, self.eps_max = eps_range
        self.scale_min, self.scale_max = scale_range
        self.trans = torch.nn.Parameter(torch.zeros(I, K, 3))
        self.log_scale = torch.nn.Parameter(torch.full((I, K, 3), math.log(0.1)))
        self.eps_raw = torch.nn.Parameter(torch.zeros(I, K, 2))
        self.quat = torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0]).repeat(I, K, 1))
        self.register_buffer("active", torch.zeros(I, K, dtype=torch.bool))

    def params(self):
        scale = self.log_scale.exp().clamp(self.scale_min, self.scale_max)
        eps = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(self.eps_raw)
        return {"trans": self.trans, "scale": scale, "eps": eps, "rot": quat_to_matrix(self.quat)}

    @torch.no_grad()
    def set_slot(self, i, k, t, R, a, eps=(1.0, 1.0)):
        self.trans[i, k] = torch.as_tensor(t, dtype=torch.float32)
        self.log_scale[i, k] = torch.as_tensor(a, dtype=torch.float32).clamp(self.scale_min, self.scale_max).log()
        e = (torch.as_tensor(eps, dtype=torch.float32) - self.eps_min) / (self.eps_max - self.eps_min)
        self.eps_raw[i, k] = torch.logit(e.clamp(1e-3, 1 - 1e-3))
        self.quat[i, k] = matrix_to_quat(torch.as_tensor(R, dtype=torch.float32)[None])[0]
        self.active[i, k] = True


def sdf_slots(par: dict, x: torch.Tensor, inst: torch.Tensor, sdf=sdf_radial) -> torch.Tensor:
    """SDF of points x (S, 3) belonging to instances inst (S,) to all K slots of their instance -> (S, K)."""
    t, R, a, e = par["trans"][inst], par["rot"][inst], par["scale"][inst], par["eps"][inst]  # (S, K, ...)
    q = torch.einsum("skj,skji->ski", x[:, None] - t, R)
    return sdf(q, a, e[..., 0], e[..., 1])


def sdf_all(par: dict, x: torch.Tensor, sdf=sdf_radial) -> torch.Tensor:
    """SDF of points x (S, 3) to every SQ (flattened (M, ...) params) -> (S, M)."""
    q = torch.einsum("smj,mji->smi", x[:, None] - par["trans"][None], par["rot"])
    return sdf(q, par["scale"][None], par["eps"][None, :, 0], par["eps"][None, :, 1])
