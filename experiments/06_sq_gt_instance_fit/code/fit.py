"""Limit test: fit K SQs to every GT instance with a forward and a backward SDF loss (see config.yaml).

    uv run python experiments/06_sq_gt_instance_fit/code/fit.py [--config path] [--name run] [--set key=value ...]
    e.g. --set loss.forward.mode=enclosure

SQ j belongs to instance j // K. Writes results/runs/<name>/{config.yaml, metrics.json, snapshots.npz}
(same snapshot format as experiment 05, so the viewer is shared) and logs to trackio (results/large_files/trackio).
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

EXP = Path(__file__).resolve().parent.parent
RESULTS = EXP / "results"
os.environ.setdefault("TRACKIO_DIR", str(RESULTS / "large_files" / "trackio"))  # before experiment 05 sets its own
sys.path.insert(0, str(EXP.parent / "05_sq_loss_optim/code"))
import trackio  # noqa: E402
from optimize import (DEVICE, InstanceSampler, fibonacci_sphere, load_config, load_points, load_scene,  # noqa: E402,F401
                      load_semantics, random_rotation, sdf_lib, surface_samples)
from structure import fit as box_fit, matrix_to_quat  # noqa: E402

PROJECT = "sq_gt_instance_fit"


# ---------------------------------------------------------------- data

class InstanceGrids:
    """Per-instance unsigned distance to the instance's points (EDT), trilinear lookup.
    Outside the padded box: value at the border + distance to the box."""

    def __init__(self, pos: torch.Tensor, y: torch.Tensor, n_inst: int, res: float, pad: float):
        from scipy.ndimage import distance_transform_edt
        p_all, y_all = pos.cpu().numpy(), y.cpu().numpy()
        self.grids, self.lo, self.hi = [], [], []
        self.voxels = 0
        for k in range(n_inst):
            p = p_all[y_all == k]
            lo = p.min(0) - pad
            dims = np.ceil((p.max(0) + pad - lo) / res).astype(int) + 1
            occ = np.zeros(dims, bool)
            idx = np.clip(np.round((p - lo) / res).astype(int), 0, dims - 1)
            occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
            dist = distance_transform_edt(~occ).astype(np.float32) * res
            self.grids.append(torch.from_numpy(dist).permute(2, 1, 0)[None, None].contiguous().to(DEVICE))
            self.lo.append(torch.from_numpy(lo).float().to(DEVICE))
            self.hi.append(torch.from_numpy(lo + (dims - 1) * res).float().to(DEVICE))
            self.voxels += dist.size

    def __call__(self, k: int, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.lo[k], self.hi[k]
        xc = torch.maximum(torch.minimum(x, hi), lo)
        g = (xc - lo) / (hi - lo) * 2 - 1
        u = torch.nn.functional.grid_sample(self.grids[k], g.view(1, -1, 1, 1, 3), mode="bilinear",
                                            padding_mode="border", align_corners=True).view(x.shape[:-1])
        return u + ((x - xc).square().sum(-1) + 1e-12).sqrt() - 1e-6


@torch.no_grad()
def kmeans(x: torch.Tensor, k: int, iters: int, g: torch.Generator) -> torch.Tensor:
    """k-means++ seeding + Lloyd iterations -> labels (P,)."""
    c = x[torch.randint(len(x), (1,), device=DEVICE, generator=g)]
    for _ in range(1, k):
        d2 = torch.cdist(x, c).min(1).values.square()
        c = torch.cat([c, x[torch.multinomial(d2 + 1e-12, 1, generator=g)]])
    for _ in range(iters):
        lab = torch.cdist(x, c).argmin(1)
        for m in range(k):
            sel = lab == m
            if sel.any():
                c[m] = x[sel].mean(0)
    return torch.cdist(x, c).argmin(1)


# ---------------------------------------------------------------- SQ parameters

class SQs(torch.nn.Module):
    def __init__(self, cfg: dict, t: torch.Tensor, R: torch.Tensor, a: torch.Tensor):
        super().__init__()
        par = cfg["parametrization"]
        self.eps_min, self.eps_max = par["eps"]["min"], par["eps"]["max"]
        self.scale_min, self.scale_max = par["scale"]["min"], par["scale"]["max"]
        eps0 = (torch.tensor(cfg["init"]["eps"]) - self.eps_min) / (self.eps_max - self.eps_min)
        self.translation = torch.nn.Parameter(t.clone())
        self.log_scale = torch.nn.Parameter(a.clamp(self.scale_min, self.scale_max).log())
        self.eps_raw = torch.nn.Parameter(torch.logit(eps0).to(t.device).expand(len(t), 2).clone())
        self.rotation = torch.nn.Parameter(matrix_to_quat(R))  # quaternion (w, x, y, z)
        ex = cfg["existence"]
        self.has_existence = ex["enabled"]
        self.existence_logit = torch.nn.Parameter(torch.full((len(t),), math.log(ex["init"] / (1 - ex["init"])),
                                                             device=t.device), requires_grad=ex["enabled"])
        self.split = cfg["split"]["enabled"]
        K = cfg["sqs_per_instance"]
        alive = torch.ones(len(t), dtype=torch.bool, device=t.device)
        if self.split:
            alive.view(-1, K)[:, 1:] = False  # one SQ per instance, the other slots are free
        self.register_buffer("alive", alive)

    def forward(self):
        scale = self.log_scale.exp().clamp(self.scale_min, self.scale_max)
        eps = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(self.eps_raw)
        w, x, y, z = torch.nn.functional.normalize(self.rotation, dim=1).unbind(1)
        rot = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                           2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                           2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
        if self.has_existence:
            existence = torch.sigmoid(self.existence_logit)
        else:
            existence = self.alive.float()  # all ones unless split mode
        return {"scale": scale, "eps": eps, "rot": rot, "trans": self.translation, "existence": existence,
                "alive": self.alive}


@torch.no_grad()
def init_sqs(cfg: dict, scene: dict, g: torch.Generator) -> SQs:
    """k-means into K clusters per instance, principal-axes box fit per cluster.
    Split mode: one box fit to the whole instance, copied into all slots (only slot 0 is alive)."""
    ini, K = cfg["init"], cfg["sqs_per_instance"]
    scale_min = cfg["parametrization"]["scale"]["min"]
    ts, Rs, As = [], [], []
    for k in range(len(scene["instance_ids"])):
        P = scene["pos"][scene["y"] == k]
        if len(P) > ini["kmeans_max_points"]:
            P = P[torch.randperm(len(P), device=DEVICE, generator=g)[:ini["kmeans_max_points"]]]
        if cfg["split"]["enabled"]:
            t, R, a = box_fit(P, ini, scale_min)
            ts += [t] * K
            Rs += [R] * K
            As += [a] * K
            continue
        lab = kmeans(P, K, ini["kmeans_iters"], g)
        for m in range(K):
            Q = P[lab == m]
            t, R, a = box_fit(Q if len(Q) >= 4 else P, ini, scale_min)
            ts.append(t), Rs.append(R), As.append(a)
    return SQs(cfg, torch.stack(ts), torch.stack(Rs), torch.stack(As)).to(DEVICE)


def sdf(cfg: dict, q: torch.Tensor, a: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return sdf_lib.METHODS[cfg["loss"]["sdf"]](q, a, eps[..., 0], eps[..., 1])


def own_sdf(cfg: dict, sq: dict, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(P, K) SDF of every point to the K SQs of its own instance."""
    K = cfg["sqs_per_instance"]
    idx = y[:, None] * K + torch.arange(K, device=DEVICE)                       # (P, K)
    q = torch.einsum("pkj,pkji->pki", x[:, None] - sq["trans"][idx], sq["rot"][idx])
    d = sdf(cfg, q, sq["scale"][idx], sq["eps"][idx])
    return torch.where(sq["alive"][idx], d, torch.full_like(d, 1e3))  # free slots never win a min


def softmin(d: torch.Tensor, tau: float, dim: int) -> torch.Tensor:
    return d.amin(dim) if tau <= 0 else -tau * torch.logsumexp(-d / tau, dim)


@torch.no_grad()
def union_weight(cfg: dict, sq: dict, ys: torch.Tensor) -> torch.Tensor:
    """(N, S) weight that a surface sample of SQ j lies outside the other SQs of its instance."""
    K = cfg["sqs_per_instance"]
    N, S = ys.shape[:2]
    n_inst = N // K
    X = ys.reshape(n_inst, K * S, 3)[torch.arange(N, device=DEVICE) // K]       # (N, K*S, 3): own instance's samples
    q = torch.einsum("npj,nji->npi", X - sq["trans"][:, None], sq["rot"])
    d = sdf(cfg, q, sq["scale"][:, None], sq["eps"][:, None])                    # (N, K*S)
    inside = sq["existence"][:, None] * torch.sigmoid(-d / cfg["loss"]["backward"]["union_sharpness"])
    logout = torch.log1p(-inside.clamp(max=1 - 1e-6)).view(n_inst, K, K * S)
    total = logout.sum(1)                                                         # (n_inst, K*S)
    s = torch.arange(K * S, device=DEVICE)
    own = logout[:, s // S, s]                                                    # the sample's own SQ
    return (total - own).exp().view(N, S)


def instance_distance(grids: InstanceGrids, ys: torch.Tensor, K: int) -> torch.Tensor:
    """(N, S) distance of surface samples of SQ j to the points of instance j // K."""
    N, S = ys.shape[:2]
    return torch.cat([grids(k, ys[k * K:(k + 1) * K].reshape(-1, 3)).view(K, S) for k in range(N // K)])


# ---------------------------------------------------------------- loss

def expected_forward(cfg: dict, sq: dict, d: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Expected forward value under independent Bernoulli existence (Paschalidou et al. 2019):
    SQs sorted by key, E = sum_i e_(i) prod_{k<i} (1 - e_(k)) value_(i) + prod_k (1 - e_k) fallback."""
    K, mode = cfg["sqs_per_instance"], cfg["loss"]["forward"]["mode"]
    key = d.abs() if mode == "min_abs" else d
    value = torch.relu(d) if mode == "enclosure" else d.abs()
    order = key.argsort(1)
    e = sq["existence"][y[:, None] * K + order]                                 # (P, K), sorted
    v = value.gather(1, order)
    none_before = torch.cumprod(torch.cat([torch.ones_like(e[:, :1]), 1 - e], 1), 1)  # (P, K + 1)
    return (e * none_before[:, :-1] * v).sum(1) + none_before[:, -1] * cfg["existence"]["fallback"]


def backward_per_instance(cfg: dict, sq: dict, per_sq: torch.Tensor) -> torch.Tensor:
    """(n_inst,) mean of the per-SQ backward values over the instance's alive SQs (free slots do not dilute it)."""
    K = cfg["sqs_per_instance"]
    n_alive = sq["alive"].view(-1, K).sum(1).clamp_min(1)
    return per_sq.view(-1, K).sum(1) / n_alive


def loss_fn(cfg: dict, sq: dict, x: torch.Tensor, y: torch.Tensor, grids: InstanceGrids, dirs: torch.Tensor):
    F, B = cfg["loss"]["forward"], cfg["loss"]["backward"]
    d = own_sdf(cfg, sq, x, y)                                                  # (P, K)
    if cfg["existence"]["enabled"]:
        f = expected_forward(cfg, sq, d, y)
    elif F["mode"] == "union_abs":
        f = softmin(d, F["softmin"], 1).abs()
    elif F["mode"] == "enclosure":
        f = torch.relu(softmin(d, F["softmin"], 1))
    elif F["mode"] == "min_abs":
        f = softmin(d.abs(), F["softmin"], 1)
    else:
        raise ValueError(F["mode"])
    fwd = f.mean()
    ys = surface_samples(sq, dirs)                                              # (N, S, 3)
    v = union_weight(cfg, sq, ys)
    u = instance_distance(grids, ys, cfg["sqs_per_instance"])
    e = sq["existence"]
    bwd = backward_per_instance(cfg, sq, (e[:, None] * v * torch.relu(u - B["delta"])).mean(1)).mean()
    pars = e.view(-1, cfg["sqs_per_instance"]).sum(1).mean()
    total = F["weight"] * fwd + B["weight"] * bwd + cfg["loss"]["parsimony"]["weight"] * pars
    return total, {"loss/forward": fwd.item(), "loss/backward": bwd.item(), "loss/parsimony": pars.item(),
                   "loss/total": total.item(), "batch/union_visible": v.mean().item()}


# ---------------------------------------------------------------- full-scene evaluation

def bin_names(bins: list[int]) -> list[str]:
    edges = ["0"] + [f"{b // 1000}k" for b in bins]
    return [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f"{edges[-1]}+"]


@torch.no_grad()
def evaluate(cfg: dict, sq: dict, scene: dict, grids: InstanceGrids) -> dict:
    """Instance union SDF U_k(p) = min_{j in k} d_pj on all points.
    forward: |U_y(p)|; enclosed: U_y(p) <= tol; leaked: min_{k != y} U_k(p) < -tol; prediction: argmin_k U_k(p)."""
    x, y = scene["pos"], scene["y"]
    K, n_inst, tol = cfg["sqs_per_instance"], len(scene["instance_ids"]), cfg["metrics"]["tolerance"]
    N = len(sq["trans"])
    active = sq["existence"] > cfg["metrics"].get("active_existence", 0.5)
    sq = {**sq, "existence": active.float()}
    chunk = max(4096, (1 << 25) // N)
    own, foreign, pred, owner = [], [], [], []
    for i in range(0, len(x), chunk):
        xc, yc = x[i:i + chunk], y[i:i + chunk]
        d = sdf(cfg, sdf_lib.to_local(xc, sq["rot"], sq["trans"]), sq["scale"][:, None], sq["eps"][:, None])
        dk = torch.where(active[:, None], d, torch.full_like(d, float("inf"))).view(n_inst, K, -1)
        U = dk.amin(1)                                                          # (n_inst, c)
        c = torch.arange(len(xc), device=DEVICE)
        own.append(U[yc, c])
        pred.append(U.argmin(0))
        owner.append(yc * K + dk[yc, :, c].argmin(1))
        U[yc, c] = float("inf")
        foreign.append(U.amin(0))
    own, foreign, pred, owner = map(torch.cat, (own, foreign, pred, owner))

    count = torch.bincount(y, minlength=n_inst).float()
    per_inst = lambda v: torch.zeros(n_inst, device=DEVICE).index_add_(0, y, v.float()) / count  # noqa: E731
    enclosed_i, leaked_i, fwd_i = per_inst(own <= tol), per_inst(foreign < -tol), per_inst(own.abs().clamp(max=1.0))
    conf = torch.bincount(y * n_inst + pred, minlength=n_inst * n_inst).view(n_inst, n_inst).float()
    iou = conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp_min(1)

    ys = surface_samples(sq, fibonacci_sphere(256).to(DEVICE))
    v = union_weight(cfg, sq, ys)
    u = instance_distance(grids, ys, K)
    vi, ui = (v * sq["existence"][:, None]).view(n_inst, -1), u.view(n_inst, -1)
    bwd_i = (vi * ui).sum(1) / vi.sum(1).clamp_min(1e-9)
    air_i = (vi * (ui > cfg["metrics"]["air_delta"])).sum(1) / vi.sum(1).clamp_min(1e-9)

    m = {
        "eval/mean_iou": iou.mean().item(),
        "eval/accuracy": (pred == y).float().mean().item(),
        "eval/forward_inst": fwd_i.mean().item(),
        "eval/backward_inst": bwd_i.mean().item(),
        "eval/enclosed": (own <= tol).float().mean().item(),
        "eval/enclosed_inst": enclosed_i.mean().item(),
        "eval/leaked": (foreign < -tol).float().mean().item(),
        "eval/leaked_inst": leaked_i.mean().item(),
        "eval/surface_in_air_inst": air_i.mean().item(),
        "eval/active_sqs": int(active.sum().item()),
        "eval/active_per_inst": active.float().mean().item() * K,
    }
    n_active = active.view(n_inst, K).sum(1).float()
    b = torch.bucketize(count, torch.tensor(cfg["metrics"]["size_bins"], device=DEVICE, dtype=torch.float), right=True)
    for i, name in enumerate(bin_names(cfg["metrics"]["size_bins"])):
        sel = b == i
        if sel.any():
            m[f"bins/enclosed_{name}"] = enclosed_i[sel].mean().item()
            m[f"bins/leaked_{name}"] = leaked_i[sel].mean().item()
            m[f"bins/iou_{name}"] = iou[sel].mean().item()
            m[f"bins/forward_{name}"] = fwd_i[sel].mean().item()
            m[f"bins/backward_{name}"] = bwd_i[sel].mean().item()
            m[f"bins/active_{name}"] = n_active[sel].mean().item()
    return {"metrics": m, "iou": iou.cpu().numpy(), "owned": torch.bincount(owner, minlength=N).cpu().numpy(),
            "active": active.cpu().numpy(),
            "per_instance": {"points": count.cpu().numpy(), "enclosed": enclosed_i.cpu().numpy(),
                             "leaked": leaked_i.cpu().numpy(), "forward": fwd_i.cpu().numpy(),
                             "backward": bwd_i.cpu().numpy(), "active": n_active.cpu().numpy()}}


# ---------------------------------------------------------------- top-down splitting

def split_due(cfg: dict, step: int) -> bool:
    s = cfg["split"]
    last = s["stop"] + (s["every"] if s.get("verify", False) else 0)  # one more check to verify the last proposals
    return s["enabled"] and s["start"] <= step <= last and (step - s["start"]) % s["every"] == 0


@torch.no_grad()
def instance_stats(cfg: dict, sq: dict, scene: dict, grids: InstanceGrids) -> dict:
    """Exact per-instance loss L_k = forward_k + backward_k on all points (same normalization as the training loss),
    per-SQ share of the instance's forward error, owned points and per-point owner."""
    K, mode = cfg["sqs_per_instance"], cfg["loss"]["forward"]["mode"]
    x, y = scene["pos"], scene["y"]
    n_inst = len(scene["instance_ids"])
    owner, err = [], torch.zeros(len(sq["trans"]), device=DEVICE)
    for i in range(0, len(x), 262144):
        xc, yc = x[i:i + 262144], y[i:i + 262144]
        d = own_sdf(cfg, sq, xc, yc)
        m = (d.abs() if mode == "min_abs" else d).argmin(1)
        dm = d.gather(1, m[:, None]).squeeze(1)
        j = yc * K + m
        owner.append(j)
        err.index_add_(0, j, torch.relu(dm) if mode == "enclosure" else dm.abs())
    owner = torch.cat(owner)
    count = torch.bincount(y, minlength=n_inst).float()
    share = (err / count.repeat_interleave(K)).view(n_inst, K)
    ys = surface_samples(sq, fibonacci_sphere(256).to(DEVICE))
    bw = sq["existence"][:, None] * union_weight(cfg, sq, ys) * torch.relu(instance_distance(grids, ys, K)
                                                                         - cfg["loss"]["backward"]["delta"])
    L = cfg["loss"]["forward"]["weight"] * share.sum(1) + cfg["loss"]["backward"]["weight"] * backward_per_instance(cfg, sq, bw.mean(1))
    owned = torch.bincount(owner, minlength=len(sq["trans"])).view(n_inst, K)
    return {"L": L, "share": share, "owned": owned, "owner": owner}


class Splitter:
    """Every `every` steps: (1) judge the splits proposed at the previous check, (2) propose new ones.

    Proposal: per instance with a free slot, split the alive, not cooling-down SQ with the largest share c_j of the
    instance's mean forward error, if c_j > min_gain (2-means of its owned points, box fit per half). Both halves
    inherit the parent's Adam second moments (first moments zeroed).
    Verification (verify: true, fair): a shadow copy of the model trains on the same batches without the proposals.
    A split is kept if L_k(split) < L_k(shadow) - min_gain; otherwise instance k is copied back from the shadow
    (parameters, slots, Adam state), and the split SQ cools down for `retry_cooldown` checks."""

    NAMES = ("translation", "log_scale", "eps_raw", "rotation", "existence_logit")

    def __init__(self, cfg: dict, model: SQs, opt, g: torch.Generator):
        self.cfg, self.g, self.model, self.opt = cfg, g, model, opt
        self.cooldown = torch.zeros(len(model.alive), dtype=torch.long, device=DEVICE)  # check index it may retry at
        self.check = 0
        self.pending = []  # instances with a trial split
        self.shadow, self.shadow_opt = None, None
        if cfg["split"].get("verify", False):
            self.shadow = copy.deepcopy(model)
            self.shadow_opt = torch.optim.Adam([{"params": [getattr(self.shadow, n)], "lr": gr["lr"]}
                                                for n, gr in zip(self.NAMES, opt.param_groups)])

    def _params(self, m):
        return [getattr(m, n) for n in self.NAMES]

    @torch.no_grad()
    def _copy(self, src, src_opt, dst, dst_opt, rows=None):
        """Copy parameters, alive mask and Adam state (all rows, or the given rows) from src to dst."""
        for ps, pd in zip(self._params(src), self._params(dst)):
            if rows is None:
                pd.data.copy_(ps.data)
            else:
                pd.data[rows] = ps.data[rows]
            ss = src_opt.state.get(ps)
            if not ss:
                continue
            sd = dst_opt.state.setdefault(pd, {})
            for key, v in ss.items():
                if rows is None or key == "step":
                    sd[key] = v.clone()
                else:
                    sd[key][rows] = v[rows]
        if rows is None:
            dst.alive.copy_(src.alive)
        else:
            dst.alive[rows] = src.alive[rows]

    @property
    def shadow_active(self) -> bool:
        return bool(self.pending)

    def train_shadow(self, x, y, grids, dirs):
        for gs, gm in zip(self.shadow_opt.param_groups, self.opt.param_groups):
            gs["lr"] = gm["lr"]
        loss, _ = loss_fn(self.cfg, self.shadow(), x, y, grids, dirs)
        self.shadow_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.shadow_opt.step()

    @torch.no_grad()
    def step(self, step: int, scene: dict, grids: InstanceGrids) -> dict:
        S, K, ini = self.cfg["split"], self.cfg["sqs_per_instance"], self.cfg["init"]
        model, opt = self.model, self.opt
        n_inst = len(scene["instance_ids"])
        self.check += 1
        stats = instance_stats(self.cfg, model(), scene, grids)
        kept, undone, touched, gains = 0, 0, set(), []
        if self.pending:
            L_sh = instance_stats(self.cfg, self.shadow(), scene, grids)["L"]
            for k, j in self.pending:
                gains.append((L_sh[k] - stats["L"][k]).item())
                if stats["L"][k] < L_sh[k] - S["min_gain"]:
                    kept += 1
                    continue
                rows = torch.arange(k * K, (k + 1) * K, device=DEVICE)
                self._copy(self.shadow, self.shadow_opt, model, opt, rows)
                self.cooldown[j] = self.check + S.get("retry_cooldown", 2)
                touched.add(k)
                undone += 1
            self.pending = []
        if self.shadow is not None:
            self._copy(model, opt, self.shadow, self.shadow_opt)  # shadow = state without the new proposals

        n_split = 0
        if step <= S["stop"]:
            x = scene["pos"]
            alive = model.alive.view(n_inst, K)
            share = stats["share"].clone()
            share[~alive | (self.cooldown > self.check).view(n_inst, K)] = -1
            free = ~alive
            best = share.argmax(1)
            cand = free.any(1) & (share.max(1).values > S["min_gain"]) & \
                (stats["owned"].gather(1, best[:, None]).squeeze(1) >= S["min_points"])
            for k in torch.nonzero(cand).squeeze(1).tolist():
                if k in touched:  # its stats are stale after the restore
                    continue
                j = k * K + best[k].item()
                f = k * K + torch.nonzero(free[k])[0].item()
                P = x[stats["owner"] == j]
                if len(P) > ini["kmeans_max_points"]:
                    P = P[torch.randperm(len(P), device=DEVICE, generator=self.g)[:ini["kmeans_max_points"]]]
                lab = kmeans(P, 2, S["kmeans_iters"], self.g)
                parts = [P[lab == 0], P[lab == 1]]
                if min(len(q) for q in parts) < 4:
                    continue
                for slot, Q in zip((j, f), parts):
                    t, R, a = box_fit(Q, ini, model.scale_min)
                    model.translation.data[slot] = t
                    model.rotation.data[slot] = matrix_to_quat(R[None])[0]
                    model.log_scale.data[slot] = a.clamp(model.scale_min, model.scale_max).log()
                model.eps_raw.data[f] = model.eps_raw.data[j]
                model.alive[f] = True
                for p in self._params(model):  # halves inherit the parent's second moments
                    st = opt.state.get(p)
                    if st:
                        st["exp_avg"][[j, f]] = 0
                        st["exp_avg_sq"][f] = st["exp_avg_sq"][j]
                n_split += 1
                if self.shadow is not None:
                    self.pending.append((k, j))
        out = {"split/splits": n_split, "split/kept": kept, "split/undone": undone,
               "split/sqs_per_inst": model.alive.float().sum().item() / n_inst}
        if gains:  # gain of the judged splits: L_k(shadow) - L_k(split), in m
            gq = torch.tensor(gains).quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()
            out.update({"split/gain_p10": gq[0], "split/gain_median": gq[1], "split/gain_p90": gq[2],
                        "split/gain_positive": sum(x > 0 for x in gains) / len(gains)})
        return out


# ---------------------------------------------------------------- main loop

def run(cfg: dict, name: str):
    torch.manual_seed(cfg["seed"])
    g = torch.Generator(device=DEVICE).manual_seed(cfg["seed"])
    scene = load_scene(cfg)
    n_inst, K = len(scene["instance_ids"]), cfg["sqs_per_instance"]
    t0 = time.perf_counter()
    model = init_sqs(cfg, scene, g)
    B = cfg["loss"]["backward"]
    grids = InstanceGrids(scene["pos"], scene["y"], n_inst, B["grid_res"], B["grid_pad"])
    print(f"{name}: {len(scene['pos']):,} labeled points, {n_inst} instances, {n_inst * K} SQs; "
          f"init + {grids.voxels / 1e6:.0f} M grid voxels in {time.perf_counter() - t0:.1f} s", flush=True)
    sampler = InstanceSampler(scene["y"], g)
    dirs0 = fibonacci_sphere(B["samples_per_sq"]).to(DEVICE)
    lr = cfg["optim"]["lr"]
    opt = torch.optim.Adam([
        {"params": [model.translation], "lr": lr["translation"]},
        {"params": [model.log_scale], "lr": lr["log_scale"]},
        {"params": [model.eps_raw], "lr": lr["eps"]},
        {"params": [model.rotation], "lr": lr["rotation"]},
        {"params": [model.existence_logit], "lr": lr.get("existence", 0.05)},
    ])
    steps = cfg["optim"]["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))

    out = RESULTS / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    trackio.init(project=PROJECT, name=name, config=cfg, auto_log_gpu=False, auto_log_cpu=False)

    sq_label = (np.arange(n_inst * K) // K)
    snaps, history = [], []

    def snapshot(step):
        with torch.no_grad():
            sq = model()
        ev = evaluate(cfg, sq, scene, grids)
        snaps.append({"step": step, **{k: v.detach().cpu().numpy() for k, v in sq.items()}, "sq_label": sq_label,
                      "purity": np.ones(len(sq_label), np.float32), "active": ev["active"],
                      "owned": ev["owned"]})
        return ev

    splitter = Splitter(cfg, model, opt, g)
    t_start = time.perf_counter()
    for step in range(steps + 1):
        if step % cfg["logging"]["snapshot_every"] == 0 or step == steps:
            ev = snapshot(step)
            if step % cfg["logging"]["eval_every"] == 0 or step == steps:
                history.append({"step": step, **ev["metrics"]})
                trackio.log(ev["metrics"], step=step)
                m = ev["metrics"]
                print(f"step {step:5d}  mIoU {m['eval/mean_iou']:.3f}  fwd {100 * m['eval/forward_inst']:.2f} cm  "
                      f"bwd {100 * m['eval/backward_inst']:.2f} cm  enclosed {m['eval/enclosed_inst']:.3f}  "
                      f"leaked {m['eval/leaked_inst']:.3f}  SQs/inst {m['eval/active_per_inst']:.2f}", flush=True)
        if step == steps:
            break
        if split_due(cfg, step):
            slog = splitter.step(step, scene, grids)
            trackio.log(slog, step=step)
            gain = (f"  gain p10/50/90 {1000 * slog['split/gain_p10']:.1f} / {1000 * slog['split/gain_median']:.1f} / "
                    f"{1000 * slog['split/gain_p90']:.1f} mm, > 0: {slog['split/gain_positive']:.2f}") if "split/gain_median" in slog else ""
            print(f"step {step:5d}  split: {slog['split/kept']} kept, {slog['split/undone']} undone, "
                  f"{slog['split/splits']} new -> {slog['split/sqs_per_inst']:.2f} SQs/inst{gain}", flush=True)
        t = time.perf_counter()
        idx = sampler(cfg["sampling"]["points_per_step"])
        dirs = dirs0 @ random_rotation(g).T
        loss, logs = loss_fn(cfg, model(), scene["pos"][idx], scene["y"][idx], grids, dirs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if splitter.shadow_active:  # same batch, same learning rate, without the trial splits
            splitter.train_shadow(scene["pos"][idx], scene["y"][idx], grids, dirs)
        sched.step()
        torch.cuda.synchronize()
        if step % 10 == 0:
            logs.update({"time/step_ms": 1000 * (time.perf_counter() - t), "lr/translation": sched.get_last_lr()[0]})
            trackio.log(logs, step=step)
    trackio.finish()

    elapsed = time.perf_counter() - t_start
    keys = [k for k in snaps[0] if k != "step"]
    np.savez_compressed(out / "snapshots.npz", step=np.array([s["step"] for s in snaps]),
                        **{k: np.stack([s[k] for s in snaps]) for k in keys},
                        instance_ids=scene["instance_ids"], R=scene["R"], t=scene["t"])
    snap, big = out / "snapshots.npz", RESULTS / "large_files" / "runs" / name / "snapshots.npz"
    big.unlink(missing_ok=True)
    if snap.stat().st_size > 1e6:  # >1MB results go to results/large_files (not versioned)
        big.parent.mkdir(parents=True, exist_ok=True)
        snap.replace(big)
    (out / "metrics.json").write_text(json.dumps({
        "name": name, "seconds": elapsed, "final": history[-1], "history": history, "final_iou": ev["iou"].tolist(),
        "per_instance": {k: v.tolist() for k, v in ev["per_instance"].items()}}, indent=1))
    print(f"done in {elapsed:.0f} s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "config.yaml"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. loss.forward.mode=enclosure")
    args = ap.parse_args()
    cfg = load_config(Path(args.config), args.set)
    name = args.name or f"{cfg['data']['scene']}_{cfg['loss']['forward']['mode']}_{time.strftime('%m%d-%H%M%S')}"
    run(copy.deepcopy(cfg), name)


if __name__ == "__main__":
    main()
