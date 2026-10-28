"""GPU core shared by experiments 09 and 10: fit superquadric unions to many voxel *sets* in parallel.

A set is a target voxel set (a part of an instance, a whole instance, or a part of the scene blob) with a sampling
box (its bounding box + margin). Each set has K slots (SQs); the loss is experiment 08's logistic occupancy loss on
the union SDF d = min_j d_j of the set's active slots:
    L_s = mean_T softplus(d / tau) + (|box| / |T|) * mean_box w softplus(-d / tau) + w_region * (SQ outside box)
Negatives are uniform samples in the box, weighted by the label of their voxel: own target 0, other part of the same
group ("sibling") w_sib, other instance w_other, free w_free, unknown w_unknown, exterior w_exterior.
Groups = instances (exp 10) or connected components of the whole solid (exp 09).

Volume, SQ parameterization and SDF are experiment 08's (code/volume.py, code/sq.py).
"""

from pathlib import Path
import math
import sys

import numpy as np
from scipy import ndimage
import torch

EXP08 = Path(__file__).resolve().parents[2] / "08_volumetric_decomp" / "code"
sys.path.insert(0, str(EXP08))
from sq import SQSet, sdf_radial  # noqa: E402
from volume import EXTERIOR, FREE, UNKNOWN, cached_volume, fill_planar, load_scene  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = dict(w_free=1.0, w_other=0.1, w_unknown=0.03, w_exterior=1.0, w_sib=0.1)
STRICT = dict(w_free=1.0, w_other=0.3, w_unknown=0.1, w_exterior=0.1)  # 08's strict metric weights (reeval.py)


class Scene:
    """Labeled volume on the GPU + per-instance target voxels (optionally with 08's planar fill)."""

    def __init__(self, scene, res=0.04, min_points=20, planar=True, labels="instance"):
        self.cfg = {"scene": scene, "res": res, "min_points": min_points, "labels": labels}
        self.sc = load_scene(scene, min_points=min_points, labels=labels)
        vol = cached_volume(self.cfg, self.sc, "clean2")
        self.res, self.lo_np, self.dims = float(vol["res"]), vol["lo"], vol["dims"]
        L = vol["label"].astype(np.int64)
        self.L_np = L
        self.n_inst = len(self.sc["instance_ids"])
        G = np.where(L >= 0, L, -1)
        self.targets = []
        for k in range(self.n_inst):
            vox = np.argwhere(L == k)
            if len(vox) == 0:
                vox = np.unique(np.clip(np.round((self.sc["pos"][self.sc["y"] == k] - self.lo_np) / self.res).astype(int),
                                        0, self.dims - 1), axis=0)
            if planar:
                e = fill_planar(vox, self.res, 0.2, 0.5)
                e = e[((e >= 0) & (e < self.dims)).all(1)]
                if len(e):
                    vox = np.unique(np.concatenate([vox, e]), axis=0)
                    m = L[tuple(e.T)] < 0
                    G[tuple(e[m].T)] = k
            self.targets.append(vox)
        self.L = torch.from_numpy(L).to(DEV)
        self.G = torch.from_numpy(G).to(DEV)
        self.lo = torch.tensor(self.lo_np, dtype=torch.float32, device=DEV)
        self.dims_t = torch.tensor(self.dims, device=DEV)
        self.P = None  # part id grid (set by Sets)

    def centers(self, vox):
        return self.lo_np + vox * self.res

    def lookup(self, x):
        v = torch.round((x - self.lo) / self.res).long()
        v = torch.minimum(torch.maximum(v, torch.zeros_like(v)), self.dims_t - 1)
        return v[:, 0], v[:, 1], v[:, 2]


class Sets:
    """Batch of target sets. parts: list of (N_i, 3) voxel index arrays; group: group id per set; is_part: whether
    a set is a part of its group (siblings exist) or the whole group."""

    def __init__(self, scene, parts, group, is_part=True, margin=0.2, P=None, own=None):
        self.scene, self.S = scene, len(parts)
        cnt = np.array([len(p) for p in parts])
        self.cnt_np = cnt
        allv = np.concatenate(parts)
        self.vox = torch.from_numpy(allv).to(DEV)
        self.x = torch.from_numpy(scene.centers(allv).astype(np.float32)).to(DEV)
        self.sid = torch.from_numpy(np.repeat(np.arange(self.S), cnt)).to(DEV)
        self.cnt = torch.from_numpy(cnt).to(DEV)
        self.off = torch.from_numpy(np.concatenate([[0], np.cumsum(cnt)[:-1]])).to(DEV)
        lo = np.stack([p.min(0) for p in parts]) * scene.res + scene.lo_np - margin
        hi = np.stack([p.max(0) for p in parts]) * scene.res + scene.lo_np + margin
        self.box_lo = torch.tensor(lo, dtype=torch.float32, device=DEV)
        self.box_hi = torch.tensor(hi, dtype=torch.float32, device=DEV)
        self.nbox = torch.tensor(np.prod(hi - lo, 1) / scene.res ** 3, dtype=torch.float32, device=DEV)
        self.group = torch.as_tensor(np.asarray(group), device=DEV).long()
        self.is_part = is_part
        if is_part and P is None:  # part id grid (parts of one batch are disjoint, except planar fill: last wins)
            P = torch.full(tuple(scene.dims), -1, dtype=torch.long, device=DEV)
            P[self.vox[:, 0], self.vox[:, 1], self.vox[:, 2]] = self.sid
            own = torch.arange(self.S, device=DEV)[:, None]
        # P: grid of partition ids; own (S, m): partition ids that make up set s (e.g. the two parts of a merge pair)
        self.P = P
        self.own = None if own is None else torch.as_tensor(own, device=DEV).long()

    def sample_neg(self, n, g=None):
        S = self.S
        r = torch.rand(S, n, 3, device=DEV, generator=g)
        x = (self.box_lo[:, None] + r * (self.box_hi - self.box_lo)[:, None]).reshape(-1, 3)
        s = torch.arange(S, device=DEV).repeat_interleave(n)
        return x, s

    def neg_weight(self, x, s, w):
        sc = self.scene
        i, j, k = sc.lookup(x)
        lab, grp = sc.L[i, j, k], sc.G[i, j, k]
        out = torch.full_like(x[:, 0], w["w_other"])
        out = torch.where(lab == FREE, torch.full_like(out, w["w_free"]), out)
        out = torch.where(lab == UNKNOWN, torch.full_like(out, w["w_unknown"]), out)
        out = torch.where(lab == EXTERIOR, torch.full_like(out, w["w_exterior"]), out)
        same = grp == self.group[s]
        if self.is_part:
            own = (self.P[i, j, k][:, None] == self.own[s]).any(1)
            out = torch.where(same, torch.full_like(out, w["w_sib"]), out)
            out = torch.where(own, torch.zeros_like(out), out)
        else:
            out = torch.where(same, torch.zeros_like(out), out)
        return out

    def sample_pos(self, n, g=None):
        r = torch.rand(self.S, n, device=DEV, generator=g)
        idx = (self.off[:, None] + (r * self.cnt[:, None]).long()).reshape(-1)
        idx = torch.minimum(idx, (self.off + self.cnt - 1).repeat_interleave(n))
        return self.x[idx], torch.arange(self.S, device=DEV).repeat_interleave(n)


def union_sdf(par, active, x, s):
    t, R, a, e = par["trans"][s], par["rot"][s], par["scale"][s], par["eps"][s]  # (N, K, ...)
    q = torch.einsum("nkj,nkji->nki", x[:, None] - t, R)
    d = sdf_radial(q, a, e[..., 0], e[..., 1])
    return torch.where(active[s], d, torch.full_like(d, 1e3))


def loss_fn(model, sets, tau, n_pos, n_neg, w, g=None, w_region=10.0):
    par = model.params()
    xp, sp = sets.sample_pos(n_pos, g)
    xn, sn = sets.sample_neg(n_neg, g)
    wn = sets.neg_weight(xn, sn, w)
    dp = union_sdf(par, model.active, xp, sp).amin(1)
    dn = union_sdf(par, model.active, xn, sn).amin(1)
    lp = torch.nn.functional.softplus(dp / tau).view(sets.S, n_pos).mean(1)
    ln = (wn * torch.nn.functional.softplus(-dn / tau)).view(sets.S, n_neg).mean(1) * sets.nbox / sets.cnt
    ext = (par["rot"].abs() * par["scale"][..., None, :]).sum(-1)
    out = torch.relu(par["trans"] + ext - sets.box_hi[:, None]) + torch.relu(sets.box_lo[:, None] - par["trans"] + ext)
    reg = (out.sum(-1) * model.active).sum(1)
    return (lp + ln).sum() + w_region * reg.sum()


def optimize(model, sets, steps=300, tau=0.02, lr=0.01, n_pos=256, n_neg=512, w=W, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    for _ in range(steps):
        loss = loss_fn(model, sets, tau, n_pos, n_neg, w, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


@torch.no_grad()
def iou(model, sets, w=W, n_eval=8192, seed=1, chunk=1 << 18, tol=0.0):
    """Hard IoU per set: exact on targets, negatives estimated from n_eval uniform box samples. tol (m): a target
    counts as covered if d < tol, a negative as covered only if d < -tol (discretization-tolerant IoU: a thin part
    of 1-2 voxels cannot reach a high strict voxel IoU even with a perfect SQ)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    par = model.params()
    inside = torch.cat([(union_sdf(par, model.active, sets.x[a:a + chunk], sets.sid[a:a + chunk]).amin(1) < tol)
                        for a in range(0, len(sets.x), chunk)])
    tp = torch.zeros(sets.S, device=DEV).index_add(0, sets.sid, inside.float())
    fp = torch.zeros(sets.S, device=DEV)
    per = max(1, chunk // sets.S)
    for a in range(0, n_eval, per):
        n = min(per, n_eval - a)
        xn, sn = sets.sample_neg(n, g)
        wn = sets.neg_weight(xn, sn, w)
        d = union_sdf(par, model.active, xn, sn).amin(1)
        fp += (wn * (d < -tol).float()).view(sets.S, n).sum(1)
    fp = fp / n_eval * sets.nbox
    return tp / (sets.cnt + fp).clamp(min=1), tp / sets.cnt, fp


# ---------------------------------------------------------------- initialization / helpers

def box_fit(x, res):
    """PCA box (center, rotation with columns = axes, half extents) of points x."""
    mu = x.mean(0)
    c = x - mu
    if len(x) < 4:
        return mu, np.eye(3), np.full(3, res)
    _, vec = np.linalg.eigh(c.T @ c)
    R = vec[:, ::-1].copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    q = c @ R
    lo, hi = q.min(0), q.max(0)
    return mu + R @ ((lo + hi) / 2), R, (hi - lo) / 2 + res / 2


def single_fit(scene, parts, group, is_part=True, steps=300, eps_init=0.5, w=W, eps_max=1.2, P=None, own=None,
               n_eval=8192, tol=0.0):
    """One SQ per part, all parts in parallel. Returns (model, sets, iou, recall)."""
    sets = Sets(scene, parts, group, is_part, P=P, own=own)
    model = SQSet(len(parts), 1, eps_range=(0.1, eps_max))
    for i, p in enumerate(parts):
        t, R, a = box_fit(scene.centers(p), scene.res)
        model.set_slot(i, 0, t, R, a, (eps_init, eps_init))
    model.to(DEV)
    optimize(model, sets, steps, w=w)
    q, rec, _ = iou(model, sets, w, n_eval=n_eval, tol=tol)
    return model, sets, q.cpu().numpy(), rec.cpu().numpy()


def components(vox, connectivity=3):
    """Connected components of a voxel index set -> list of index arrays into vox."""
    a = vox.min(0)
    grid = np.zeros(vox.max(0) - a + 1, bool)
    grid[tuple((vox - a).T)] = True
    st = ndimage.generate_binary_structure(3, connectivity)
    lab, n = ndimage.label(grid, structure=st)
    l = lab[tuple((vox - a).T)]
    return [np.nonzero(l == c)[0] for c in range(1, n + 1)]


def model_to_sqs(model, owner):
    """Active slots -> flat SQ dict (08's sqs.npz format); owner[s] = instance index of set s."""
    par = {k: v.detach().cpu().numpy() for k, v in model.params().items()}
    act = model.active.cpu().numpy()
    ii, kk = np.nonzero(act)
    return {k: v[ii, kk] for k, v in par.items()} | {"instance": np.asarray(owner)[ii]}


def group_model(scene, sqs, groups, eps_max=1.2):
    """SQSet with one row per group holding the given SQs (sqs['instance'] = group index)."""
    G = len(groups)
    K = max(1, int(np.bincount(sqs["instance"], minlength=G).max()))
    model = SQSet(G, K, eps_range=(0.1, eps_max))
    slot = np.zeros(G, int)
    for m, gi in enumerate(sqs["instance"]):
        model.set_slot(int(gi), int(slot[gi]), sqs["trans"][m], sqs["rot"][m], sqs["scale"][m],
                       np.clip(sqs["eps"][m], 0.1 + 1e-3, eps_max - 1e-3))
        slot[gi] += 1
    return model.to(DEV)
