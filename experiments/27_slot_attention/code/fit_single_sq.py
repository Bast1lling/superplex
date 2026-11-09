"""One SQ per GT instance, fit by direct optimization (no SuperFlex) -> alternative targets (`--mask_source inst1`).

    uv run python fit_single_sq.py [scene ...]            (default room_2)   -> data/replica_single_sq/<scene>.npz

GT points: exp 12's `gt(scene)` (1 cm Replica points, instance index = the label index of the cached frames).
Per instance (up to 4096 random points), all instances of a scene in one batch:
- init: PCA of the points (centre = mean, axes = principal directions, half-axes = half the 2-98 % extent per axis, at
  least 1 cm), exponents 0.5;
- half-axes >= MIN_HALF (1.5 cm): walls and floors become >= 3 cm slabs (exp 22's targets were 3 cm slabs), not sheets
  (volumetric IoU of a sheet is meaningless);
- loss: mean |SDF| of the points (`common.sdf`, the near-metric distance) + mean distance of the SQ's cube samples
  (294) to their nearest point (keeps the SQ from enclosing empty space) + sidedness relu(1 cm - SDF(p + 1 cm * n))
  with the GT outward normal n (the slab lies behind the surface, not centred on it);
- Adam, lr 1e-2, 2000 steps; parameters: log half-axes, exponents 0.1 + 1.8 sigmoid, 6D rotation, translation.
Saved in the Replica world frame: scale, exponents, rotation (columns = SQ axes), translation, instance index; plus
per instance mean / median |SDF| of all its points and the sample -> point distance.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as C

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "12_rgbd_multiview" / "code"))
from views import gt  # noqa: E402

OUT = Path("/home/sebastian/Documents/Uni/Thesis/data/replica_single_sq")
NPTS, STEPS, LR = 4096, 2000, 1e-2
MIN_HALF = 0.015  # m, minimum half-axis: walls / floors become slabs >= 3 cm thick (as exp 22's), not sheets
DELTA = 0.01  # m, sidedness: a point moved DELTA along its outward normal must be >= DELTA outside the SQ


def rot6d(r):
    a, b = r[..., :3], r[..., 3:]
    x = F.normalize(a, dim=-1)
    y = F.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)  # columns


def fit(scene):
    g = gt(scene)
    inst = g["inst"]
    K = int(inst.max()) + 1
    rng = np.random.RandomState(0)
    pts, nrm = [], []
    for k in range(K):
        p, n = g["pos"][inst == k], g["nrm"][inst == k]
        sel = rng.choice(len(p), NPTS, replace=len(p) < NPTS)
        pts.append(p[sel]), nrm.append(n[sel])
    X = torch.tensor(np.stack(pts), device=C.DEV)  # (K, n, 3)
    Nn = torch.tensor(np.stack(nrm), device=C.DEV)
    # PCA init
    c = X.mean(1)
    U, S, Vh = torch.linalg.svd(X - c[:, None], full_matrices=False)
    R0 = Vh.transpose(1, 2)  # columns = principal axes
    R0 = torch.where((torch.linalg.det(R0) < 0)[:, None, None], R0 * torch.tensor([1, 1, -1.0], device=C.DEV), R0)
    loc = torch.einsum("kji,knj->kni", R0, X - c[:, None])
    q = torch.quantile(loc, torch.tensor([0.02, 0.98], device=C.DEV), dim=1)  # (2, K, 3)
    half = ((q[1] - q[0]) / 2 - MIN_HALF).clamp_min(1e-3)
    c = c + torch.einsum("kij,kj->ki", R0, (q[1] + q[0]) / 2)
    params = {"ls": half.log().clone().requires_grad_(), "e": torch.full((K, 2), -0.9, device=C.DEV).requires_grad_(),  # 0.1 + 1.8 sigmoid(-0.9) ~ 0.53
              "r": torch.cat([R0[:, :, 0], R0[:, :, 1]], -1).clone().requires_grad_(), "t": c.clone().requires_grad_()}
    opt = torch.optim.Adam(params.values(), lr=LR)
    sq = lambda: {"scale": MIN_HALF + params["ls"].exp(), "shape": 0.1 + 1.8 * torch.sigmoid(params["e"]), "rotate": rot6d(params["r"]),
                  "trans": params["t"]}
    t0 = time.time()
    for step in range(STEPS):
        s = sq()
        d = C.sdf_pairs(X, s).abs().mean(1)  # points -> surface
        smp = C.surf(s, 16, "cube")  # (K, 294, 3)
        back = torch.cdist(smp, X).min(2).values.mean(1)  # surface -> points
        side = torch.relu(DELTA - C.sdf_pairs(X + DELTA * Nn, s)).mean(1)  # body behind the surface, not in front
        loss = (d + back + side).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == STEPS - 1:
            print(f"{scene} step {step} mean |sdf| {d.mean():.4f} sample->pt {back.mean():.4f} side {side.mean():.4f} {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        s = sq()
        med, mean = [], []
        for k in range(K):
            p = torch.tensor(g["pos"][inst == k], device=C.DEV)
            dd = C.sdf_pairs(p[None], {kk: v[k:k + 1] for kk, v in s.items()})[0].abs()
            med.append(float(dd.median())), mean.append(float(dd.mean()))
        back = torch.cdist(C.surf(s, 16, "cube"), X).min(2).values.mean(1)
    s = {k: v.detach() for k, v in s.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{scene}.npz", scale=s["scale"].cpu().numpy(), exponents=s["shape"].cpu().numpy(),
             rotation=s["rotate"].cpu().numpy(), translation=s["trans"].cpu().numpy(), instance=np.arange(K),
             names=np.array(g["names"]), sdf_median=np.array(med), sdf_mean=np.array(mean), back=back.cpu().numpy())
    print(f"{scene}: {K} instances, median over instances of median |sdf| {np.median(med):.4f}, mean |sdf| {np.mean(mean):.4f}")
    for k in np.argsort(mean)[::-1][:8]:
        print(f"   worst: {k:3d} {g['names'][k]:20s} n={int((inst == k).sum()):6d} mean |sdf| {mean[k]:.3f} median {med[k]:.3f}")


if __name__ == "__main__":
    for sc in sys.argv[1:] or ["room_2"]:
        fit(sc)
