"""SuperFlex's test-time optimization objective vs. ours, from three initializations, per GT instance.

    python sfopt.py --scene room_0 --init network|kmeans_net|kmeans16 --objective theirs|ours [--steps 1000]

Per instance (like SuperFlex's ShapeNet setup): 4096 surface points (its 1 cm GT points) and 100k query points
uniform in its bounding box (+10 % of the largest extent), labelled from the 4 cm GT grid of experiment 12
(`grid.gt_grid`). Each instance is centered first (SuperFlex's `normalize` computes the scale before centering,
harmless on ShapeNet, wrong for objects metres away from the origin); SuperFlex then scales it to unit size.

Initializations (16 primitive slots per instance, as the network):
- network:    SuperFlex `robust` checkpoint (primitives with existence > 0.5), as on the presentation branch.
- kmeans_net: k-means of the surface points into as many clusters as the network uses, one PCA box each (eps 1).
- kmeans16:   16 k-means boxes, then SuperFlex's pruning (after the optimization: primitives closest to <= 5 of the
              4096 surface points are removed).
Objectives, both on SuperFlex's parametrization (tapering + bending, its SDF and smooth union, its parameter groups,
1000 Adam steps, best state per instance) and the same query points:
- theirs: SuperFlex `SuperQOptimization` loss unchanged: -log soft IoU (occupancy = query point in a voxel of the
          instance; option a: other instances are outside; deep voxels inside objects belong to the nearest solid
          voxel's instance) + surface SDF term + bounding-box term + overlap term.
- ours:   the instances loss of experiments 12-15 / the presentation: softplus(d/tau) on the query points in the
          instance's voxels, softplus(-d/tau) weighted free 1, deep 0.1, other instances 0, tau = 2 cm.
Evaluation: experiment 12's per-instance vIoU (TP = GT voxels of k inside the union, FP = free voxels inside it,
here only free voxels within the instance's box + 0.5 m), with SuperFlex's (deformed) SDF.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension as cpp
from omegaconf import OmegaConf
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "12_rgbd_multiview" / "code"))
from grid import FREE, UNKNOWN, gt_grid  # noqa: E402
from views import DATA, DEV, gt  # noqa: E402

# SuperFlex compiles its PVCNN ops with -std=c++17, torch >= 2.14 headers need C++20 (submodule left untouched)
_load = cpp.load
cpp.load = lambda *a, extra_cflags=(), **kw: _load(*a, extra_cflags=[f.replace("c++17", "c++20") for f in extra_cflags], **kw)
sys.path.insert(0, str(ROOT / "papers" / "superflex"))
from superflex.optimization import SuperQOptimization  # noqa: E402
from superflex.superflex import SuperFlex  # noqa: E402
from superflex.utils.predictions_handler import PredictionHandler, extend_dict  # noqa: E402

N_SURF, N_IOU, SLOTS, BATCH, TAU = 4096, 100_000, 16, 16, 0.02
ROT_X_90 = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # network y-up -> Replica z-up (column vectors)


def instance_data(scene, grid, seed=0):
    """Per instance: centered surface points, centered query points, their labels (instance / FREE / UNKNOWN /
    beyond the grid = FREE), the instance of each deep voxel's nearest solid voxel, and the center."""
    g = gt(scene)
    rng = np.random.default_rng(seed)
    L = grid.label_np
    _, near = distance_transform_edt(L < 0, return_indices=True)  # nearest solid voxel of every voxel
    filled = np.where(L == UNKNOWN, L[tuple(near)], L)
    out = []
    for k in range(len(g["ids"])):
        P = g["pos"][g["inst"] == k].astype(np.float64)
        if len(P) < 4:
            continue
        surf = P[rng.choice(len(P), N_SURF, replace=len(P) < N_SURF)]
        lo, hi = P.min(0), P.max(0)
        pad = 0.1 * (hi - lo).max()
        q = lo - pad + rng.random((N_IOU, 3)) * (hi - lo + 2 * pad)
        i = np.round((q - grid.lo_np) / grid.res).astype(int)
        inside = ((i >= 0) & (i < L.shape)).all(1)
        i = np.minimum(np.maximum(i, 0), np.array(L.shape) - 1)
        lab = np.where(inside, L[tuple(i.T)], FREE)
        full = np.where(inside, filled[tuple(i.T)], FREE)
        c = surf.mean(0)
        out.append({"k": k, "c": c, "surf": surf - c, "q": q - c, "lab": lab, "occ": full == k, "lo": lo, "hi": hi})
    return out


def network_init(data):
    """SuperFlex robust on every instance (normalized, y-up), primitives back in the centered instance frame."""
    folder = DATA / "models" / "superflex" / "robust"
    model = SuperFlex(OmegaConf.load(folder / "config.yaml").superflex).to(DEV)
    model.load_state_dict(torch.load(folder / "ckpt.pt", map_location=DEV, weights_only=False)["model_state_dict"])
    model.eval()
    s = np.array([2 * np.abs(d["surf"]).max() for d in data])
    x = torch.tensor(np.stack([d["surf"] / si @ ROT_X_90 for d, si in zip(data, s)]), dtype=torch.float32, device=DEV)
    with torch.no_grad():
        out = model(x)
    out["bending_k"] = out["bending_k"] / torch.tensor(s, device=DEV, dtype=torch.float32)[:, None, None]
    out = {k: v.cpu().numpy() for k, v in extend_dict(out).items() if isinstance(v, torch.Tensor)}
    return {"scale": out["scale"] * s[:, None, None], "translation": (out["trans"] @ ROT_X_90.T) * s[:, None, None],
            "rotation": ROT_X_90 @ out["rotate"], "exponents": out["shape"],
            "exist": out["exist"].reshape(len(data), SLOTS), "tapering": out["tapering"], "bending": out["bending"]}


def kmeans_init(data, counts):
    """counts[i] k-means boxes of instance i's surface points, in the remaining slots exist = 0."""
    B = len(data)
    p = {"scale": np.ones((B, SLOTS, 3)), "translation": np.zeros((B, SLOTS, 3)), "rotation": np.tile(np.eye(3), (B, SLOTS, 1, 1)),
         "exponents": np.ones((B, SLOTS, 2)), "exist": np.zeros((B, SLOTS)), "tapering": np.zeros((B, SLOTS, 2)),
         "bending": np.zeros((B, SLOTS, 6))}
    for b, (d, n) in enumerate(zip(data, counts)):
        lab = KMeans(int(n), n_init=1, random_state=0).fit_predict(d["surf"])
        for j in range(int(n)):
            x = d["surf"][lab == j]
            if len(x) < 4:
                continue
            mu = x.mean(0)
            _, axes = np.linalg.eigh(np.cov((x - mu).T))
            axes[:, 2] *= np.sign(np.linalg.det(axes))
            q = (x - mu) @ axes
            p["translation"][b, j] = mu + axes @ ((q.min(0) + q.max(0)) / 2)
            p["scale"][b, j] = np.maximum((q.max(0) - q.min(0)) / 2, 0.005)
            p["rotation"][b, j], p["exist"][b, j] = axes, 1.0
    return p


class OurObjective(SuperQOptimization):
    """SuperFlex's parametrization / SDF / smooth union, but the instances loss on the same query points."""

    def compute_losses(self, forward_out):
        d = forward_out["sdfs"][:, :self.M_points_iou]
        tau = TAU / self.normalization_scale[:, None]
        inside = (self.pos * F.softplus(d / tau)).sum(1) / self.pos.sum(1).clamp_min(1)
        outside = (self.w * F.softplus(-d / tau)).sum(1) / self.free.sum(1).clamp_min(1)
        loss = inside + outside
        return loss, {}


def run_batch(data, params, objective, prune, steps, deform=True, cls=None, w_bbox=1.0):
    """SuperFlex's optimize.py loop on one batch of instances; returns the optimizer object (normalized frame).
    cls replaces the objective's optimizer class (experiment 17: masked IoU, reads `known` per query point)."""
    B = len(data)
    handler = PredictionHandler({"names": np.arange(B), "pc": np.stack([d["surf"] for d in data]),
                                 "assign_matrix": np.zeros((B, 1, SLOTS)), **params})
    ext = {"points": torch.tensor(np.stack([d["surf"] for d in data]), dtype=torch.float32, device=DEV),
           "normals": torch.zeros(B, N_SURF, 3, device=DEV),
           "points_iou": torch.tensor(np.stack([d["q"] for d in data]), dtype=torch.float32, device=DEV),
           "occupancies": torch.tensor(np.stack([d["occ"] for d in data]), device=DEV)}
    cfg = OmegaConf.create({"tapering": deform, "bending": deform, "pruning": prune, "w_bbox": w_bbox})
    cls = cls or (SuperQOptimization if objective == "theirs" else OurObjective)
    sq = cls(pred_handler=handler, indices=list(range(B)), device=DEV, external_data=ext, cfg=cfg)
    if "known" in data[0]:
        sq.known = torch.tensor(np.stack([d["known"] for d in data]), device=DEV).float()
    if objective == "ours":
        lab = torch.tensor(np.stack([d["lab"] for d in data]), device=DEV)
        k = torch.tensor([d["k"] for d in data], device=DEV)[:, None]
        sq.pos, sq.free = (lab == k).float(), (lab == FREE).float()
        sq.w = sq.free + 0.1 * (lab == UNKNOWN).float()
    opt = torch.optim.Adam(sq.get_param_groups())
    names = ["raw_scale", "raw_exponents", "raw_rotation", "translation", "raw_tapering", "raw_bending"]
    best = [float("inf")] * B
    best_p = [None] * B
    for step in range(steps):
        opt.zero_grad()
        loss, _ = sq.compute_losses(sq.forward())
        if torch.isnan(loss.mean()):
            print(f"  nan loss at step {step}")
            break
        loss.mean().backward()
        opt.step()
        with torch.no_grad():
            for b in range(B):
                if loss[b].item() < best[b]:
                    best[b] = loss[b].item()
                    best_p[b] = {n: getattr(sq, n)[b].clone() for n in names}
        if step % 250 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {loss.mean().item():.4f}", flush=True)
    with torch.no_grad():
        for b in range(B):
            if best_p[b] is not None:
                for n in names:
                    getattr(sq, n)[b].copy_(best_p[b][n])
    sq.update_handler(compute_meshes=False)  # pruning (if enabled) and parameters back into `handler`
    return sq, handler


@torch.no_grad()
def evaluate(sq, data, grid, margin=0.5, chunk=1 << 14):
    """Per instance in the batch: TP / FP / IoU against the GT grid with SuperFlex's SDF (normalized frame)."""
    L = grid.label_np
    rows = []
    for b, d in enumerate(data):
        lo = np.floor((d["lo"] - margin - grid.lo_np) / grid.res).astype(int)
        hi = np.ceil((d["hi"] + margin - grid.lo_np) / grid.res).astype(int)
        idx = np.stack(np.meshgrid(*[np.arange(a, z + 1) for a, z in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
        inside = ((idx >= 0) & (idx < L.shape)).all(1)
        lab = np.full(len(idx), FREE)
        lab[inside] = L[tuple(idx[inside].T)]
        keep = (lab == d["k"]) | (lab == FREE)
        x = grid.lo_np + idx[keep] * grid.res - d["c"]
        x = (torch.tensor(x, dtype=torch.float32, device=DEV) - sq.normalization_translation[b]) / sq.normalization_scale[b]
        occ = []
        for a in range(0, len(x), chunk):
            s = sq.sdf_batch(x[a:a + chunk].T[None].expand(len(data), -1, -1).contiguous())[b]  # (N, m)
            s[~sq.exist_mask[b]] = float("inf")
            occ.append(s.min(0).values < 0)
        occ = torch.cat(occ).cpu().numpy()
        is_k = lab[keep] == d["k"]
        tp, fp, n = int((occ & is_k).sum()), int((occ & ~is_k).sum()), int(is_k.sum())
        rows.append({"k": d["k"], "gt_vox": n, "tp": tp, "fp": fp, "iou": tp / max(n + fp, 1),
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(n, 1), "K": int(sq.exist_mask[b].sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room_0")
    ap.add_argument("--init", choices=["network", "kmeans_net", "kmeans16"], required=True)
    ap.add_argument("--objective", choices=["theirs", "ours"], required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--plain", action="store_true", help="plain SQs: no tapering / bending (zeroed in the init)")
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0)
    grid = gt_grid(a.scene)
    data = instance_data(a.scene, grid)
    net = network_init(data)
    if a.init == "network":
        params = net
    else:
        counts = (net["exist"] > 0.5).sum(1).clip(1) if a.init == "kmeans_net" else np.full(len(data), SLOTS)
        params = kmeans_init(data, counts)
    if a.plain:
        params["tapering"], params["bending"] = np.zeros_like(params["tapering"]), np.zeros_like(params["bending"])
    rows, out = [], {k: [] for k in ("scale", "exponents", "rotation", "translation", "tapering", "bending", "exist")}
    for a0 in range(0, len(data), BATCH):
        sl = slice(a0, a0 + BATCH)
        sq, h = run_batch(data[sl], {k: v[sl].copy() for k, v in params.items()}, a.objective, a.init == "kmeans16", a.steps, not a.plain)
        rows += evaluate(sq, data[sl], grid)
        for k in out:
            out[k].append(getattr(h, k))
        out["translation"][-1] = out["translation"][-1] + np.stack([d["c"] for d in data[sl]])[:, None]  # un-center
    sqs = {k: np.concatenate(v) for k, v in out.items()} | {"instance": np.array([d["k"] for d in data])}
    summ = {"instances": len(rows), "viou": float(np.mean([r["iou"] for r in rows])),
            "precision": float(np.mean([r["precision"] for r in rows])), "recall": float(np.mean([r["recall"] for r in rows])),
            "primitives": int(sum(r["K"] for r in rows)), "seconds": round(time.time() - t0, 1)}
    name = f"{a.scene.replace('_', '')}_{a.init}_{a.objective}" if a.steps else f"{a.scene.replace('_', '')}_{a.init}_nooptim"
    name += "_plain" * a.plain
    d = EXP / "results" / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "sqs.npz", **sqs)
    (d / "metrics.json").write_text(json.dumps({"summary": summ, "per_instance": rows, "args": vars(a)}, indent=1))
    print(name, json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
