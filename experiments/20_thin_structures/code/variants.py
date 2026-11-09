"""Which part of the fit breaks on thin chairs: loss, distance, or initialization? Plain SQs, same loop as experiment 19.

    python variants.py --out results/grid.json

Losses: `iou` is SuperFlex's. `vox` is the voxel occupancy loss of experiments 12-16: mean softplus(d / tau) over the
occupied query points and surface samples + w_free * mean softplus(-d / tau) over the free query points.
Distances: `radial` is SuperFlex's, along the ray through the center, far too large next to a long thin primitive.
`scaled` is g / |grad g| with g the degree-1 inside-outside function, exact on the faces of a box.
Starts: `kmeans` with K boxes, `network` is the superflex checkpoint's prediction without tapering and bending.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "19_sq_count_sweep" / "code"))
from fit_sqs import DATA, DEV, kmeans_boxes, load_chair, optimize  # noqa: E402
from fit_superflex import CHECKPOINT, SuperFlex, SuperQOptimization, PredictionHandler, predict  # noqa: E402

FITS = DATA.parent / "sq_fits" / "test"
TARGET = "fbc560d04067a6b58ce66f12ba927a2b"
TOLERANCE = 0.01


def scaled_distance(local, scale, e1, e2):
    """g / |grad g|, local (.., 3), scale (.., 3) and e1, e2 (..). Coordinates are divided by the largest one first,
    the powers would overflow otherwise, and g is homogeneous."""
    a = (local / scale).abs().clamp_min(1e-6)
    m = a.amax(-1, keepdim=True)
    ax, ay, az = (a / m).clamp_min(1e-6).unbind(-1)
    xy = (ax ** (2 / e2) + ay ** (2 / e2)).clamp_min(1e-20)  # the powers below may be negative
    f = (xy ** (e2 / e1) + az ** (2 / e1)).clamp_min(1e-20)
    g = f ** (e1 / 2)
    outer = f ** (e1 / 2 - 1)
    grad = torch.stack([outer * xy ** (e2 / e1 - 1) * ax ** (2 / e2 - 1),
                        outer * xy ** (e2 / e1 - 1) * ay ** (2 / e2 - 1),
                        outer * az ** (2 / e1 - 1)], -1) / scale
    return (m[..., 0] * g - 1) / grad.norm(dim=-1).clamp_min(1e-6)


class Variant(SuperQOptimization):
    def setup(self, loss, distance, tau, w_free, cap=None):
        """cap: truncates the voxel loss's distance on the surface samples, like SuperFlex's surface term."""
        self.loss_kind, self.distance, self.tau, self.w_free, self.cap = loss, distance, tau, w_free, cap
        return self

    def sdf_batch(self, points):
        if self.distance == "radial":
            return super().sdf_batch(points)
        local = (self.rotation().transpose(-2, -1) @ (points.unsqueeze(1) - self.translation.unsqueeze(-1))).transpose(-2, -1)
        e = self.exponents()
        return scaled_distance(local, self.scale().unsqueeze(2), e[..., 0, None], e[..., 1, None])

    def compute_losses(self, forward_out):
        if self.loss_kind == "iou":
            return super().compute_losses(forward_out)
        d = forward_out["sdfs"]
        occupied = self.occupancies.float()
        query = F.softplus(d[:, :self.M_points_iou] / self.tau)
        surface = d[:, self.M_points_iou:]
        if self.cap:
            surface = self.cap * torch.tanh(surface / self.cap)
            query = F.softplus(self.cap * torch.tanh(d[:, :self.M_points_iou] / self.cap) / self.tau)
        inside = ((occupied * query).sum(1) + F.softplus(surface / self.tau).sum(1)) / (occupied.sum(1) + self.M_points_surf)
        free = 1 - occupied
        outside = (free * F.softplus(-d[:, :self.M_points_iou] / self.tau)).sum(1) / free.sum(1)
        zero = torch.zeros_like(inside)
        return inside + self.w_free * outside, {"sdf": inside, "bbox": outside, "overlap": zero}


def kmeans_params(chairs, K):
    boxes, _ = zip(*[kmeans_boxes(c["surf"], K) for c in chairs])
    B = len(chairs)
    return {key: np.stack([b[key] for b in boxes]) for key in boxes[0]} | {
        "exponents": np.ones((B, K, 2)), "exist": np.ones((B, K)), "tapering": np.zeros((B, K, 2)),
        "bending": np.zeros((B, K, 6)), "names": np.arange(B), "assign_matrix": np.zeros((B, 1, K)),
        "pc": np.stack([c["surf"] for c in chairs])}


def fit(chairs, handler, cfg):
    surf = np.stack([c["surf"] for c in chairs])
    external = {"points": torch.tensor(surf, device=DEV),
                "points_iou": torch.tensor(np.stack([c["q"] for c in chairs]), device=DEV),
                "occupancies": torch.tensor(np.stack([c["occ"] for c in chairs]), device=DEV)}
    sq = Variant(pred_handler=handler, indices=list(range(len(chairs))), device=DEV, external_data=external,
                 cfg=OmegaConf.create({"tapering": False, "bending": False, "pruning": False}))
    sq.setup(cfg["loss"], cfg["distance"], cfg.get("tau", 0.01), cfg.get("w_free", 1.0))
    return optimize(sq, cfg.get("steps", 1000)) | {"exist": sq.exist_mask.cpu().numpy()}


def union_distance(x, r, b):
    """Scaled distance of x (M, 3) to the union of chair b's existing primitives."""
    keep = torch.tensor(r["exist"][b], device=DEV)
    R, t, s, e = [torch.tensor(r[k][b], device=DEV, dtype=torch.float32)[keep] for k in ("rotation", "translation", "scale", "exponents")]
    local = (x[None] - t[:, None]) @ R
    return scaled_distance(local, s[:, None], e[:, 0, None], e[:, 1, None]).amin(0)


@torch.no_grad()
def evaluate(r, chairs):
    """IoU, IoU without the query points within TOLERANCE of the surface, and surface recall at that tolerance."""
    rows = []
    for b, c in enumerate(chairs):
        q = torch.tensor(c["q"], device=DEV)
        occ = torch.tensor(c["occ"], device=DEV)
        pred = union_distance(q, r, b) < 0
        far = torch.tensor(c["far"], device=DEV)
        iou = (pred & occ).sum() / (pred | occ).sum().clamp_min(1)
        # with the shell around the surface ignored, what remains of the chair may be empty for thin chairs: count the
        # shell's occupied points as hit if the prediction reaches within TOLERANCE of them
        reached = union_distance(q[occ], r, b) < TOLERANCE
        free_hit = (pred & ~occ & far).sum()
        relaxed = reached.sum() / (occ.sum() + free_hit).clamp_min(1)
        d = union_distance(torch.tensor(c["surface"], device=DEV), r, b)
        rows.append({"iou": iou.item(), "relaxed_iou": relaxed.item(), "surface_recall": (d.abs() < TOLERANCE).float().mean().item()})
    return rows


def pick_chairs(seed=0):
    ids = (DATA / "test.lst").read_text().split()
    quality = json.loads((DATA.parent / "quality.json").read_text())["objects"]
    order = sorted((i for i in ids if i != TARGET), key=lambda i: quality[i]["occupancy"])
    rng = np.random.default_rng(seed)
    thin = list(rng.choice(order[:len(order) // 4], 15, replace=False))
    rest = list(rng.choice(order[len(order) // 4:], 16, replace=False))
    return [TARGET] + thin + rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids = pick_chairs()
    chairs = [load_chair(i) for i in ids]
    for i, c in zip(ids, chairs):
        c["surface"] = np.load(DATA / i / "pointcloud.npz")["points"].astype(np.float32)[::5]
        c["far"] = cKDTree(c["surface"]).query(c["q"], workers=-1)[0] > TOLERANCE

    model = SuperFlex(OmegaConf.load(CHECKPOINT / "config.yaml").superflex).to(DEV)
    model.load_state_dict(torch.load(CHECKPOINT / "ckpt.pt", map_location=DEV, weights_only=False)["model_state_dict"])
    model.eval()
    surf = np.stack([c["surf"] for c in chairs])

    grid = [{"start": start, "loss": "iou", "distance": "scaled"} for start in ("kmeans", "network")]
    for start in ("kmeans", "network"):
        for tau in (0.002, 0.005):
            for w_free in (20.0, 50.0):
                grid.append({"start": start, "loss": "vox", "distance": "radial", "tau": tau, "w_free": w_free})

    rows = []
    for cfg in grid:
        handler = (predict(model, surf, plain=True) if cfg["start"] == "network"
                   else PredictionHandler(kmeans_params(chairs, 10)))
        r = fit(chairs, handler, cfg)
        scores = evaluate(r, chairs)
        mean = lambda key, part: float(np.mean([s[key] for s in scores[part]]))
        row = cfg | {"target_iou": scores[0]["iou"], "target_relaxed": scores[0]["relaxed_iou"], "target_recall": scores[0]["surface_recall"]}
        for name, part in (("thin", slice(1, 16)), ("random", slice(16, 32))):
            row |= {f"{name}_{key}": mean(key, part) for key in ("iou", "relaxed_iou", "surface_recall")}
        rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"ids": ids, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
