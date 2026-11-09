"""Fits stay stuck with the parts their start missed. Grow: add a primitive where the surface is still uncovered.

    python grow.py --out ../results/grow.json --fits /home/sebastian/data/ShapeNet-Chairs-RGBD/sq_fits/exp20

Plain SQs, IoU loss, scaled distance. ROUNDS times: fit, cluster the surface samples further than TOLERANCE outside of
all primitives (DBSCAN) and put one PCA box on the largest cluster. Then SuperFlex's pruning and a last fit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from superflex.utils.transforms import mat2quat
from variants import (CHECKPOINT, DATA, DEV, TOLERANCE, PredictionHandler, SuperFlex, Variant, evaluate, kmeans_params,
                      load_chair, optimize, pick_chairs, predict)

SLOTS, ROUNDS, ROUND_STEPS, LAST_STEPS = 24, 6, 250, 500
MIN_CLUSTER = 8


def pad(handler, slots):
    """Copies of the handler's arrays with `slots` primitives, the additional ones not existing."""
    def grow(a, fill):
        extra = np.broadcast_to(fill, (a.shape[0], slots - a.shape[1]) + a.shape[2:])
        return np.concatenate([a, extra], 1)
    return PredictionHandler({
        "names": handler.names, "pc": handler.pc, "assign_matrix": np.zeros((len(handler.names), 1, slots)),
        "scale": grow(handler.scale, 1.0), "translation": grow(handler.translation, 0.0),
        "rotation": grow(handler.rotation, np.eye(3)), "exponents": grow(handler.exponents, 1.0),
        "exist": grow(handler.exist, 0.0), "tapering": grow(handler.tapering, 0.0), "bending": grow(handler.bending, 0.0)})


def make_sq(chairs, handler, loss="iou", distance="scaled", tau=0.01, w_free=1.0):
    external = {"points": torch.tensor(np.stack([c["surf"] for c in chairs]), device=DEV),
                "points_iou": torch.tensor(np.stack([c["q"] for c in chairs]), device=DEV),
                "occupancies": torch.tensor(np.stack([c["occ"] for c in chairs]), device=DEV)}
    sq = Variant(pred_handler=handler, indices=list(range(len(chairs))), device=DEV, external_data=external,
                 cfg=OmegaConf.create({"tapering": False, "bending": False, "pruning": False}))
    return sq.setup(loss, distance, tau, w_free)


@torch.no_grad()
def add_primitives(sq):
    """One new primitive per chair, on its largest cluster of uncovered surface points. Returns how many were added."""
    surf = sq.points[:, sq.M_points_iou:]
    d = sq.sdf_batch(surf.transpose(1, 2).contiguous())
    d[~sq.exist_mask.unsqueeze(-1).expand_as(d)] = float("inf")
    uncovered = d.amin(1) > TOLERANCE
    added = 0
    for b in range(surf.shape[0]):
        x = surf[b][uncovered[b]].cpu().numpy()
        free = (~sq.exist_mask[b]).nonzero()
        if len(x) < MIN_CLUSTER or not len(free):
            continue
        labels = DBSCAN(eps=0.03, min_samples=3).fit_predict(x)
        if labels.max() < 0:
            continue
        x = x[labels == np.bincount(labels[labels >= 0]).argmax()]
        if len(x) < MIN_CLUSTER:
            continue
        mu = x.mean(0)
        _, axes = np.linalg.eigh(np.cov((x - mu).T))
        axes[:, 2] *= np.sign(np.linalg.det(axes))
        local = (x - mu) @ axes
        j = free[0, 0]
        scale = np.maximum((local.max(0) - local.min(0)) / 2, 0.005)
        sq.translation[b, j] = torch.tensor(mu + axes @ ((local.min(0) + local.max(0)) / 2), device=DEV)
        sq.raw_scale[b, j] = torch.log(torch.tensor(scale, device=DEV, dtype=torch.float32) - sq.minS)
        sq.raw_rotation[b, j] = mat2quat(torch.tensor(axes, device=DEV, dtype=torch.float32)[None])[0]
        sq.raw_exponents[b, j] = 0.0
        sq.exist_mask[b, j] = True
        added += 1
    return added


def fit_grown(chairs, handler):
    sq = make_sq(chairs, pad(handler, SLOTS))
    for _ in range(ROUNDS):
        optimize(sq, ROUND_STEPS)
        if not add_primitives(sq):
            break
    optimize(sq, ROUND_STEPS)
    with torch.no_grad():
        sq.exist_mask &= ~sq.prune()
    return optimize(sq, LAST_STEPS) | {"exist": sq.exist_mask.cpu().numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fits", type=Path, required=True)
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
    start = {"network": lambda: predict(model, surf, plain=True),
             "kmeans": lambda: PredictionHandler(kmeans_params(chairs, 10))}

    runs = {"network_scaled": lambda: dict(sq=make_sq(chairs, start["network"]())),
            "network_voxel": lambda: dict(sq=make_sq(chairs, start["network"](), "vox", "radial", 0.005, 50.0)),
            "kmeans_scaled": lambda: dict(sq=make_sq(chairs, start["kmeans"]()))}
    args.fits.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, make in runs.items():
        sq = make()["sq"]
        r = optimize(sq, 2250) | {"exist": sq.exist_mask.cpu().numpy()}
        rows.append(report(name, r, chairs, ids, args.fits))
    for name in start:
        rows.append(report(f"{name}_grown", fit_grown(chairs, start[name]()), chairs, ids, args.fits))
    args.out.write_text(json.dumps(rows, indent=1))


def report(name, r, chairs, ids, folder):
    np.savez(folder / f"{name}.npz", ids=np.array(ids), **r)
    s = evaluate(r, chairs)
    count = r["exist"].sum(1)
    row = {"name": name, "target": s[0], "target_count": int(count[0])}
    for group, part in (("thin", range(1, 16)), ("random", range(16, 32))):
        row[group] = {key: float(np.mean([s[i][key] for i in part])) for key in s[0]} | {"count": float(count[list(part)].mean())}
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    main()
