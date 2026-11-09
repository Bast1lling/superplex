"""Fit exactly K plain superquadrics to every chair of a split with SuperFlex's objective.

    python fit_sqs.py --split test --k 4 --out /home/sebastian/data/ShapeNet-Chairs-RGBD/sq_fits

One process per (split, K): `torch.compile` in SuperQOptimization specializes on tensor shapes.
Init: K k-means boxes of the 4096 surface points. A cluster too small for a box still gets a tiny primitive,
so every chair keeps all K. No pruning, no tapering or bending. Loss, learning rates and loop are SuperFlex's;
w_bbox = K / 16 because it averages the bbox term over all slots, and the network has 16 of them.

Output `<out>/<split>/kKK.npz`, primitives in the dataset's frame: ids (N), scale (N, K, 3), exponents (N, K, 2),
rotation (N, K, 3, 3), translation (N, K, 3), per chair iou, loss, sdf, bbox, overlap, tiny_inits.
Finished batches are kept in `kKK/`, so a stopped run resumes.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "papers" / "superflex"))
from superflex.optimization import SuperQOptimization  # noqa: E402
from superflex.utils.predictions_handler import PredictionHandler  # noqa: E402

DATA = Path("/home/sebastian/data/ShapeNet-Chairs-RGBD/03001627")
DEV = "cuda"
SLOTS, STEPS, BATCH, MIN_SCALE = 16, 1000, 64, 0.005
RAW = ["raw_scale", "raw_exponents", "raw_rotation", "translation", "raw_tapering", "raw_bending"]
KEYS = ["scale", "exponents", "rotation", "translation", "iou", "loss", "sdf", "bbox", "overlap", "tiny_inits"]


def load_chair(model_id):
    surf = np.load(DATA / model_id / "pointcloud_4096.npz")["points"].astype(np.float32)
    query = np.load(DATA / model_id / "points.npz")
    q = query["points"].astype(np.float32)
    return {"surf": surf, "q": q, "occ": np.unpackbits(query["occupancies"])[:len(q)].astype(bool)}


def kmeans_boxes(surf, K):
    """K boxes around the k-means clusters of the surface points, and how many clusters were too small for one."""
    box = {"scale": np.full((K, 3), MIN_SCALE), "translation": np.zeros((K, 3)), "rotation": np.tile(np.eye(3), (K, 1, 1))}
    labels = KMeans(K, n_init=1, random_state=0).fit_predict(surf)
    tiny = 0
    for j in range(K):
        x = surf[labels == j]
        if len(x) < 4:
            box["translation"][j] = x.mean(0) if len(x) else surf[j]
            tiny += 1
            continue
        mu = x.mean(0)
        _, axes = np.linalg.eigh(np.cov((x - mu).T))
        axes[:, 2] *= np.sign(np.linalg.det(axes))
        local = (x - mu) @ axes
        box["translation"][j] = mu + axes @ ((local.min(0) + local.max(0)) / 2)
        box["scale"][j] = np.maximum((local.max(0) - local.min(0)) / 2, MIN_SCALE)
        box["rotation"][j] = axes
    return box, tiny


def fit_batch(chairs, K):
    B = len(chairs)
    boxes, tiny = zip(*[kmeans_boxes(c["surf"], K) for c in chairs])
    surf = np.stack([c["surf"] for c in chairs])
    handler = PredictionHandler({
        "names": np.arange(B), "pc": surf, "assign_matrix": np.zeros((B, 1, K)),
        **{key: np.stack([b[key] for b in boxes]) for key in boxes[0]},
        "exponents": np.ones((B, K, 2)), "exist": np.ones((B, K)),
        "tapering": np.zeros((B, K, 2)), "bending": np.zeros((B, K, 6))})
    external = {"points": torch.tensor(surf, device=DEV),
                "points_iou": torch.tensor(np.stack([c["q"] for c in chairs]), device=DEV),
                "occupancies": torch.tensor(np.stack([c["occ"] for c in chairs]), device=DEV)}
    cfg = OmegaConf.create({"tapering": False, "bending": False, "pruning": False, "w_bbox": K / SLOTS})
    sq = SuperQOptimization(pred_handler=handler, indices=list(range(B)), device=DEV, external_data=external, cfg=cfg)
    return optimize(sq, STEPS) | {"tiny_inits": np.array(tiny)}


def optimize(sq, steps):
    """SuperFlex's loop with the best state per chair; returns primitives in the dataset's frame and the scores."""
    optimizer = torch.optim.Adam(sq.get_param_groups())
    best_loss = torch.full((len(sq.indices),), float("inf"), device=DEV)
    best = {name: getattr(sq, name).detach().clone() for name in RAW}
    for _ in range(steps):
        optimizer.zero_grad()
        loss, _ = sq.compute_losses(sq.forward())
        loss.mean().backward()
        with torch.no_grad():  # before the step: these parameters gave this loss
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            for name in RAW:
                best[name][improved] = getattr(sq, name)[improved]
        optimizer.step()

    with torch.no_grad():
        for name in RAW:
            getattr(sq, name).copy_(best[name])
        out = sq.forward()
        loss, terms = sq.compute_losses(out)
        pred = out["sdfs"][:, :sq.M_points_iou] < 0
        iou = (pred & sq.occupancies).sum(1) / (pred | sq.occupancies).sum(1).clamp_min(1)
        norm_t, norm_s = sq.normalization_translation[:, None], sq.normalization_scale[:, None, None]
        result = {"scale": sq.scale() * norm_s, "exponents": sq.exponents(), "rotation": sq.rotation(),
                  "translation": sq.translation * norm_s + norm_t, "iou": iou, "loss": loss, "sdf": terms["sdf"],
                  "bbox": terms["bbox"], "overlap": terms["overlap"]}
    return {key: value.cpu().numpy() for key, value in result.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    final = args.out / args.split / f"k{args.k:02d}.npz"
    if final.exists():
        return
    shards = final.with_suffix("")
    shards.mkdir(parents=True, exist_ok=True)
    ids = (DATA / f"{args.split}.lst").read_text().split()[:args.limit]
    for start in range(0, len(ids), BATCH):
        shard = shards / f"{start:05d}.npz"
        if shard.exists():
            continue
        tic = time.time()
        batch = ids[start:start + BATCH]
        result = fit_batch([load_chair(i) for i in batch], args.k)
        tmp = shard.with_suffix(".tmp.npz")
        np.savez(tmp, **result)
        tmp.rename(shard)
        print(f"{args.split} K={args.k} {start + len(batch)}/{len(ids)}  {time.time() - tic:.1f}s  "
              f"iou {result['iou'].mean():.3f}  nan {int(np.isnan(result['loss']).sum())}", flush=True)

    parts = [np.load(p) for p in sorted(shards.glob("[0-9]*.npz"))]
    np.savez(final, ids=np.array(ids), **{key: np.concatenate([p[key] for p in parts]) for key in KEYS})
    shutil.rmtree(shards)


if __name__ == "__main__":
    main()
