"""Occupancy labels against independent references, before and after the fix for missing sheets.

    python occupancy.py --out ../results/occupancy.json [--limit 48] [--tolerance 0.005 0.0075]

Solid reference: ONet's surface samples, from watertight meshes, with outward normals. A query point is inside if it
lies behind the normal of its nearest sample. The mesh's own winding number is useless here (doubled faces cancel).
Shell reference: query points within BAND of 1M area-weighted samples of the ShapeNet mesh; its recall says how much of
the real surface the labels contain, zero-thickness parts included.
Fix: the fusion labels, plus every query point within a tolerance of the 100k surface samples.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "18_chair_rgbd_dataset" / "code"))
import build_dataset as dataset  # noqa: E402
from variants import DATA, DEV  # noqa: E402

ONET = Path("/home/sebastian/Documents/Uni/GF4Shape/data/ShapeNet/03001627")
BAND = 0.005


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


@torch.no_grad()
def compare(model_id, tolerances):
    verts, faces, *_ = dataset.load_mesh(dataset.CORE / model_id / "models" / "model_normalized.obj", DEV)
    z = np.load(DATA / model_id / "points.npz")
    q = z["points"].astype(np.float32)
    fusion = np.unpackbits(z["occupancies_fusion"] if "occupancies_fusion" in z else z["occupancies"])[:len(q)].astype(bool)
    dense = dataset.sample_surface(verts, faces, 1_000_000, torch.Generator(DEV).manual_seed(0))[0].cpu().numpy()
    shell = cKDTree(dense).query(q, workers=-1)[0] <= BAND
    onet = np.load(ONET / model_id / "pointcloud.npz")
    points, normals = onet["points"].astype(np.float32), onet["normals"].astype(np.float32)
    nearest = cKDTree(points).query(q, workers=-1)[1]
    solid = ((q - points[nearest]) * normals[nearest]).sum(1) < 0
    near = cKDTree(np.load(DATA / model_id / "pointcloud.npz")["points"].astype(np.float32)).query(q, workers=-1)[0]
    row = {"id": model_id, "solid": float(solid.mean()), "shell": float(shell.mean()), "fusion": float(fusion.mean()),
           "iou_fusion": iou(fusion, solid), "shell_recall_fusion": float((fusion & shell).sum() / max(shell.sum(), 1))}
    for t in tolerances:
        fixed = fusion | (near <= t)
        row |= {f"iou_{t}": iou(fixed, solid), f"shell_recall_{t}": float((fixed & shell).sum() / max(shell.sum(), 1)),
                f"occupied_{t}": float(fixed.mean())}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tolerance", type=float, nargs="+", default=[0.005, 0.0075])
    args = ap.parse_args()
    ids = [i for i in (DATA / "test.lst").read_text().split() if (ONET / i).is_dir()]
    ids = [ids[i] for i in np.random.default_rng(0).permutation(len(ids))[:args.limit]]
    rows = []
    for n, model_id in enumerate(ids):
        tic = time.time()
        rows.append(compare(model_id, args.tolerance))
        print(n + 1, len(ids), f"{time.time() - tic:.1f}s", flush=True)
        args.out.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
