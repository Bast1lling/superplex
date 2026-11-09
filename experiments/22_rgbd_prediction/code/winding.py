"""Generalized winding number at every UNKNOWN voxel of experiment 12's GT grid (fast winding numbers for oriented
point clouds, Barill et al. 2018, libigl), as an inside test for occupancy.

    uv run --with libigl python winding.py [scene ...]

Points: the labelled 1 cm GT points with their normals, area per point pi r^2 / 16 with r the distance to its 16th
nearest neighbour. Per UNKNOWN voxel center:
- w_inst = max over instances k of the winding number of k's points alone (instances whose box + 0.5 m contains the
  voxel), label = that k;
- w_scene = winding number of all labelled points together (an object standing on the floor is closed by the floor).
w ~ 1 inside a closed surface, ~ 0 outside, about 0.5 just behind a large flat wall.
Saves results/large_files/winding/<scene>.npz (UNKNOWN voxel indices, w_inst, label, w_scene) and prints per class how
many UNKNOWN voxels each threshold adds.
"""

import sys
from pathlib import Path

import igl
import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parent / "12_rgbd_multiview" / "code"))
from grid import UNKNOWN, gt_grid  # noqa: E402
from views import gt  # noqa: E402

OUT = EXP / "results" / "large_files" / "winding"
K, MARGIN = 16, 0.5


def winding(scene):
    g, grid = gt(scene), gt_grid(scene)
    keep = g["inst"] >= 0
    P, N, inst = g["pos"][keep].astype(np.float64), g["nrm"][keep].astype(np.float64), g["inst"][keep]
    from scipy.spatial import cKDTree
    r = cKDTree(P).query(P, K + 1, workers=-1)[0][:, -1]
    A = np.pi * r**2 / K
    idx = np.argwhere(grid.label_np == UNKNOWN)
    Q = grid.lo_np.astype(np.float64) + idx * grid.res
    w_scene = igl.fast_winding_number(P, N, A, Q)
    w_inst, label = np.zeros(len(Q)), np.full(len(Q), -1)
    for k in np.unique(inst):
        m = inst == k
        lo, hi = P[m].min(0) - MARGIN, P[m].max(0) + MARGIN
        q = np.flatnonzero(((Q >= lo) & (Q <= hi)).all(1))
        if len(q) == 0:
            continue
        w = igl.fast_winding_number(P[m], N[m], A[m], Q[q])
        better = w > w_inst[q]
        w_inst[q[better]], label[q[better]] = w[better], k
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{scene}.npz", idx=idx.astype(np.int16), w_inst=w_inst.astype(np.float32),
                        label=label.astype(np.int16), w_scene=w_scene.astype(np.float32))
    # report: solid voxels per class, and how many UNKNOWN voxels each rule would add
    names = g["names"]
    L = grid.label_np
    print(f"{scene}: {len(Q)} UNKNOWN voxels, w_inst quantiles 50/90/99 {np.quantile(w_inst, [.5, .9, .99]).round(2)}, "
          f"w_scene {np.quantile(w_scene, [.5, .9, .99]).round(2)}")
    rows = {}
    for k in np.unique(L[L >= 0]):
        n = names[k]
        row = rows.setdefault(n, np.zeros(5, int))
        row[0] += (L == k).sum()
        for j, (w, t) in enumerate([(w_inst, 0.5), (w_inst, 0.75), (w_scene, 0.5), (w_scene, 0.75)]):
            row[1 + j] += ((label == k) & (w >= t)).sum()
    print(f"   {'class':16s} {'solid':>8s}  added: inst>=0.5 inst>=0.75 scene>=0.5 scene>=0.75")
    for n, row in sorted(rows.items(), key=lambda x: -x[1][0])[:14]:
        print(f"   {n:16s} {row[0]:8d}  " + "  ".join(f"{v / row[0]:+9.0%}" for v in row[1:]))


if __name__ == "__main__":
    for s in sys.argv[1:] or ["room_0", "office_0", "office_1", "office_3"]:
        winding(s)
