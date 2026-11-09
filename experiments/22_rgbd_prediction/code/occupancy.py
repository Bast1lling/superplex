"""Per-instance occupancy: the voxels holding >= 1 of the instance's points, plus those its own points enclose.

    uv run --with libigl python occupancy.py [scene ...]       -> results/large_files/occupancy/<scene>.npz

On experiment 12's 4 cm grid (same origin and size as `gt_grid`), separately for every GT instance k:
1. surface: every voxel containing >= 1 point of k (a voxel can belong to several instances);
2. filled: every other voxel center in k's box (+ 1 voxel) where the fast winding number of k's oriented points
   (Barill et al. 2018, libigl; area per point pi r^2 / 16, r = distance to its 16th neighbour) is >= W.
No normal vote and no UNKNOWN class. Stored as flat voxel indices per instance (sorted), surface first.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parent / "12_rgbd_multiview" / "code"))
from views import gt  # noqa: E402

OUT = EXP / "results" / "large_files" / "occupancy"
RES, MARGIN, W, K = 0.04, 0.08, 0.5, 16


class Occupancy:
    """occ[k] = sorted flat voxel indices of instance k, surf[k] = its surface part; any = union over instances."""

    def __init__(self, z):
        self.lo, self.dims, self.res = z["lo"], z["dims"], float(z["res"])
        off, soff = z["offsets"], z["surf_offsets"]
        self.ids = z["ids"]
        self.occ = {int(k): z["flat"][off[i]:off[i + 1]] for i, k in enumerate(self.ids)}
        self.surf = {int(k): z["surf"][soff[i]:soff[i + 1]] for i, k in enumerate(self.ids)}
        self.any = np.unique(z["flat"])

    def flat(self, x):
        """Flat voxel index of points x (N, 3), -1 outside the grid."""
        i = np.round((x - self.lo) / self.res).astype(np.int64)
        inside = ((i >= 0) & (i < self.dims)).all(1)
        f = np.ravel_multi_index(tuple(np.clip(i, 0, self.dims - 1).T), self.dims)
        return np.where(inside, f, -1)

    def contains(self, k, x):
        return np.isin(self.flat(x), self.occ[k])

    def centers(self, flat):
        return self.lo + np.stack(np.unravel_index(flat, self.dims), 1) * self.res


def build(scene):
    import igl
    g = gt(scene)
    keep = g["inst"] >= 0
    P, N, inst = g["pos"][keep].astype(np.float64), g["nrm"][keep].astype(np.float64), g["inst"][keep]
    lo = P.min(0) - MARGIN
    dims = np.ceil((P.max(0) + MARGIN - lo) / RES).astype(int) + 1  # as grid.make_grid
    A = np.pi * cKDTree(P).query(P, K + 1, workers=-1)[0][:, -1] ** 2 / K
    ids, flats, surfs = [], [], []
    for k in np.unique(inst):
        m = inst == k
        vi = np.round((P[m] - lo) / RES).astype(int)
        surf = np.unique(np.ravel_multi_index(tuple(vi.T), dims))
        a, b = np.maximum(vi.min(0) - 1, 0), np.minimum(vi.max(0) + 1, dims - 1)
        box = np.stack(np.meshgrid(*[np.arange(x, y + 1) for x, y in zip(a, b)], indexing="ij"), -1).reshape(-1, 3)
        bf = np.ravel_multi_index(tuple(box.T), dims)
        q = ~np.isin(bf, surf)
        w = igl.fast_winding_number(P[m], N[m], A[m], lo + box[q] * RES)
        filled = bf[q][w >= W]
        ids.append(k), surfs.append(surf), flats.append(np.sort(np.concatenate([surf, filled])))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{scene}.npz", lo=lo, dims=dims, res=RES, ids=np.array(ids),
                        flat=np.concatenate(flats), offsets=np.cumsum([0] + [len(f) for f in flats]),
                        surf=np.concatenate(surfs), surf_offsets=np.cumsum([0] + [len(s) for s in surfs]))
    names = g["names"]
    rows = {}
    for k, s, f in zip(ids, surfs, flats):
        r = rows.setdefault(names[k], [0, 0])
        r[0] += len(s)
        r[1] += len(f)
    print(scene, "voxels per class, surface -> with winding fill:")
    for n, (s, f) in sorted(rows.items(), key=lambda x: -x[1][1])[:12]:
        print(f"   {n:16s} {s:8d} -> {f:8d}  (+{(f - s) / s:.0%})")


def load(scene):
    return Occupancy(np.load(OUT / f"{scene}.npz"))


if __name__ == "__main__":
    for s in sys.argv[1:] or ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]:
        build(s)
