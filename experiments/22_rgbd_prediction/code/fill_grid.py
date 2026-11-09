"""Iterative occupancy fill of experiment 12's GT grid: UNKNOWN voxels (behind a surface, > 10 cm from it) grow into
an instance only where it encloses them.

    uv run python fill_grid.py [scene ...] [--thresh 10]      -> per class: solid voxels before / after

Rule, repeated until nothing changes: an UNKNOWN voxel becomes instance k if >= `thresh` of its 26 neighbours are solid
voxels of k. Behind a flat surface a voxel touches at most 9 of them (one 3 x 3 face), so with thresh >= 10 flat walls
never grow; concave corners (up to 15) and enclosed cavities fill from their edges inwards. FREE voxels never change.
Remaining UNKNOWN voxels count as outside.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "12_rgbd_multiview" / "code"))
from grid import UNKNOWN, gt_grid  # noqa: E402
from views import gt  # noqa: E402

OFFSETS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])


def fill(label, thresh=10, max_iter=1000):
    """label: grid of instance ids (>= 0), FREE, UNKNOWN. Returns (filled copy, iterations)."""
    L = np.pad(label, 1, constant_values=-1)  # border = FREE, so the fill never leaves the grid
    for it in range(max_iter):
        cand = np.argwhere((L == UNKNOWN) & binary_dilation(L >= 0, np.ones((3, 3, 3), bool)))
        if len(cand) == 0:
            break
        nb = L[tuple((cand[:, None] + OFFSETS[None]).transpose(2, 0, 1))]  # (N, 26) neighbour labels
        same = (nb[:, :, None] == nb[:, None, :]) & (nb[:, None, :] >= 0)
        count = same.sum(2)  # per neighbour: how many neighbours share its (solid) label
        best = count.argmax(1)
        n, lab = count[np.arange(len(nb)), best], nb[np.arange(len(nb)), best]
        grow = n >= thresh
        if not grow.any():
            break
        L[tuple(cand[grow].T)] = lab[grow]
    return L[1:-1, 1:-1, 1:-1], it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["room_0", "office_0", "office_1", "office_3"])
    ap.add_argument("--thresh", type=int, default=10)
    a = ap.parse_args()
    for s in a.scenes:
        grid, names = gt_grid(s), gt(s)["names"]
        L = grid.label_np
        F, it = fill(L, a.thresh)
        print(f"{s}: {it} iterations, UNKNOWN {(L == UNKNOWN).sum()} -> {(F == UNKNOWN).sum()}")
        cls = {}
        for k in np.unique(L[L >= 0]):
            c = cls.setdefault(names[k], [0, 0])
            c[0] += int((L == k).sum())
            c[1] += int((F == k).sum())
        for n, (b, f) in sorted(cls.items(), key=lambda x: -x[1][1])[:12]:
            print(f"   {n:20s} {b:8d} -> {f:8d}  (+{(f - b) / b:.0%})")


if __name__ == "__main__":
    main()
