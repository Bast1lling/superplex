"""adaptive: like instances.py, but every instance gets as many SQs as its shape needs.

    python adaptive.py [--res 0.02] [scene ...]

Search: every part (initially one per instance) gets one SQ, fitted with the instances loss (positives = the part's
voxels, negatives = free voxels in its box, everything else ignored). A part whose SQ has a tolerant IoU < Q is split
in two (2-means of its voxels) and both halves are fitted again; a part is only split if both halves keep >= MIN_VOL
(512 cm^3 = 8 voxels at 4 cm, 64 at 2 cm). Tolerant IoU: a part voxel within TOL (half a voxel) outside the SQ counts
as covered, a free voxel only counts as wrongly covered if it is more than TOL inside (thin parts could never pass the
plain voxel IoU).
Refinement: all SQs of the search are optimised together with the instances loss (union of each instance's SQs).
"""

import numpy as np
import torch
from sklearn.cluster import KMeans

from data import DEV, cli, load_points, voxelize
from instances import STEPS, Targets, instance_sets, target_loss, union_sdf
from sq import SQs, optimize, save

Q, MIN_VOL, SEARCH_STEPS, N_EVAL = 0.7, 8 * 0.04 ** 3, 300, 4096


@torch.no_grad()
def tolerant_iou(model, targets, members, tol):
    """Per target: covered voxels / (voxels + free voxels inside), both estimated from N_EVAL samples."""
    pos, neg, free = targets.sample(N_EVAL, N_EVAL)
    covered = (union_sdf(model, members, pos) < tol).float().mean(1) * targets.count
    box_vox = (targets.hi - targets.lo).prod(1) / targets.grid.res ** 3
    leaked = ((union_sdf(model, members, neg) < -tol) & free).float().mean(1) * box_vox
    return covered / (targets.count + leaked)


def main(scene):
    grid = voxelize(load_points(scene))
    min_vox = round(MIN_VOL / grid.res ** 3)
    parts = list(instance_sets(grid).items())  # (instance, voxel indices)
    done = []  # (instance, model, index) of accepted SQs
    while parts:
        targets = Targets(grid, [v for _, v in parts])
        model = SQs.from_boxes([grid.centers(v).cpu().numpy() for _, v in parts], pad=grid.res / 2,
                               instance=torch.tensor([k for k, _ in parts], device=DEV))
        members = torch.arange(len(parts), device=DEV)[:, None]
        optimize(model, lambda: target_loss(model, targets, members), SEARCH_STEPS)
        iou = tolerant_iou(model, targets, members, grid.res / 2).tolist()
        split = []
        for i, ((k, v), q) in enumerate(zip(parts, iou)):
            half = KMeans(2, n_init=1, random_state=0).fit_predict(v) if q < Q and len(v) >= 2 * min_vox else None
            if half is None or min((half == 0).sum(), (half == 1).sum()) < min_vox:
                done.append((model, i))
            else:
                split += [(k, v[half == 0]), (k, v[half == 1])]
        print(f"{len(parts)} parts fitted, {len(parts) - len(split) // 2} accepted, {len(split) // 2} split", flush=True)
        parts = split

    model = SQs.select(done)
    sets = instance_sets(grid)
    idx = {k: (model.instance == k).nonzero()[:, 0] for k in sets}
    members = torch.full((len(sets), max(len(i) for i in idx.values())), -1, device=DEV)
    for row, i in enumerate(idx.values()):
        members[row, :len(i)] = i
    targets = Targets(grid, list(sets.values()))
    save(optimize(model, lambda: target_loss(model, targets, members), STEPS), scene, "adaptive")


if __name__ == "__main__":
    for scene in cli():
        main(scene)
