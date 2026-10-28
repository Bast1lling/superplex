"""instances: one SQ per GT instance, fitted with the volume loss per instance.

    python instances.py [--res 0.02] [scene ...]

For every instance k, samples come from its own box (+20 cm): positives = its voxels, negatives = free voxels in the
box and samples beyond the grid (otherwise SQs grow out of the scan for free). Voxels of other instances and DEEP
voxels are ignored (weight 0, so SQs may overlap them: floor under a sofa, wall behind a picture). Loss = mean over instances of [mean softplus(d / tau) on positives + mean softplus(-d / tau) on negatives],
with d = the union of the instance's SQs, so a small object counts as much as a wall.
"""

import numpy as np
import torch
import torch.nn.functional as F

from data import DEV, cli, load_points, voxelize
from sq import SQs, optimize, save, sdf
from volume import TAU

STEPS, MARGIN, N_POS, N_NEG = 2000, 0.2, 256, 256


class Targets:
    """Voxel sets that SQs are fitted to (instances here, parts in adaptive.py), sampled all at once."""

    def __init__(self, grid, sets):
        self.grid = grid
        x = [grid.centers(s) for s in sets]
        self.vox = torch.cat(x)
        self.count = torch.tensor([len(v) for v in x], device=DEV)
        self.start = torch.cumsum(self.count, 0) - self.count
        self.lo = torch.stack([v.min(0).values for v in x]) - MARGIN
        self.hi = torch.stack([v.max(0).values for v in x]) + MARGIN

    def sample(self, n_pos=N_POS, n_neg=N_NEG):
        """Positives (T, n_pos, 3) from each set's voxels; uniform points (T, n_neg, 3) in its box and whether they are empty."""
        T = len(self.count)
        pos = self.vox[self.start[:, None] + (torch.rand(T, n_pos, device=DEV) * self.count[:, None]).long()]
        neg = self.lo[:, None] + torch.rand(T, n_neg, 3, device=DEV) * (self.hi - self.lo)[:, None]
        return pos, neg, self.grid.empty(neg)


def union_sdf(model, members, x):
    """Distance of samples x (T, n, 3) to the union of each target's SQs; members (T, J) SQ indices, -1 = padding."""
    p = {k: v[members.clamp_min(0)][:, None] for k, v in model.params().items()}
    return sdf(x[:, :, None], p).masked_fill(members[:, None] < 0, float("inf")).min(-1).values


def target_loss(model, targets, members):
    pos, neg, free = targets.sample()
    inside = F.softplus(union_sdf(model, members, pos) / TAU).mean(1)
    outside = (F.softplus(-union_sdf(model, members, neg) / TAU) * free).sum(1) / free.sum(1).clamp_min(1)
    return (inside + outside).mean()


def instance_sets(grid):
    """Voxel indices of every instance (instances that lost all their voxels to neighbors are skipped)."""
    sets = {k: np.argwhere(grid.label_np == k) for k in range(grid.label_np.max() + 1)}
    return {k: s for k, s in sets.items() if len(s) >= 4}


def main(scene):
    grid = voxelize(load_points(scene))
    sets = instance_sets(grid)
    targets = Targets(grid, list(sets.values()))
    model = SQs.from_boxes([grid.centers(s).cpu().numpy() for s in sets.values()],
                           instance=torch.tensor(list(sets), device=DEV), pad=grid.res / 2)
    members = torch.arange(len(sets), device=DEV)[:, None]
    save(optimize(model, lambda: target_loss(model, targets, members), STEPS), scene, "instances")


if __name__ == "__main__":
    for scene in cli():
        main(scene)
