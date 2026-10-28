"""volume: a fixed number of SQs fitted to the solid voxels, without labels. Two variants, volume and volume_overlap:

    python volume.py [--res 0.02] [scene ...]

Same SQs, initialisation and optimiser as surface.py; only the target differs. Logistic occupancy loss of the union
(d = min_j d_j): solid voxels should be inside, softplus(d / tau), empty space outside, softplus(-d / tau)
(= cross-entropy with inside probability sigmoid(-d / tau)). Per step: N solid voxel centers, and N uniform points in
the grid box + 50 cm, of which the empty ones (free voxels or beyond the grid) are negatives; DEEP voxels are ignored.
- overlap: + OVERLAP * (sum_j p_j - max_j p_j) on the solid samples, p_j = sigmoid(-d_j / tau): every SQ beyond the
           first that covers a solid voxel costs OVERLAP.
"""

import torch
import torch.nn.functional as F

from data import DEV, cli, load_points, voxelize
from sq import K, kmeans_init, optimize, save, sdf

STEPS, N_VOXELS, TAU, PAD, OVERLAP = 2000, 16384, 0.02, 0.5, 0.1
VARIANTS = {"volume": {}, "volume_overlap": {"overlap": OVERLAP}}


def fit(scene, pts, grid, overlap=0.0):
    solid = grid.centers(torch.nonzero(grid.label >= 0))
    lo, hi = grid.lo - PAD, grid.lo + (grid.dims - 1) * grid.res + PAD
    model = kmeans_init(pts["pos"], K[scene])

    def loss():
        p = model.params()
        d_in = sdf(solid[torch.randint(len(solid), (N_VOXELS,), device=DEV)][:, None], p)
        x = lo + torch.rand(N_VOXELS, 3, device=DEV) * (hi - lo)
        empty = grid.empty(x)
        d_out = sdf(x[empty][:, None], p).min(1).values
        total = F.softplus(d_in.min(1).values / TAU).mean() + F.softplus(-d_out / TAU).mean()
        if overlap:
            inside = torch.sigmoid(-d_in / TAU)
            total = total + overlap * (inside.sum(1) - inside.max(1).values).mean()
        return total

    return optimize(model, loss, STEPS)


def main(scene):
    pts = load_points(scene)
    grid = voxelize(pts)
    for name, kw in VARIANTS.items():
        save(fit(scene, pts, grid, **kw), scene, name)


if __name__ == "__main__":
    for scene in cli():
        main(scene)
