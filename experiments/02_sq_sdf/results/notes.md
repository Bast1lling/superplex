# 02 SQ SDF: notes

Candidates (see `code/sdf.py`): radial, gauge_lb, taubin, newton_radial, newton_grid. Reference = LM (60 iters)
from the best of a 96x96 direction grid and from the radial projection.

## Sanity checks
- Sphere: all methods exact (reference vs |q| - 1: 5e-7).
- Newton methods never fall below the reference (LM is monotone), so the reference is consistent.
- gauge_lb with min(a) as Lipschitz constant is NOT a lower bound for pointy shapes (eps > 1): octahedron
  (a = 1, .3, .1, eps = 2, 2) only 63% of voxels. Fixed by using the inradius 1 / max_u G(u) (32x32 grid,
  shrunk by 0.2%): 0 violations over 200 random SQs x 4000 points.

## Accuracy snapshots (viewer API, MAE / max abs error, all voxels | near surface |GT| < 0.1 max(a))
| shape | radial | gauge_lb | taubin | newton_radial | newton_grid |
|---|---|---|---|---|---|
| rounded box a=(1,.6,.4) eps=.4 | .068/.52 \| .015/.057 | .131/.49 \| .015/.06 | .046/.42 \| .001/.012 | .001/.41 \| 0/0 | 1e-4/.011 \| 0/0 |
| box a=(1,.6,.4) eps=.1 | .091/.62 \| .023/.12 | .098/.46 \| .012/.06 | .044/.59 \| .002/.053 | .005/.58 \| 2e-4/.017 | 5e-4/.061 \| 2e-4/.017 |

- radial always overestimates |d| (never a lower bound); its near-surface error is the largest of all.
- taubin: ~10x better than radial near the surface at 2x its cost; degrades away from the surface.
- newton_radial: exact for most voxels but large outliers inside (wrong local minimum near the medial axis).
- newton_grid: fixes those outliers; residual near-edge error for box-like shapes after 10 iterations.

## Speed (RTX 5090, dense M x N, see benchmark.md)
- radial / gauge_lb ~6-7 G evals/s, taubin ~3.4 G evals/s, Newton ~0.07-0.08 G evals/s (~80x slower).
- Newton at small sizes is dominated by kernel-launch overhead (~5 ms floor for 10 iterations);
  torch.compile / a fused kernel would likely help a lot there (not tried yet).
