# 10 — Adaptive complexity: complex shapes get many primitives, simple ones one

**Question.** Experiment 08 decides structure by *marginal gain*: an extra SQ must raise the instance's IoU by
δ = 0.03 (and a cap of K = 8 SQs per instance). Complex objects (plants, tables with legs, chairs, lamps) gain
less than δ per extra SQ and stay at 1–5 primitives (office_0 plant: 2 SQs, vIoU 0.38). How can the number of
primitives follow the shape's complexity instead — a plant up to ~100, a wall 1?

**TL;DR.**
- **Decide per part, not per instance: "is this part explained by ONE SQ well enough?"** (part IoU ≥ q).
  Top-down splitting (2-means + connected components) until every part passes, then merging adjacent parts whose
  union still passes. With a **discretization-tolerant IoU** (±½ voxel) this gives the requested
  behaviour: at q = 0.8 the office_0 plant goes 2 → **26 SQs** (vIoU 0.38 → 0.68), tables 2 → 8.5 (0.76 → 0.84),
  chairs 3 → 5, while walls / doors / lamps / plugs / pillar stay at **1** (94 SQs for 40 instances); at the
  default q = 0.7: plant 19, tables 6, 74 SQs, 90 % plausible counts. ~20 s per scene on the GPU (08: ~20 min CPU).
- The tolerant IoU is essential: with the strict voxel IoU thin parts (chair legs, table rims: 1–2 voxels) can
  never pass, the search splits down to the minimum size (298 SQs: pillar 16, chair 18, rug 5).
- **An absolute price per primitive (rate–distortion, λ voxels) allocates by size, not complexity**: at the same
  budget the rug gets 8–13 SQs, sofas 6, walls up to 5, but the plant only 9 — a large, slightly curved rug has more
  absolute error than the whole plant. The per-part quality threshold is scale-invariant, which is what makes the
  allocation follow complexity. (This also explains 08: its relative δ + K cap starve complex objects, an
  absolute λ would feed large ones.)
- Convex decomposition (CoACD) is the wrong signal on voxel input: staircase surfaces look concave everywhere
  (threshold 0.05: sofa 61, door 59 parts, 930 SQs); with a coarse threshold + merging it collapses the plant to 5.
- Bottom-up merging from 20 cm cells gets stuck (907 SQs on room_0): greedy pairwise merges create irregular parts
  that no single SQ fits, so they never merge further. Top-down first, then merge ("both") is best.
- Open issue: **ring / frame-shaped flat instances** (room_1: the visible rug is a ring around the bed, the floor
  around it) cannot be one slab under a strict per-part threshold → rug 26, floor 16 SQs at q = 0.8 (1 / 3 at
  q = 0.7). Needs an amodal rule for them (08's planar fill of the hull apparently did not cover these) or a
  per-class prior.

Runs and renders: `results/runs/<run>/render.jpg`; start with `office0_both_q08_tol`, `room0_both_q08_tol`,
`room1_both_q08_tol`. Interactive viewer: `uv run python experiments/10_adaptive_complexity/viewer/server.py` →
http://localhost:8010 (08's viewer incl. voxel layer, serving `results/viewer_runs`).

## Method (`code/partfit.py`, `code/decomp.py`)
Same volume, targets (incl. 08's planar fill), SQ parameterization and loss as experiment 08 v9, but on the GPU and
reorganized around **sets**: a set is a voxel part that gets one SQ (or a group with several SQs for the final joint
refinement). Negatives are uniform samples in the set's box (+20 cm), weighted by the voxel label: own 0, other part of
the same instance 0.1, other instance 0.1, free 1, unknown 0.03, exterior 1. Thousands of sets are fitted in parallel
(300 Adam steps, ~1–3 s per batch).

Decision quantity: hard IoU of the part's SQ with **tolerance** `tol` = ½ voxel: a target voxel counts as covered
if its center is within 2 cm outside the SQ, a negative only if it is > 2 cm inside.

Methods (`--set method=...`):
- `topdown`: one SQ per open part; parts with IoU < q are split (2-means on voxel centers, then connected
  components, components < 8 voxels attached to the nearest piece), down to 16 voxels / depth 10.
- `bottomup`: 20 cm cells per instance; repeatedly fit one SQ to the union of every adjacent pair of parts (same
  instance, 26-neighborhood) and merge the best disjoint pairs with IoU ≥ q.
- `both` (**default**): `topdown`, then `bottomup`-style merging of the leaves (undoes arbitrary 2-means cuts).
- `rd` / `rd_merge`: top-down where a split is kept iff the children's absolute tolerant error (missed + weighted
  leaked voxels) is lower than the parent's by more than λ voxels.
- `coacd` / `coacd_merge`: parts from CoACD (`code/convex.py`) on the marching-cubes mesh of the instance voxels.

Afterwards (all methods): one SQ per final part → joint refinement of each instance's union (400 steps) → prune SQs
whose removal costs < 0.01 tolerant IoU.

## Evaluation
`code/compare.py` prints per-class mean K / strict vIoU for several runs. vIoU is scored like 08's `reeval.py`
(strict weights: free 1, other 0.3, unknown / exterior 0.1, no planar completion, exact centers); runs of 08 are
re-scored with the same code, so all numbers below are comparable (they differ slightly from 08's own tables
because the negatives are sampled). `plausible` = share of instances whose K is in 08's hand-written per-class
range (`08/code/plausible.py`; note it allows only 1–2 SQs for plants, so the target behaviour here *lowers* it).

## Results

Runs are named `<scene>_<method>_q<q>[_tol]` (`_tol`: tolerant decisions; without: strict voxel IoU).

| run | SQs | vIoU | prec | rec | plausible | over | K indoor-plant | K table | K chair | K lamp | K wall | K rug | K floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 08:office0_v9_exterior | 57 | 0.786 | 0.824 | 0.939 | 0.93 | 0.07 | 2.0 | 2.0 | 3.0 | 1.0 | 1.2 | 1.0 | 1.0 |
| office0_both_q07_tol | 74 | 0.780 | 0.848 | 0.906 | 0.90 | 0.10 | 19.0 | 6.0 | 3.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| office0_both_q08_tol | 94 | 0.787 | 0.853 | 0.911 | 0.85 | 0.15 | 26.0 | 8.5 | 5.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| office0_both_q09_tol | 190 | 0.812 | 0.868 | 0.925 | 0.60 | 0.40 | 51.0 | 12.0 | 5.5 | 1.0 | 4.2 | 6.0 | 7.0 |
| office0_topdown_q08_tol | 128 | 0.792 | 0.856 | 0.913 | 0.85 | 0.15 | 43.0 | 11.5 | 5.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| office0_both_q08 | 298 | 0.811 | 0.867 | 0.927 | 0.70 | 0.30 | 137.0 | 21.5 | 18.0 | 1.0 | 1.0 | 5.0 | 2.0 |
| office0_rd_lam100 | 89 | 0.788 | 0.850 | 0.912 | 0.78 | 0.23 | 9.0 | 4.5 | 3.0 | 1.0 | 1.8 | 8.0 | 2.0 |
| office0_rd_lam30 | 139 | 0.794 | 0.856 | 0.915 | 0.70 | 0.30 | 19.0 | 5.5 | 3.5 | 1.0 | 5.0 | 13.0 | 2.0 |
| office0_coacd_cx005 | 930 | 0.830 | 0.883 | 0.933 | 0.38 | 0.62 | 62.0 | 44.5 | 44.0 | 19.5 | 21.8 | 16.0 | 7.0 |
| office0_coacdmerge_cx025 | 56 | 0.774 | 0.843 | 0.900 | 0.90 | 0.10 | 5.0 | 3.0 | 4.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 08:room0_v9_exterior | 116 | 0.754 | 0.785 | 0.951 | 0.88 | 0.12 | 5.0 | 1.6 | 4.0 | 2.0 | 2.5 | 1.0 | 2.0 |
| room0_both_q07_tol | 117 | 0.733 | 0.797 | 0.878 | 0.95 | 0.05 | 5.0 | 2.2 | 2.0 | 3.5 | 2.5 | 1.0 | 1.0 |
| room0_both_q08_tol | 141 | 0.745 | 0.804 | 0.887 | 0.94 | 0.06 | 7.0 | 2.2 | 3.0 | 4.7 | 2.5 | 1.0 | 2.0 |
| room0_both_q09_tol | 263 | 0.758 | 0.816 | 0.892 | 0.73 | 0.27 | 12.0 | 4.8 | 4.0 | 8.2 | 4.2 | 13.0 | 18.0 |
| room0_topdown_q08_tol | 170 | 0.748 | 0.806 | 0.889 | 0.88 | 0.12 | 9.0 | 3.6 | 3.0 | 5.8 | 4.8 | 1.0 | 2.0 |
| room0_bottomup_q08_tol | 907 | 0.758 | 0.817 | 0.890 | 0.65 | 0.35 | 9.0 | 3.8 | 4.0 | 22.7 | 7.8 | 1.0 | 239.0 |
| 08:room1_v9_exterior | 75 | 0.746 | 0.789 | 0.927 | 0.86 | 0.14 | 3.0 | – | – | 2.3 | 2.0 | 2.0 | 3.0 |
| room1_both_q07_tol | 66 | 0.729 | 0.806 | 0.867 | 0.94 | 0.06 | 4.0 | – | – | 2.0 | 1.8 | 1.0 | 3.0 |
| room1_both_q08_tol | 121 | 0.742 | 0.817 | 0.875 | 0.84 | 0.16 | 8.0 | – | – | 2.3 | 2.0 | 26.0 | 16.0 |
| room1_both_q09_tol | 641 | 0.763 | 0.832 | 0.894 | 0.45 | 0.55 | 13.0 | – | – | 21.3 | 7.5 | 139.0 | 96.0 |
| room1_topdown_q08_tol | 218 | 0.744 | 0.819 | 0.877 | 0.82 | 0.18 | 11.0 | – | – | 3.2 | 3.5 | 84.0 | 35.0 |

- **q is the one knob** and trades parsimony for detail smoothly on compact shapes (office_0 plant 19 / 26 / 51 SQs
  at q 0.7 / 0.8 / 0.9, walls / lamps stay 1). **q = 0.7 is the best operating point** (new default in
  `decomp.py`): 66–117 SQs per room, 90–95 % plausible counts, and the room_1 rug / floor stay at 1 / 3 SQs. At
  q ≥ 0.8 flat ring-shaped instances explode (room_1 rug 26 at 0.8, 139 at 0.9).
- Merging after top-down matters (office_0 128 → 94 SQs, room_1 218 → 121 at the same quality).
- Mean strict vIoU barely moves between methods (0.73–0.83): it is dominated by the many small / flat instances and
  by amodal completion (08's lesson); the per-class K columns and the renders are the informative part.
- room_0's complex objects are small at 4 cm (plant 634 points): their tolerant fit passes early, so K grows only a
  little vs. 08 (plant 5 → 5–7). Complexity only earns many primitives when the object is also large enough to
  resolve — a finer voxel grid for small objects would be the natural next step.


## Files
- `code/partfit.py` GPU core (sets, sampling, loss, tolerant IoU, batched single-SQ fit). Also used by experiment 09.
- `code/decomp.py` decomposition methods, final refinement, scoring, saving (08's `sqs.npz` format + render).
- `code/convex.py` CoACD parts. `code/compare.py` per-class comparison table.
- `viewer/server.py` wrapper of 08's viewer (port 8010); `results/viewer_runs/` exported runs.
- `results/runs/<run>/{config.json, metrics.json, sqs.npz, render.jpg}`, `results/logs/` sweep logs.

    cd code && uv run python decomp.py --name <run> --set scene=room_1 [--set method=both --set q=0.8 ...]
    uv run python compare.py 08:office0_v9_exterior <run> ...
    uv run python ../../08_volumetric_decomp/code/export_viewer.py --exp 10 <run>    # -> results/viewer_runs, then viewer/server.py
